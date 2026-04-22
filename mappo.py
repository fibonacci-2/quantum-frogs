"""Stage 5: MAPPO — cooperative 2-frog (centralized critic, decentralised actors).

CTDE paradigm:
  - Training : centralised critic sees global state; both actors share team reward.
  - Execution: decentralised actors use only their own observation.
  - Communication arises from the shared value function that captures how
    joint behaviour affects joint returns, guiding both actors to cooperate.
"""

import argparse
import glob
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env import QuantumFrog2FrogEnv

ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)                       # logits

    def get_dist(self, x):
        return Categorical(logits=self.forward(x))


class CriticNet(nn.Module):
    """Centralised critic: global state → V(s).

    Global state = full grid observation (already includes both frog positions
    + all car positions/velocities), so no extra concatenation needed.
    """
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)           # (B,)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, device):
        T, N, D = rollout_steps, n_envs, obs_dim
        self.obs      = torch.zeros(T, N, D, device=device)
        self.act_a    = torch.zeros(T, N, dtype=torch.long, device=device)
        self.act_b    = torch.zeros(T, N, dtype=torch.long, device=device)
        self.logp_a   = torch.zeros(T, N, device=device)
        self.logp_b   = torch.zeros(T, N, device=device)
        self.reward   = torch.zeros(T, N, device=device)   # team reward
        self.done     = torch.zeros(T, N, device=device)
        self.value    = torch.zeros(T, N, device=device)
        self.adv      = torch.zeros(T, N, device=device)
        self.returns  = torch.zeros(T, N, device=device)
        self.T = T

    def compute_gae(self, last_value, gamma, gae_lambda):
        last_gae = torch.zeros_like(last_value)
        for t in reversed(range(self.T)):
            next_val   = last_value if t == self.T - 1 else self.value[t + 1]
            next_done  = self.done[t]
            delta      = self.reward[t] + gamma * next_val * (1.0 - next_done) - self.value[t]
            last_gae   = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
            self.adv[t] = last_gae
        self.returns = self.adv + self.value

    def flatten(self):
        T, N = self.T, self.obs.shape[1]
        def f(t): return t.view(T * N, *t.shape[2:])
        return (f(self.obs), f(self.act_a), f(self.act_b),
                f(self.logp_a), f(self.logp_b), f(self.adv), f(self.returns))


# ---------------------------------------------------------------------------
# MAPPO trainer
# ---------------------------------------------------------------------------

class MAPPOTrainer:
    def __init__(self, obs_dim, n_actions, device,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_param=0.2, ppo_epochs=4, mini_batch_size=512,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.device        = device
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_param    = clip_param
        self.ppo_epochs    = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor_a = ActorNet(obs_dim, n_actions).to(device)
        self.actor_b = ActorNet(obs_dim, n_actions).to(device)
        self.critic  = CriticNet(obs_dim).to(device)

        all_params = (list(self.actor_a.parameters())
                      + list(self.actor_b.parameters())
                      + list(self.critic.parameters()))
        self.optimizer = optim.Adam(all_params, lr=lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs_t, epsilon=0.0):
        """Epsilon-greedy during warm-up; otherwise sample from policy."""
        dist_a = self.actor_a.get_dist(obs_t)
        dist_b = self.actor_b.get_dist(obs_t)
        if epsilon > 0:
            n = obs_t.shape[0]
            mask = torch.rand(n, device=self.device) < epsilon
            act_a = dist_a.sample()
            act_b = dist_b.sample()
            rand_a = torch.randint(0, 5, (n,), device=self.device)
            rand_b = torch.randint(0, 5, (n,), device=self.device)
            act_a = torch.where(mask, rand_a, act_a)
            act_b = torch.where(mask, rand_b, act_b)
        else:
            act_a = dist_a.sample()
            act_b = dist_b.sample()
        logp_a = dist_a.log_prob(act_a)
        logp_b = dist_b.log_prob(act_b)
        return act_a, act_b, logp_a, logp_b

    @torch.no_grad()
    def value(self, obs_t):
        return self.critic(obs_t)

    @torch.no_grad()
    def greedy_act(self, obs_flat):
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_a = int(self.actor_a(obs_t).argmax(dim=-1).item())
        act_b = int(self.actor_b(obs_t).argmax(dim=-1).item())
        return act_a, act_b

    def update(self, buf):
        obs, act_a, act_b, old_lp_a, old_lp_b, adv, returns = buf.flatten()

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.shape[0]
        actor_losses, critic_losses, ent_losses = [], [], []

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, self.mini_batch_size):
                mb = idx[start:start + self.mini_batch_size]
                o  = obs[mb];    aa = act_a[mb];    ab = act_b[mb]
                la = old_lp_a[mb]; lb = old_lp_b[mb]
                adv_mb = adv[mb]; ret_mb = returns[mb]

                dist_a = self.actor_a.get_dist(o)
                dist_b = self.actor_b.get_dist(o)
                new_lp_a = dist_a.log_prob(aa)
                new_lp_b = dist_b.log_prob(ab)
                ent_a = dist_a.entropy().mean()
                ent_b = dist_b.entropy().mean()

                def ppo_loss(new_lp, old_lp):
                    ratio = torch.exp(new_lp - old_lp)
                    s1 = ratio * adv_mb
                    s2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_mb
                    return -torch.min(s1, s2).mean()

                actor_loss  = ppo_loss(new_lp_a, la) + ppo_loss(new_lp_b, lb)
                v_pred      = self.critic(o)
                critic_loss = nn.functional.mse_loss(v_pred, ret_mb)
                entropy     = ent_a + ent_b

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor_a.parameters())
                    + list(self.actor_b.parameters())
                    + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                ent_losses.append(entropy.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(ent_losses)

    def save(self, path):
        torch.save({
            "actor_a": self.actor_a.state_dict(),
            "actor_b": self.actor_b.state_dict(),
            "critic":  self.critic.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_a.load_state_dict(ckpt["actor_a"])
        self.actor_b.load_state_dict(ckpt["actor_b"])
        self.critic.load_state_dict(ckpt["critic"])


# ---------------------------------------------------------------------------
# Training loop (vectorised envs)
# ---------------------------------------------------------------------------

def train_mappo(total_timesteps, n_envs=32, num_cars=4, car_speeds=(1, 2),
                device="cpu", seed=0, show_progress=True,
                rollout_steps=128, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_param=0.2, ppo_epochs=4, mini_batch_size=512,
                value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    envs = [QuantumFrog2FrogEnv(num_cars=num_cars, car_speeds=car_speeds) for _ in range(n_envs)]
    obs_dim = int(np.prod(envs[0].observation_space.shape))

    trainer = MAPPOTrainer(obs_dim, 5, device, lr=lr, gamma=gamma,
                           gae_lambda=gae_lambda, clip_param=clip_param,
                           ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
                           value_coef=value_coef, entropy_coef=entropy_coef,
                           max_grad_norm=max_grad_norm)

    buf = RolloutBuffer(rollout_steps, n_envs, obs_dim, device)

    obs_np = np.zeros((n_envs, obs_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs_np[i] = o.flatten()

    ep_rewards, ep_lengths, ep_wins = [], [], []
    ep_reward_a, ep_reward_b = [], []
    running_r   = np.zeros(n_envs)
    running_ra  = np.zeros(n_envs)
    running_rb  = np.zeros(n_envs)
    running_len = np.zeros(n_envs, dtype=int)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_timesteps, disable=not show_progress, unit="step")
    except ImportError:
        pbar = None

    t_collected = 0

    while t_collected < total_timesteps:
        # ---- collect rollout ----
        for step in range(rollout_steps):
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            act_a_t, act_b_t, lp_a_t, lp_b_t = trainer.act(obs_t)
            val_t = trainer.value(obs_t)

            buf.obs[step]    = obs_t
            buf.act_a[step]  = act_a_t
            buf.act_b[step]  = act_b_t
            buf.logp_a[step] = lp_a_t
            buf.logp_b[step] = lp_b_t
            buf.value[step]  = val_t

            act_a_np = act_a_t.cpu().numpy()
            act_b_np = act_b_t.cpu().numpy()

            team_r = np.zeros(n_envs)
            dones  = np.zeros(n_envs)

            for i, env in enumerate(envs):
                o, _, term, trunc, info = env.step([act_a_np[i], act_b_np[i]])
                ra = info["reward_a"]
                rb = info["reward_b"]
                done = term or trunc
                team_r[i] = ra + rb
                dones[i]  = float(done)

                running_r[i]   += ra + rb
                running_ra[i]  += ra
                running_rb[i]  += rb
                running_len[i] += 1

                if done:
                    both_win = info.get("both_reached_top", False)
                    a_done   = info.get("frog_a_done", False)
                    b_done   = info.get("frog_b_done", False)
                    win_code = 2 if both_win else (1 if (a_done or b_done) else 0)
                    ep_rewards.append(running_r[i])
                    ep_lengths.append(running_len[i])
                    ep_wins.append(win_code)
                    ep_reward_a.append(running_ra[i])
                    ep_reward_b.append(running_rb[i])
                    running_r[i] = running_ra[i] = running_rb[i] = 0.0
                    running_len[i] = 0
                    o, _ = env.reset()

                obs_np[i] = o.flatten()

            buf.reward[step] = torch.tensor(team_r, dtype=torch.float32, device=device)
            buf.done[step]   = torch.tensor(dones,  dtype=torch.float32, device=device)

        # bootstrap value for last step
        with torch.no_grad():
            last_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            last_val = trainer.value(last_obs)
        buf.compute_gae(last_val, gamma, gae_lambda)

        trainer.update(buf)

        t_collected += rollout_steps * n_envs
        if pbar:
            pbar.update(rollout_steps * n_envs)

    if pbar:
        pbar.close()

    return trainer, {
        "ep_rewards":  ep_rewards,
        "ep_lengths":  ep_lengths,
        "ep_wins":     ep_wins,
        "ep_reward_a": ep_reward_a,
        "ep_reward_b": ep_reward_b,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_across_car_counts(trainer, car_counts, num_episodes=200,
                                car_speeds=(1, 2), max_steps=200):
    results = {}
    for nc in car_counts:
        env = QuantumFrog2FrogEnv(num_cars=nc, car_speeds=car_speeds)
        both_wins = a_wins = b_wins = 0
        ep_steps = []
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=ep)
            obs_flat = obs.flatten().astype(np.float32)
            done, steps = False, 0
            while not done and steps < max_steps:
                act_a, act_b = trainer.greedy_act(obs_flat)
                obs, _, term, trunc, info = env.step([act_a, act_b])
                obs_flat = obs.flatten().astype(np.float32)
                done = term or trunc
                steps += 1
            ep_steps.append(steps)
            if info.get("both_reached_top", False): both_wins += 1
            if info.get("frog_a_done", False):      a_wins += 1
            if info.get("frog_b_done", False):      b_wins += 1
        results[nc] = {
            "both_win_rate": both_wins / num_episodes,
            "a_win_rate":    a_wins    / num_episodes,
            "b_win_rate":    b_wins    / num_episodes,
            "avg_steps":     sum(ep_steps) / num_episodes,
            "steps_list":    ep_steps,
        }
        print(
            f"  cars={nc}: both={results[nc]['both_win_rate']*100:.1f}%"
            f"  A={results[nc]['a_win_rate']*100:.1f}%"
            f"  B={results[nc]['b_win_rate']*100:.1f}%"
            f"  avg_steps={results[nc]['avg_steps']:.1f}"
        )
    return results


def collect_action_distributions(trainer, num_cars, key_episodes,
                                  car_speeds=(1, 2), max_steps=200):
    env = QuantumFrog2FrogEnv(num_cars=num_cars, car_speeds=car_speeds)
    out = {}
    for ep_seed in key_episodes:
        obs, _ = env.reset(seed=ep_seed)
        obs_flat = obs.flatten().astype(np.float32)
        done, steps = False, 0
        ca, cb = np.zeros(5, dtype=int), np.zeros(5, dtype=int)
        while not done and steps < max_steps:
            aa, ab = trainer.greedy_act(obs_flat)
            ca[aa] += 1; cb[ab] += 1
            obs, _, term, trunc, info = env.step([aa, ab])
            obs_flat = obs.flatten().astype(np.float32)
            done = term or trunc; steps += 1
        learned_steps = steps
        learned_win   = bool(info.get("both_reached_top", False))

        obs, _ = env.reset(seed=ep_seed)
        done, steps = False, 0
        rca, rcb = np.zeros(5, dtype=int), np.zeros(5, dtype=int)
        while not done and steps < max_steps:
            j = env.action_space.sample()
            rca[j[0]] += 1; rcb[j[1]] += 1
            obs, _, term, trunc, info = env.step(j)
            done = term or trunc; steps += 1
        random_steps = steps
        random_win   = bool(info.get("both_reached_top", False))

        def p(c): return (c / max(1, c.sum())).tolist()
        pa, pb, rpa, rpb = p(ca), p(cb), p(rca), p(rcb)
        out[str(ep_seed)] = {
            "a_counts": ca.tolist(), "a_probs": pa,
            "b_counts": cb.tolist(), "b_probs": pb,
            "rand_a_probs": rpa, "rand_b_probs": rpb,
            "l1_a": float(np.abs(np.array(pa) - np.array(rpa)).sum()),
            "l1_b": float(np.abs(np.array(pb) - np.array(rpb)).sum()),
            "learned_steps": learned_steps, "learned_win": learned_win,
            "random_steps": random_steps,   "random_win":  random_win,
        }
    return out


# ---------------------------------------------------------------------------
# Comparison tables (Stage 4 IDQN vs Stage 5 MAPPO)
# ---------------------------------------------------------------------------

def load_stage4_means(runs_root, car_counts):
    dirs = sorted(glob.glob(os.path.join(runs_root, "idqn", "idqn_*_gpu*")))
    if not dirs:
        return None
    both = {c: [] for c in car_counts}
    a_r  = {c: [] for c in car_counts}
    b_r  = {c: [] for c in car_counts}
    stp  = {c: [] for c in car_counts}
    for d in dirs:
        p = os.path.join(d, "eval_results.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            ev = json.load(f)
        for c in car_counts:
            if str(c) not in ev:
                continue
            both[c].append(ev[str(c)]["both_win_rate"] * 100)
            a_r[c].append(ev[str(c)]["a_win_rate"]    * 100)
            b_r[c].append(ev[str(c)]["b_win_rate"]    * 100)
            stp[c].append(ev[str(c)]["avg_steps"])
    return {c: {"both": np.mean(both[c]) if both[c] else float("nan"),
                "a":    np.mean(a_r[c])  if a_r[c]  else float("nan"),
                "b":    np.mean(b_r[c])  if b_r[c]  else float("nan"),
                "steps":np.mean(stp[c])  if stp[c]  else float("nan")}
            for c in car_counts}


def print_comparison_tables(stage4, stage5_eval, car_counts, stage5_seeds):
    sep  = "+" + "-" * 8 + ("+" + "-" * 16) * len(car_counts) + "+"
    hdr  = f"| {'Cars':>6} |" + "".join(f"  {c} car{'s' if c>1 else ' '}      |" for c in car_counts)

    def row(label, vals, fmt="{:>6.1f}%"):
        return f"| {label:>6} |" + "".join(fmt.format(v) + "         |" for v in vals)

    print("\n" + "=" * 70)
    print("  COOPERATION GAP — Stage 4 (IDQN) vs Stage 5 (MAPPO)")
    print("=" * 70)

    # ── Both-win rate ──
    print("\n  BOTH FROGS WIN RATE  (mean over seeds, 200 eval episodes each)")
    print(sep); print(hdr); print(sep)
    if stage4:
        s4 = [stage4[c]["both"] for c in car_counts]
        print(row("IDQN", s4))
    s5 = [stage5_eval[c]["both_win_rate"] * 100 for c in car_counts]
    print(row("MAPPO", s5))
    if stage4:
        delta = [s5[i] - s4[i] for i in range(len(car_counts))]
        print(row("Δ MAPPO-IDQN", delta, fmt="{:>+6.1f}%"))
    print(sep)

    # ── Per-agent ──
    print("\n  INDIVIDUAL AGENT SUCCESS  (MAPPO mean, 200 eval episodes)")
    print(sep); print(hdr); print(sep)
    a_vals = [stage5_eval[c]["a_win_rate"] * 100 for c in car_counts]
    b_vals = [stage5_eval[c]["b_win_rate"] * 100 for c in car_counts]
    print(row("Frog A", a_vals))
    print(row("Frog B", b_vals))
    print(sep)

    # ── Avg steps ──
    print("\n  AVG EPISODE LENGTH (steps)")
    sep2 = "+" + "-" * 8 + ("+" + "-" * 14) * len(car_counts) + "+"
    hdr2 = f"| {'Cars':>6} |" + "".join(f"  {c} car{'s' if c>1 else ' '}    |" for c in car_counts)
    print(sep2); print(hdr2); print(sep2)
    if stage4:
        s4s = [stage4[c]["steps"] for c in car_counts]
        print(f"| {'IDQN':>6} |" + "".join(f"  {v:>6.1f}     |" for v in s4s))
    s5s = [stage5_eval[c]["avg_steps"] for c in car_counts]
    print(f"| {'MAPPO':>6} |" + "".join(f"  {v:>6.1f}     |" for v in s5s))
    print(sep2)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_report(run_dir, params, train_stats, eval_results, action_comparison,
                    stage4_means, car_counts):
    ep_rewards = train_stats["ep_rewards"]
    ep_wins    = train_stats["ep_wins"]
    ep_ra      = train_stats["ep_reward_a"]
    ep_rb      = train_stats["ep_reward_b"]

    episodes = list(range(1, len(ep_rewards) + 1))
    window   = max(20, min(500, max(20, len(ep_rewards) // 20)))

    def rolling(data, w):
        if len(data) < w:
            return [None] * len(data)
        arr = np.array(data, dtype=float)
        cs  = np.cumsum(arr)
        cs[w:] = cs[w:] - cs[:-w]
        return [None] * (w - 1) + (cs[w - 1:] / w).tolist()

    smooth_r  = rolling(ep_rewards, window)
    smooth_ra = rolling(ep_ra, window)
    smooth_rb = rolling(ep_rb, window)
    win_vals  = [1.0 if w == 2 else (0.5 if w == 1 else 0.0) for w in ep_wins]
    smooth_w  = [v * 100 if v is not None else None for v in rolling(win_vals, window)]

    both_rates = [eval_results[c]["both_win_rate"] * 100 for c in car_counts]
    a_rates    = [eval_results[c]["a_win_rate"]    * 100 for c in car_counts]
    b_rates    = [eval_results[c]["b_win_rate"]    * 100 for c in car_counts]
    avg_steps  = [eval_results[c]["avg_steps"]          for c in car_counts]

    s4_both = [stage4_means[c]["both"] if stage4_means else None for c in car_counts] if stage4_means else []
    s4_steps= [stage4_means[c]["steps"] if stage4_means else None for c in car_counts] if stage4_means else []

    sorted_ak = sorted(action_comparison.keys(), key=lambda x: int(x))
    l1_a = [action_comparison[k]["l1_a"] for k in sorted_ak]
    l1_b = [action_comparison[k]["l1_b"] for k in sorted_ak]
    params_html = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in params.items())

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>MAPPO Report — {params['run_id']}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;background:#fafafa}}
  h1{{color:#1a1a2e}} h2{{color:#16213e;border-bottom:2px solid #0f3460;padding-bottom:4px}}
  table.params{{border-collapse:collapse;margin:10px 0 30px}}
  table.params td{{padding:4px 16px;border:1px solid #ccc}}
  .plot{{margin:20px 0 40px}}
</style>
</head><body>
<h1>🐸🐸 Quantum Frog — Stage 5 MAPPO Report</h1>
<h2>Run Parameters</h2>
<table class="params">{params_html}</table>

<h2>1. Training Curves</h2>
<p>Rolling average window = {window} episodes.</p>
<div id="tr" class="plot"></div>
<div id="ta" class="plot"></div>
<div id="tw" class="plot"></div>

<h2>2. Difficulty Curve — Stage 4 IDQN vs Stage 5 MAPPO</h2>
<div id="dc" class="plot"></div>

<h2>3. Individual Success Rates</h2>
<div id="ind" class="plot"></div>

<h2>4. Average Episode Length</h2>
<div id="stp" class="plot"></div>

<h2>5. Episode Length Distribution</h2>
<div id="box" class="plot"></div>

<h2>6. Action Distribution vs Random</h2>
<label for="agSel"><b>Agent:</b></label>
<select id="agSel"><option value="a">Frog A</option><option value="b">Frog B</option></select>
&nbsp;
<label for="epSel"><b>Episode seed:</b></label>
<select id="epSel"></select>
<div id="ac" class="plot"></div>
<div id="l1" class="plot"></div>

<script>
const epIdx={json.dumps(episodes)},smoothR={json.dumps(smooth_r)},smoothRA={json.dumps(smooth_ra)};
const smoothRB={json.dumps(smooth_rb)},smoothW={json.dumps(smooth_w)};
const cars={json.dumps(car_counts)},bothR={json.dumps(both_rates)},aR={json.dumps(a_rates)},bR={json.dumps(b_rates)};
const avgS={json.dumps(avg_steps)},s4Both={json.dumps(s4_both)},s4Steps={json.dumps(s4_steps)};
const sD={json.dumps({str(c): eval_results[c]['steps_list'] for c in car_counts})};
const aLab={json.dumps(ACTION_LABELS)},aC={json.dumps(action_comparison)},aK={json.dumps(sorted_ak)};
const l1A={json.dumps(l1_a)},l1B={json.dumps(l1_b)};

Plotly.newPlot('tr',[{{x:epIdx,y:smoothR,type:'scatter',mode:'lines',name:'Total reward',line:{{color:'#0f3460'}}}}],
  {{title:'Rolling Avg Total Reward',xaxis:{{title:'Episode'}},yaxis:{{title:'Reward'}}}},{{responsive:true}});

Plotly.newPlot('ta',[
  {{x:epIdx,y:smoothRA,type:'scatter',mode:'lines',name:'Frog A',line:{{color:'#e94560'}}}},
  {{x:epIdx,y:smoothRB,type:'scatter',mode:'lines',name:'Frog B',line:{{color:'#533483'}}}}
],{{title:'Per-Agent Rolling Avg Reward',xaxis:{{title:'Episode'}},yaxis:{{title:'Reward'}}}},{{responsive:true}});

Plotly.newPlot('tw',[{{x:epIdx,y:smoothW,type:'scatter',mode:'lines',name:'Both-win %',line:{{color:'#e94560'}}}}],
  {{title:'Rolling Both-Win Rate %',xaxis:{{title:'Episode'}},yaxis:{{title:'%',range:[0,100]}}}},{{responsive:true}});

const dcTraces=[
  {{x:cars,y:bothR,type:'scatter',mode:'lines+markers',name:'MAPPO both-win',marker:{{size:10,color:'#0f3460'}}}}
];
if(s4Both.length)dcTraces.push({{x:cars,y:s4Both,type:'scatter',mode:'lines+markers',name:'IDQN both-win (S4)',
  line:{{dash:'dash'}},marker:{{size:8,color:'#e94560'}}}});
Plotly.newPlot('dc',dcTraces,{{title:'Both-Win Rate vs Car Count',xaxis:{{title:'Cars',dtick:1}},yaxis:{{title:'%',range:[0,105]}}}},{{responsive:true}});

Plotly.newPlot('ind',[
  {{x:cars,y:aR,type:'scatter',mode:'lines+markers',name:'Frog A',marker:{{size:9,color:'#e94560'}}}},
  {{x:cars,y:bR,type:'scatter',mode:'lines+markers',name:'Frog B',marker:{{size:9,color:'#533483'}}}}
],{{title:'Individual Success Rate vs Car Count',xaxis:{{title:'Cars',dtick:1}},yaxis:{{title:'%',range:[0,105]}}}},{{responsive:true}});

const stpT=[{{x:cars,y:avgS,type:'bar',name:'MAPPO',marker:{{color:'#0f3460'}}}}];
if(s4Steps.length)stpT.push({{x:cars,y:s4Steps,type:'bar',name:'IDQN',marker:{{color:'#e94560'}}}});
Plotly.newPlot('stp',stpT,{{title:'Avg Episode Length vs Car Count',barmode:'group',xaxis:{{title:'Cars',dtick:1}},yaxis:{{title:'Steps'}}}},{{responsive:true}});

Plotly.newPlot('box',cars.map(c=>({{y:sD[String(c)],type:'box',name:c+' cars'}})),
  {{title:'Episode Length Distribution',yaxis:{{title:'Steps'}}}},{{responsive:true}});

function renderAC(){{
  const ag=document.getElementById('agSel').value,ek=document.getElementById('epSel').value,d=aC[ek];
  const lp=(ag==='a'?d.a_probs:d.b_probs).map(v=>v*100),rp=(ag==='a'?d.rand_a_probs:d.rand_b_probs).map(v=>v*100);
  const l1=ag==='a'?d.l1_a:d.l1_b;
  Plotly.newPlot('ac',[
    {{x:aLab,y:lp,type:'bar',name:'MAPPO greedy',marker:{{color:'#0f3460'}}}},
    {{x:aLab,y:rp,type:'bar',name:'Random',marker:{{color:'#e94560'}}}}
  ],{{title:`Frog ${{ag.toUpperCase()}} seed ${{ek}} L1=${{l1.toFixed(3)}}`,barmode:'group',yaxis:{{title:'%',range:[0,100]}},
    annotations:[{{text:`Learned: steps=${{d.learned_steps}} win=${{d.learned_win}} | Random: steps=${{d.random_steps}} win=${{d.random_win}}`,
      showarrow:false,xref:'paper',yref:'paper',x:0,y:1.12,xanchor:'left'}}]}},{{responsive:true}});
  Plotly.newPlot('l1',[
    {{x:aK.map(Number),y:l1A,type:'scatter',mode:'lines+markers',name:'Frog A',marker:{{size:9,color:'#e94560'}}}},
    {{x:aK.map(Number),y:l1B,type:'scatter',mode:'lines+markers',name:'Frog B',marker:{{size:9,color:'#533483'}}}}
  ],{{title:'L1 Dist from Random',xaxis:{{title:'Episode seed'}},yaxis:{{title:'L1',range:[0,2]}}}},{{responsive:true}});
}}
const sel=document.getElementById('epSel');
aK.forEach(k=>{{const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o);}});
sel.addEventListener('change',renderAC);document.getElementById('agSel').addEventListener('change',renderAC);
if(aK.length){{sel.value=aK[0];renderAC();}}
</script></body></html>"""

    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report saved: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",     type=int,   default=300_000)
    parser.add_argument("--train-cars",    type=int,   default=4)
    parser.add_argument("--eval-episodes", type=int,   default=200)
    parser.add_argument("--n-envs",        type=int,   default=32)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--gpu-id",        type=int,   default=None)
    parser.add_argument("--run-tag",       type=str,   default="")
    parser.add_argument("--no-progress",   action="store_true")
    parser.add_argument("--cpu",           action="store_true")
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.timesteps
    TRAIN_CARS      = args.train_cars
    CAR_SPEEDS      = (1, 2)
    EVAL_CAR_COUNTS = [1, 2, 3, 4, 5, 6]
    EVAL_EPISODES   = args.eval_episodes
    KEY_EPISODES    = [1, 500, 1000]

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.cpu:
        device = "cpu"
    elif args.gpu_id is not None:
        device = "cuda"
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_id  = datetime.now().strftime("mappo_%Y%m%d_%H%M%S") + run_suffix
    run_dir = os.path.join("runs", "mappo", run_id)
    os.makedirs(run_dir, exist_ok=True)

    params = {
        "run_id":         run_id,
        "timestamp":      datetime.now().isoformat(),
        "algorithm":      "MAPPO (centralised critic, decentralised actors)",
        "stage":          5,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_envs":         args.n_envs,
        "train_cars":     TRAIN_CARS,
        "car_speeds":     list(CAR_SPEEDS),
        "seed":           args.seed,
        "device":         device,
        "gpu_id":         args.gpu_id,
        "grid_size":      8,
        "num_agents":     2,
        "policy":         "Actor-Critic (CTDE: centralised V, decentralised π)",
        "eval_car_counts": EVAL_CAR_COUNTS,
        "eval_episodes":  EVAL_EPISODES,
        "key_episodes":   KEY_EPISODES,
    }

    print(f"=== Run: {run_id} ===")
    print(f"Output: {run_dir}/  n_envs={args.n_envs}\n")

    print("Training MAPPO (centralised critic)...")
    t0 = time.time()
    trainer, train_stats = train_mappo(
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=args.n_envs,
        num_cars=TRAIN_CARS,
        car_speeds=CAR_SPEEDS,
        device=device,
        seed=args.seed,
        show_progress=(not args.no_progress),
    )
    train_time = time.time() - t0
    params["train_time_sec"]   = round(train_time, 1)
    params["episodes_recorded"] = len(train_stats["ep_rewards"])

    trainer.save(os.path.join(run_dir, f"mappo_stage5_cars{TRAIN_CARS}_{TOTAL_TIMESTEPS}steps.pt"))
    np.savez(
        os.path.join(run_dir, f"training_curves_cars{TRAIN_CARS}_{TOTAL_TIMESTEPS}steps.npz"),
        rewards=np.array(train_stats["ep_rewards"]),
        wins=np.array(train_stats["ep_wins"]),
        lengths=np.array(train_stats["ep_lengths"]),
        reward_a=np.array(train_stats["ep_reward_a"]),
        reward_b=np.array(train_stats["ep_reward_b"]),
    )
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("\nEvaluating across car counts...")
    eval_results = evaluate_across_car_counts(
        trainer, EVAL_CAR_COUNTS, num_episodes=EVAL_EPISODES, car_speeds=CAR_SPEEDS
    )
    with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
        json.dump({str(k): v for k, v in eval_results.items()}, f, indent=2)

    print("\nCollecting action-distribution evidence...")
    action_comparison = collect_action_distributions(
        trainer, TRAIN_CARS, KEY_EPISODES, car_speeds=CAR_SPEEDS
    )
    with open(os.path.join(run_dir, "action_distribution_key_episodes.json"), "w") as f:
        json.dump(action_comparison, f, indent=2)

    runs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    stage4_means = load_stage4_means(runs_root, EVAL_CAR_COUNTS)

    print("\nGenerating report...")
    generate_report(run_dir, params, train_stats, eval_results, action_comparison,
                    stage4_means, EVAL_CAR_COUNTS)

    print(f"\nDone in {train_time:.0f}s.  Files in {run_dir}/")

    print_comparison_tables(stage4_means, eval_results, EVAL_CAR_COUNTS, [args.seed])
