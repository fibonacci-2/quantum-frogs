"""Stage 4: Independent DQN on 2-frog setting (2 cars, speed 1, no cooperation)."""

import argparse
import json
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import QuantumFrog2FrogEnv

ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, float(reward), next_obs, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(act, dtype=np.int64),
            np.array(rew, dtype=np.float32),
            np.array(nobs, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, obs_dim, n_actions, device, lr=1e-3, gamma=0.99,
                 buffer_size=100_000, batch_size=128):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        self.q_net = QNet(obs_dim, n_actions).to(device)
        self.target_net = QNet(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, obs_flat, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_net(obs_t).argmax(dim=1).item())

    def select_action_batch(self, obs_batch, epsilon):
        """obs_batch: (N, obs_dim) numpy array. Returns (N,) int array."""
        n = len(obs_batch)
        mask = np.random.random(n) < epsilon
        actions = np.random.randint(0, self.n_actions, size=n)
        if not mask.all():
            greedy_idx = np.where(~mask)[0]
            obs_t = torch.tensor(obs_batch[greedy_idx], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions[greedy_idx] = self.q_net(obs_t).argmax(dim=1).cpu().numpy()
        return actions

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)
        obs_t = torch.tensor(obs, device=self.device)
        act_t = torch.tensor(act, device=self.device).unsqueeze(1)
        rew_t = torch.tensor(rew, device=self.device)
        nobs_t = torch.tensor(nobs, device=self.device)
        done_t = torch.tensor(done, device=self.device)

        with torch.no_grad():
            max_q_next = self.target_net(nobs_t).max(dim=1).values
            td_target = rew_t + self.gamma * max_q_next * (1.0 - done_t)

        q_vals = self.q_net(obs_t).gather(1, act_t).squeeze(1)
        loss = nn.functional.mse_loss(q_vals, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_idqn(total_timesteps, num_cars=2, car_speeds=(1,), device="cpu", seed=0,
               show_progress=True, lr=1e-3, buffer_size=100_000, batch_size=128,
               learning_starts=5000, gamma=0.99, train_freq=4,
               target_update_interval=1000, exploration_fraction=0.3,
               eps_start=1.0, eps_end=0.05):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = QuantumFrog2FrogEnv(num_cars=num_cars, car_speeds=car_speeds)
    obs_dim = int(np.prod(env.observation_space.shape))  # 3*8*8 = 192

    agent_a = DQNAgent(obs_dim, 5, device, lr=lr, gamma=gamma,
                       buffer_size=buffer_size, batch_size=batch_size)
    agent_b = DQNAgent(obs_dim, 5, device, lr=lr, gamma=gamma,
                       buffer_size=buffer_size, batch_size=batch_size)

    ep_rewards, ep_lengths, ep_wins = [], [], []
    ep_reward_a, ep_reward_b = [], []

    obs, info = env.reset(seed=seed)
    obs_flat = obs.flatten().astype(np.float32)
    ep_r = ep_ra = ep_rb = 0.0
    ep_len = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_timesteps, disable=not show_progress, unit="step")
    except ImportError:
        pbar = None

    for t_step in range(total_timesteps):
        frac = min(1.0, t_step / max(1, exploration_fraction * total_timesteps))
        epsilon = eps_start + frac * (eps_end - eps_start)

        if t_step < learning_starts:
            joint = env.action_space.sample()
            action_a, action_b = int(joint[0]), int(joint[1])
        else:
            action_a = agent_a.select_action(obs_flat, epsilon)
            action_b = agent_b.select_action(obs_flat, epsilon)

        next_obs, reward, terminated, truncated, info = env.step([action_a, action_b])
        next_obs_flat = next_obs.flatten().astype(np.float32)
        done = terminated or truncated

        reward_a = info["reward_a"]
        reward_b = info["reward_b"]

        agent_a.store(obs_flat, action_a, reward_a, next_obs_flat, float(done))
        agent_b.store(obs_flat, action_b, reward_b, next_obs_flat, float(done))

        if t_step >= learning_starts and t_step % train_freq == 0:
            agent_a.train_step()
            agent_b.train_step()

        if t_step >= learning_starts and t_step % target_update_interval == 0:
            agent_a.update_target()
            agent_b.update_target()

        obs_flat = next_obs_flat
        ep_r += reward
        ep_ra += reward_a
        ep_rb += reward_b
        ep_len += 1

        if pbar:
            pbar.update(1)

        if done:
            both_win = info.get("both_reached_top", False)
            a_done = info.get("frog_a_done", False)
            b_done = info.get("frog_b_done", False)
            win_code = 2 if both_win else (1 if (a_done or b_done) else 0)
            ep_rewards.append(ep_r)
            ep_lengths.append(ep_len)
            ep_wins.append(win_code)
            ep_reward_a.append(ep_ra)
            ep_reward_b.append(ep_rb)

            obs, info = env.reset()
            obs_flat = obs.flatten().astype(np.float32)
            ep_r = ep_ra = ep_rb = 0.0
            ep_len = 0

    if pbar:
        pbar.close()

    return agent_a, agent_b, {
        "ep_rewards": ep_rewards,
        "ep_lengths": ep_lengths,
        "ep_wins": ep_wins,
        "ep_reward_a": ep_reward_a,
        "ep_reward_b": ep_reward_b,
    }


# ---------------------------------------------------------------------------
# Vectorized training (multiple parallel envs per GPU — better GPU util)
# ---------------------------------------------------------------------------

def train_idqn_vec(total_timesteps, n_envs=32, num_cars=2, car_speeds=(1,),
                   device="cpu", seed=0, show_progress=True,
                   lr=1e-3, buffer_size=200_000, batch_size=1024,
                   learning_starts=5000, gamma=0.99, train_freq=1,
                   target_update_interval=500, exploration_fraction=0.3,
                   eps_start=1.0, eps_end=0.05):
    """Run n_envs environments in parallel (sequential Python, no subprocess overhead).

    Collects n_envs transitions per loop iteration, trains both agents every
    train_freq iterations with batch_size >= 1024.  GPU utilisation is ~5x
    higher than the single-env version because the batch fills GPU memory
    bandwidth more efficiently.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    envs = [QuantumFrog2FrogEnv(num_cars=num_cars, car_speeds=car_speeds) for _ in range(n_envs)]
    obs_dim = int(np.prod(envs[0].observation_space.shape))

    agent_a = DQNAgent(obs_dim, 5, device, lr=lr, gamma=gamma,
                       buffer_size=buffer_size, batch_size=batch_size)
    agent_b = DQNAgent(obs_dim, 5, device, lr=lr, gamma=gamma,
                       buffer_size=buffer_size, batch_size=batch_size)

    # Reset all envs
    obs_flat = np.zeros((n_envs, obs_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs_flat[i] = o.flatten()

    ep_rewards, ep_lengths, ep_wins, ep_reward_a, ep_reward_b = [], [], [], [], []
    running_r  = np.zeros(n_envs)
    running_ra = np.zeros(n_envs)
    running_rb = np.zeros(n_envs)
    running_len = np.zeros(n_envs, dtype=int)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_timesteps, disable=not show_progress, unit="step")
    except ImportError:
        pbar = None

    t_collected = 0
    t_iter = 0
    while t_collected < total_timesteps:
        frac = min(1.0, t_collected / max(1, exploration_fraction * total_timesteps))
        epsilon = eps_start + frac * (eps_end - eps_start)

        if t_collected < learning_starts:
            act_a = np.array([env.action_space.sample()[0] for env in envs])
            act_b = np.array([env.action_space.sample()[1] for env in envs])
        else:
            act_a = agent_a.select_action_batch(obs_flat, epsilon)
            act_b = agent_b.select_action_batch(obs_flat, epsilon)

        for i, env in enumerate(envs):
            next_o, _, terminated, truncated, info = env.step([act_a[i], act_b[i]])
            next_flat = next_o.flatten().astype(np.float32)
            done = terminated or truncated
            ra = info["reward_a"]
            rb = info["reward_b"]

            agent_a.store(obs_flat[i], act_a[i], ra, next_flat, float(done))
            agent_b.store(obs_flat[i], act_b[i], rb, next_flat, float(done))

            running_r[i]  += ra + rb
            running_ra[i] += ra
            running_rb[i] += rb
            running_len[i] += 1

            if done:
                both_win = info.get("both_reached_top", False)
                a_done   = info.get("frog_a_done", False)
                b_done   = info.get("frog_b_done", False)
                win_code = 2 if both_win else (1 if (a_done or b_done) else 0)
                ep_rewards.append(running_r[i]);  ep_lengths.append(running_len[i])
                ep_wins.append(win_code);         ep_reward_a.append(running_ra[i])
                ep_reward_b.append(running_rb[i])
                running_r[i] = running_ra[i] = running_rb[i] = 0.0
                running_len[i] = 0
                o, _ = env.reset()
                next_flat = o.flatten().astype(np.float32)

            obs_flat[i] = next_flat

        t_collected += n_envs
        t_iter += 1
        if pbar:
            pbar.update(n_envs)

        if t_collected >= learning_starts and t_iter % train_freq == 0:
            agent_a.train_step()
            agent_b.train_step()

        if t_collected >= learning_starts and t_iter % target_update_interval == 0:
            agent_a.update_target()
            agent_b.update_target()

    if pbar:
        pbar.close()

    return agent_a, agent_b, {
        "ep_rewards": ep_rewards, "ep_lengths": ep_lengths,
        "ep_wins": ep_wins, "ep_reward_a": ep_reward_a, "ep_reward_b": ep_reward_b,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_across_car_counts(agent_a, agent_b, car_counts, num_episodes=200,
                                car_speeds=(1,), max_steps=200):
    results = {}
    for nc in car_counts:
        env = QuantumFrog2FrogEnv(num_cars=nc, car_speeds=car_speeds)
        both_wins = a_wins = b_wins = 0
        ep_steps = []
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep)
            obs_flat = obs.flatten().astype(np.float32)
            done, steps = False, 0
            while not done and steps < max_steps:
                action_a_val = agent_a.select_action(obs_flat, epsilon=0.0)
                action_b_val = agent_b.select_action(obs_flat, epsilon=0.0)
                obs, _, terminated, truncated, info = env.step([action_a_val, action_b_val])
                obs_flat = obs.flatten().astype(np.float32)
                done = terminated or truncated
                steps += 1
            ep_steps.append(steps)
            if info.get("both_reached_top", False):
                both_wins += 1
            if info.get("frog_a_done", False):
                a_wins += 1
            if info.get("frog_b_done", False):
                b_wins += 1
        results[nc] = {
            "both_win_rate": both_wins / num_episodes,
            "a_win_rate": a_wins / num_episodes,
            "b_win_rate": b_wins / num_episodes,
            "avg_steps": sum(ep_steps) / num_episodes,
            "steps_list": ep_steps,
        }
        print(
            f"  cars={nc}: both={results[nc]['both_win_rate']*100:.1f}%"
            f"  A={results[nc]['a_win_rate']*100:.1f}%"
            f"  B={results[nc]['b_win_rate']*100:.1f}%"
            f"  avg_steps={results[nc]['avg_steps']:.1f}"
        )
    return results


def collect_action_distributions(agent_a, agent_b, num_cars, key_episodes,
                                  car_speeds=(1,), max_steps=200):
    env = QuantumFrog2FrogEnv(num_cars=num_cars, car_speeds=car_speeds)
    out = {}

    for ep_seed in key_episodes:
        # Learned agents
        obs, info = env.reset(seed=ep_seed)
        obs_flat = obs.flatten().astype(np.float32)
        done, steps = False, 0
        counts_a = np.zeros(5, dtype=int)
        counts_b = np.zeros(5, dtype=int)
        while not done and steps < max_steps:
            act_a = agent_a.select_action(obs_flat, epsilon=0.0)
            act_b = agent_b.select_action(obs_flat, epsilon=0.0)
            counts_a[act_a] += 1
            counts_b[act_b] += 1
            obs, _, terminated, truncated, info = env.step([act_a, act_b])
            obs_flat = obs.flatten().astype(np.float32)
            done = terminated or truncated
            steps += 1
        learned_steps = steps
        learned_win = bool(info.get("both_reached_top", False))

        # Random baseline
        obs, info = env.reset(seed=ep_seed)
        done, steps = False, 0
        rand_counts_a = np.zeros(5, dtype=int)
        rand_counts_b = np.zeros(5, dtype=int)
        while not done and steps < max_steps:
            joint = env.action_space.sample()
            rand_counts_a[joint[0]] += 1
            rand_counts_b[joint[1]] += 1
            obs, _, terminated, truncated, info = env.step(joint)
            done = terminated or truncated
            steps += 1
        random_steps = steps
        random_win = bool(info.get("both_reached_top", False))

        def probs(c):
            return (c / max(1, c.sum())).tolist()

        pa = probs(counts_a)
        pb = probs(counts_b)
        rpa = probs(rand_counts_a)
        rpb = probs(rand_counts_b)
        l1_a = float(np.abs(np.array(pa) - np.array(rpa)).sum())
        l1_b = float(np.abs(np.array(pb) - np.array(rpb)).sum())

        out[str(ep_seed)] = {
            "a_counts": counts_a.tolist(), "a_probs": pa,
            "b_counts": counts_b.tolist(), "b_probs": pb,
            "rand_a_probs": rpa, "rand_b_probs": rpb,
            "l1_a": l1_a, "l1_b": l1_b,
            "learned_steps": learned_steps, "learned_win": learned_win,
            "random_steps": random_steps, "random_win": random_win,
        }
    return out


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_report(run_dir, params, train_stats, eval_results, action_comparison):
    ep_rewards = train_stats["ep_rewards"]
    ep_wins = train_stats["ep_wins"]
    ep_reward_a = train_stats["ep_reward_a"]
    ep_reward_b = train_stats["ep_reward_b"]

    episodes = list(range(1, len(ep_rewards) + 1))
    window = max(20, min(500, max(20, len(ep_rewards) // 20)))

    def rolling(data, w):
        if len(data) < w:
            return [None] * len(data)
        arr = np.array(data, dtype=float)
        cs = np.cumsum(arr)
        cs[w:] = cs[w:] - cs[:-w]
        out = cs[w - 1:] / w
        return [None] * (w - 1) + out.tolist()

    smooth_r = rolling(ep_rewards, window)
    smooth_ra = rolling(ep_reward_a, window)
    smooth_rb = rolling(ep_reward_b, window)
    # win rate: both win = 1.0, partial = 0.5, fail = 0.0
    win_vals = [1.0 if w == 2 else (0.5 if w == 1 else 0.0) for w in ep_wins]
    smooth_w = [v * 100 if v is not None else None for v in rolling(win_vals, window)]

    car_counts = sorted(eval_results.keys())
    both_rates = [eval_results[c]["both_win_rate"] * 100 for c in car_counts]
    a_rates = [eval_results[c]["a_win_rate"] * 100 for c in car_counts]
    b_rates = [eval_results[c]["b_win_rate"] * 100 for c in car_counts]
    avg_steps = [eval_results[c]["avg_steps"] for c in car_counts]

    sorted_action_keys = sorted(action_comparison.keys(), key=lambda x: int(x))
    l1_a_series = [action_comparison[k]["l1_a"] for k in sorted_action_keys]
    l1_b_series = [action_comparison[k]["l1_b"] for k in sorted_action_keys]

    params_html = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in params.items())

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>IDQN Report — {params['run_id']}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #fafafa; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 4px; }}
  table.params {{ border-collapse: collapse; margin: 10px 0 30px; }}
  table.params td {{ padding: 4px 16px; border: 1px solid #ccc; }}
  .plot {{ margin: 20px 0 40px; }}
</style>
</head><body>

<h1>🐸🐸 Quantum Frog — Stage 4 Independent DQN Report</h1>
<h2>Run Parameters</h2>
<table class="params">{params_html}</table>

<h2>1. Training Curves</h2>
<p>Rolling average over {window} episodes.</p>
<div id="training_reward" class="plot"></div>
<div id="training_agents" class="plot"></div>
<div id="training_winrate" class="plot"></div>

<h2>2. Success Rate vs. Car Count (Difficulty Curve)</h2>
<div id="difficulty_curve" class="plot"></div>

<h2>3. Average Episode Length vs. Car Count</h2>
<div id="avg_steps" class="plot"></div>

<h2>4. Episode Length Distribution per Car Count</h2>
<div id="steps_box" class="plot"></div>

<h2>5. Learning Evidence: Action Distribution vs Random Guessing</h2>
<label for="agentSel"><b>Agent:</b></label>
<select id="agentSel"><option value="a">Frog A</option><option value="b">Frog B</option></select>
&nbsp;&nbsp;
<label for="episodeSelector"><b>Episode seed:</b></label>
<select id="episodeSelector"></select>
<div id="action_compare" class="plot"></div>
<div id="action_divergence" class="plot"></div>

<script>
const epIdx = {json.dumps(episodes)};
const smoothR = {json.dumps(smooth_r)};
const smoothRA = {json.dumps(smooth_ra)};
const smoothRB = {json.dumps(smooth_rb)};
const smoothW = {json.dumps(smooth_w)};
const carCounts = {json.dumps(car_counts)};
const bothRates = {json.dumps(both_rates)};
const aRates = {json.dumps(a_rates)};
const bRates = {json.dumps(b_rates)};
const avgSteps = {json.dumps(avg_steps)};
const stepsData = {json.dumps({str(c): eval_results[c]['steps_list'] for c in car_counts})};
const actionLabels = {json.dumps(ACTION_LABELS)};
const actionComparison = {json.dumps(action_comparison)};
const actionEpisodeKeys = {json.dumps(sorted_action_keys)};
const l1ASeries = {json.dumps(l1_a_series)};
const l1BSeries = {json.dumps(l1_b_series)};

Plotly.newPlot('training_reward',
  [{{x: epIdx, y: smoothR, type:'scatter', mode:'lines', name:'Total reward', line:{{color:'#0f3460'}}}}],
  {{title:'Rolling Avg Total Episode Reward', xaxis:{{title:'Episode'}}, yaxis:{{title:'Reward'}}}},
  {{responsive:true}});

Plotly.newPlot('training_agents', [
  {{x: epIdx, y: smoothRA, type:'scatter', mode:'lines', name:'Frog A reward', line:{{color:'#e94560'}}}},
  {{x: epIdx, y: smoothRB, type:'scatter', mode:'lines', name:'Frog B reward', line:{{color:'#533483'}}}}
], {{title:'Per-Agent Rolling Avg Reward', xaxis:{{title:'Episode'}}, yaxis:{{title:'Reward'}}}}, {{responsive:true}});

Plotly.newPlot('training_winrate',
  [{{x: epIdx, y: smoothW, type:'scatter', mode:'lines', name:'Both-win rate', line:{{color:'#e94560'}}}}],
  {{title:'Rolling Both-Win Rate % (both frogs reach top)', xaxis:{{title:'Episode'}}, yaxis:{{title:'Win %', range:[0,100]}}}},
  {{responsive:true}});

Plotly.newPlot('difficulty_curve', [
  {{x: carCounts, y: bothRates, type:'scatter', mode:'lines+markers', name:'Both win', marker:{{size:10, color:'#0f3460'}}}},
  {{x: carCounts, y: aRates,    type:'scatter', mode:'lines+markers', name:'Frog A win', marker:{{size:8, color:'#e94560'}}}},
  {{x: carCounts, y: bRates,    type:'scatter', mode:'lines+markers', name:'Frog B win', marker:{{size:8, color:'#533483'}}}}
], {{title:'Success Rate vs Car Count', xaxis:{{title:'Number of Cars', dtick:1}}, yaxis:{{title:'Win %', range:[0,105]}}}}, {{responsive:true}});

Plotly.newPlot('avg_steps', [{{x: carCounts, y: avgSteps, type:'bar', marker:{{color:'#e94560'}}}}],
  {{title:'Avg Episode Length vs Car Count', xaxis:{{title:'Number of Cars', dtick:1}}, yaxis:{{title:'Avg Steps'}}}}, {{responsive:true}});

const boxTraces = carCounts.map(c => ({{y: stepsData[String(c)], type:'box', name: c+' cars'}}));
Plotly.newPlot('steps_box', boxTraces, {{title:'Episode Length Distribution', yaxis:{{title:'Steps'}}}}, {{responsive:true}});

function renderActionCompare() {{
  const agent = document.getElementById('agentSel').value;
  const epKey = document.getElementById('episodeSelector').value;
  const d = actionComparison[epKey];
  const learnedPct = (agent === 'a' ? d.a_probs : d.b_probs).map(v => v * 100);
  const randomPct  = (agent === 'a' ? d.rand_a_probs : d.rand_b_probs).map(v => v * 100);
  const l1 = agent === 'a' ? d.l1_a : d.l1_b;
  Plotly.newPlot('action_compare', [
    {{x: actionLabels, y: learnedPct, type:'bar', name:'Learned (greedy)', marker:{{color:'#0f3460'}}}},
    {{x: actionLabels, y: randomPct,  type:'bar', name:'Random baseline',  marker:{{color:'#e94560'}}}}
  ], {{
    title: `Action Distribution — Frog ${{agent.toUpperCase()}} seed ${{epKey}} | L1=${{l1.toFixed(3)}}`,
    barmode:'group', yaxis:{{title:'Action %', range:[0,100]}},
    annotations:[{{
      text:`Learned: steps=${{d.learned_steps}}, win=${{d.learned_win}} | Random: steps=${{d.random_steps}}, win=${{d.random_win}}`,
      showarrow:false, xref:'paper', yref:'paper', x:0, y:1.12, xanchor:'left'
    }}]
  }}, {{responsive:true}});

  Plotly.newPlot('action_divergence', [
    {{x: actionEpisodeKeys.map(Number), y: l1ASeries, type:'scatter', mode:'lines+markers', name:'Frog A', marker:{{size:9, color:'#e94560'}}}},
    {{x: actionEpisodeKeys.map(Number), y: l1BSeries, type:'scatter', mode:'lines+markers', name:'Frog B', marker:{{size:9, color:'#533483'}}}}
  ], {{
    title:'L1 Distance from Random (higher = more structured policy)',
    xaxis:{{title:'Episode seed'}}, yaxis:{{title:'L1 distance', range:[0,2]}}
  }}, {{responsive:true}});
}}

const selector = document.getElementById('episodeSelector');
actionEpisodeKeys.forEach(k => {{
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = k;
  selector.appendChild(opt);
}});
selector.addEventListener('change', renderActionCompare);
document.getElementById('agentSel').addEventListener('change', renderActionCompare);
if (actionEpisodeKeys.length > 0) {{
  selector.value = actionEpisodeKeys[0];
  renderActionCompare();
}}
</script>
</body></html>"""

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
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--train-cars", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Parallel envs per run (>1 uses vectorised trainer, improves GPU util)")
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.timesteps
    TRAIN_CARS = args.train_cars
    CAR_SPEEDS = (1,)
    EVAL_CAR_COUNTS = [1, 2, 3, 4, 5, 6]
    EVAL_EPISODES = args.eval_episodes
    KEY_EPISODES = [1, 500, 1000]

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.cpu:
        device = "cpu"
    elif args.gpu_id is not None:
        device = "cuda"
    else:
        device = "auto"

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_id = datetime.now().strftime("idqn_%Y%m%d_%H%M%S") + run_suffix
    run_dir = os.path.join("runs", "idqn", run_id)
    os.makedirs(run_dir, exist_ok=True)

    params = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "algorithm": "Independent DQN",
        "stage": 4,
        "total_timesteps": TOTAL_TIMESTEPS,
        "train_cars": TRAIN_CARS,
        "car_speeds": list(CAR_SPEEDS),
        "seed": args.seed,
        "device": device,
        "gpu_id": args.gpu_id,
        "grid_size": 8,
        "num_agents": 2,
        "policy": "MlpPolicy (flattened obs, independent per agent)",
        "n_envs": args.n_envs,
        "eval_car_counts": EVAL_CAR_COUNTS,
        "eval_episodes": EVAL_EPISODES,
        "key_episodes": KEY_EPISODES,
    }

    print(f"=== Run: {run_id} ===")
    print(f"Output: {run_dir}/  n_envs={args.n_envs}\n")

    print("Training Independent DQN (2 agents)...")
    t0 = time.time()
    if args.n_envs > 1:
        agent_a, agent_b, train_stats = train_idqn_vec(
            total_timesteps=TOTAL_TIMESTEPS,
            n_envs=args.n_envs,
            num_cars=TRAIN_CARS,
            car_speeds=CAR_SPEEDS,
            device=device,
            seed=args.seed,
            show_progress=(not args.no_progress),
        )
    else:
        agent_a, agent_b, train_stats = train_idqn(
            total_timesteps=TOTAL_TIMESTEPS,
            num_cars=TRAIN_CARS,
            car_speeds=CAR_SPEEDS,
            device=device,
            seed=args.seed,
            show_progress=(not args.no_progress),
        )
    train_time = time.time() - t0
    params["train_time_sec"] = round(train_time, 1)
    params["episodes_recorded"] = len(train_stats["ep_rewards"])

    agent_a.save(os.path.join(run_dir, f"agent_a_stage4_cars{TRAIN_CARS}_{TOTAL_TIMESTEPS}steps.pt"))
    agent_b.save(os.path.join(run_dir, f"agent_b_stage4_cars{TRAIN_CARS}_{TOTAL_TIMESTEPS}steps.pt"))

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
        agent_a, agent_b, EVAL_CAR_COUNTS,
        num_episodes=EVAL_EPISODES, car_speeds=CAR_SPEEDS,
    )
    with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
        json.dump({str(k): v for k, v in eval_results.items()}, f, indent=2)

    print("\nCollecting action-distribution evidence...")
    action_comparison = collect_action_distributions(
        agent_a, agent_b, num_cars=TRAIN_CARS,
        key_episodes=KEY_EPISODES, car_speeds=CAR_SPEEDS,
    )
    with open(os.path.join(run_dir, "action_distribution_key_episodes.json"), "w") as f:
        json.dump(action_comparison, f, indent=2)

    print("\nGenerating report...")
    generate_report(run_dir, params, train_stats, eval_results, action_comparison)

    print(f"\nDone in {train_time:.0f}s. Files in {run_dir}/")
