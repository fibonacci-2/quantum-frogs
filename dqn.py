"""Stage 3: DQN on harder single-frog setting (3-4 cars, mixed speeds 1-2)."""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env import QuantumFrogEnv


ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_wins = []
        self._running_rewards = None
        self._running_lengths = None

    def _on_training_start(self):
        n_envs = int(self.training_env.num_envs)
        self._running_rewards = np.zeros(n_envs, dtype=float)
        self._running_lengths = np.zeros(n_envs, dtype=int)

    def _on_step(self):
        rewards = np.array(self.locals.get("rewards", []), dtype=float)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if rewards.size:
            self._running_rewards[: rewards.size] += rewards
            self._running_lengths[: rewards.size] += 1

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            self.ep_rewards.append(float(self._running_rewards[i]))
            self.ep_lengths.append(int(self._running_lengths[i]))
            self.ep_wins.append(1 if info.get("reached_top", False) else 0)
            self._running_rewards[i] = 0.0
            self._running_lengths[i] = 0
        return True


def make_train_env(num_cars, car_speeds):
    def _factory():
        env = QuantumFrogEnv(render_mode=None, num_cars=num_cars, car_speeds=car_speeds)
        env = FlattenObservation(env)
        return env

    return _factory


def train_dqn(total_timesteps=150000, train_cars=4, car_speeds=(1, 2), device="auto", seed=0, show_progress=True):
    env = DummyVecEnv([make_train_env(train_cars, car_speeds)])
    callback = EpisodeStatsCallback()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=seed,
        device=device,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=show_progress)
    return model, callback


def evaluate_across_car_counts(model, car_counts, num_episodes=200, car_speeds=(1, 2), max_steps=200):
    results = {}
    for nc in car_counts:
        env = QuantumFrogEnv(render_mode=None, num_cars=nc, car_speeds=car_speeds)
        wins, total_steps, ep_steps = 0, 0, []
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep)
            obs = obs.flatten()
            done, steps = False, 0
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                obs = obs.flatten()
                done = terminated or truncated
                steps += 1
            if info["reached_top"]:
                wins += 1
            ep_steps.append(steps)
            total_steps += steps
        results[nc] = {
            "win_rate": wins / num_episodes,
            "avg_steps": total_steps / num_episodes,
            "steps_list": ep_steps,
        }
        print(f"  cars={nc}: win={results[nc]['win_rate'] * 100:.1f}%, avg_steps={results[nc]['avg_steps']:.1f}")
    return results


def collect_action_distributions(model, num_cars, key_episodes, car_speeds=(1, 2), max_steps=200):
    env = QuantumFrogEnv(render_mode=None, num_cars=num_cars, car_speeds=car_speeds)
    out = {}

    for ep_seed in key_episodes:
        obs, info = env.reset(seed=ep_seed)
        obs = obs.flatten()
        done = False
        steps = 0
        learned_counts = np.zeros(5, dtype=int)
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            learned_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs.flatten()
            done = terminated or truncated
            steps += 1
        learned_steps = steps
        learned_win = bool(info["reached_top"])

        obs, info = env.reset(seed=ep_seed)
        done = False
        steps = 0
        random_counts = np.zeros(5, dtype=int)
        while not done and steps < max_steps:
            action = env.action_space.sample()
            random_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        random_steps = steps
        random_win = bool(info["reached_top"])

        learned_probs = (learned_counts / max(1, learned_counts.sum())).tolist()
        random_probs = (random_counts / max(1, random_counts.sum())).tolist()
        l1_distance = float(np.abs(np.array(learned_probs) - np.array(random_probs)).sum())

        out[str(ep_seed)] = {
            "learned_counts": learned_counts.tolist(),
            "learned_probs": learned_probs,
            "learned_steps": learned_steps,
            "learned_win": learned_win,
            "random_counts": random_counts.tolist(),
            "random_probs": random_probs,
            "random_steps": random_steps,
            "random_win": random_win,
            "l1_distance": l1_distance,
        }

    return out


def generate_report(run_dir, params, train_rewards, train_wins, eval_results, action_comparison, train_timesteps):
    episodes = list(range(1, len(train_rewards) + 1))
    window = max(20, min(500, max(20, len(train_rewards) // 20)))

    def rolling(data, w):
        if len(data) < w:
            return [None] * len(data)
        arr = np.array(data, dtype=float)
        cumsum = np.cumsum(arr)
        cumsum[w:] = cumsum[w:] - cumsum[:-w]
        out = cumsum[w - 1:] / w
        return [None] * (w - 1) + out.tolist()

    smooth_rewards = rolling(train_rewards, window)
    smooth_wins = [v * 100 if v is not None else None for v in rolling(train_wins, window)]

    car_counts = sorted(eval_results.keys())
    win_rates = [eval_results[c]["win_rate"] * 100 for c in car_counts]
    avg_steps = [eval_results[c]["avg_steps"] for c in car_counts]

    sorted_action_keys = sorted(action_comparison.keys(), key=lambda x: int(x))
    l1_by_episode = [action_comparison[k]["l1_distance"] for k in sorted_action_keys]
    params_html = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in params.items())

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset=\"utf-8\">
<title>DQN Report — {params['run_id']}</title>
<script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #fafafa; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 4px; }}
  table.params {{ border-collapse: collapse; margin: 10px 0 30px; }}
  table.params td {{ padding: 4px 16px; border: 1px solid #ccc; }}
  .plot {{ margin: 20px 0 40px; }}
</style>
</head><body>

<h1>🐸 Quantum Frog — Stage 3 DQN Report</h1>
<h2>Run Parameters</h2>
<table class=\"params\">{params_html}</table>

<h2>1. Training Curves</h2>
<p>Episode reward and win rate rolling average (window={window}).</p>
<div id=\"training_reward\" class=\"plot\"></div>
<div id=\"training_winrate\" class=\"plot\"></div>

<h2>2. Success Rate vs. Car Count (Difficulty Curve)</h2>
<div id=\"difficulty_curve\" class=\"plot\"></div>

<h2>3. Average Episode Length vs. Car Count</h2>
<div id=\"avg_steps\" class=\"plot\"></div>

<h2>4. Episode Length Distribution per Car Count</h2>
<div id=\"steps_box\" class=\"plot\"></div>

<h2>5. Learning Evidence: Action Distribution vs Random Guessing</h2>
<label for=\"episodeSelector\"><b>Episode seed:</b></label>
<select id=\"episodeSelector\"></select>
<div id=\"action_compare\" class=\"plot\"></div>
<div id=\"action_divergence\" class=\"plot\"></div>

<script>
const epIdx = {json.dumps(episodes)};
const smoothR = {json.dumps(smooth_rewards)};
const smoothW = {json.dumps(smooth_wins)};
const carCounts = {json.dumps(car_counts)};
const winRates = {json.dumps(win_rates)};
const avgSteps = {json.dumps(avg_steps)};
const stepsData = {json.dumps({str(c): eval_results[c]['steps_list'] for c in car_counts})};
const actionLabels = {json.dumps(ACTION_LABELS)};
const actionComparison = {json.dumps(action_comparison)};
const actionEpisodeKeys = {json.dumps(sorted_action_keys)};
const l1ByEpisode = {json.dumps(l1_by_episode)};

Plotly.newPlot('training_reward', [{{x: epIdx, y: smoothR, type:'scatter', mode:'lines', line:{{color:'#0f3460'}}}}],
  {{title:'Rolling Avg Episode Reward', xaxis:{{title:'Episode index'}}, yaxis:{{title:'Reward'}}}}, {{responsive:true}});

Plotly.newPlot('training_winrate', [{{x: epIdx, y: smoothW, type:'scatter', mode:'lines', line:{{color:'#e94560'}}}}],
  {{title:'Rolling Win Rate %', xaxis:{{title:'Episode index'}}, yaxis:{{title:'Win %', range:[0,100]}}}}, {{responsive:true}});

Plotly.newPlot('difficulty_curve', [{{x: carCounts, y: winRates, type:'scatter', mode:'lines+markers', marker:{{size:10, color:'#0f3460'}}}}],
  {{title:'Success Rate vs Car Count', xaxis:{{title:'Number of Cars', dtick:1}}, yaxis:{{title:'Win %', range:[0,105]}}}}, {{responsive:true}});

Plotly.newPlot('avg_steps', [{{x: carCounts, y: avgSteps, type:'bar', marker:{{color:'#e94560'}}}}],
  {{title:'Avg Episode Length vs Car Count', xaxis:{{title:'Number of Cars', dtick:1}}, yaxis:{{title:'Avg Steps'}}}}, {{responsive:true}});

let boxTraces = carCounts.map(c => ({{y: stepsData[String(c)], type:'box', name: c+' cars'}}));
Plotly.newPlot('steps_box', boxTraces, {{title:'Episode Length Distribution', yaxis:{{title:'Steps'}}}}, {{responsive:true}});

function renderActionCompare(epKey) {{
  const d = actionComparison[epKey];
  const learnedPct = d.learned_probs.map(v => v * 100);
  const randomPct = d.random_probs.map(v => v * 100);
  Plotly.newPlot('action_compare', [
    {{x: actionLabels, y: learnedPct, type:'bar', name:'Learned (greedy)', marker:{{color:'#0f3460'}}}},
    {{x: actionLabels, y: randomPct, type:'bar', name:'Random baseline', marker:{{color:'#e94560'}}}}
  ], {{
    title: `Action Distribution — Episode ${{epKey}} | L1 distance=${{d.l1_distance.toFixed(3)}}`,
    barmode:'group',
    yaxis:{{title:'Action %', range:[0,100]}},
    annotations:[{{
      text:`Learned: steps=${{d.learned_steps}}, win=${{d.learned_win}} | Random: steps=${{d.random_steps}}, win=${{d.random_win}}`,
      showarrow:false, xref:'paper', yref:'paper', x:0, y:1.12, xanchor:'left'
    }}]
  }}, {{responsive:true}});
}}

const selector = document.getElementById('episodeSelector');
actionEpisodeKeys.forEach(k => {{
  const opt = document.createElement('option');
  opt.value = k;
  opt.textContent = k;
  selector.appendChild(opt);
}});
selector.addEventListener('change', (e) => renderActionCompare(e.target.value));
if (actionEpisodeKeys.length > 0) {{
  selector.value = actionEpisodeKeys[0];
  renderActionCompare(actionEpisodeKeys[0]);
}}

Plotly.newPlot('action_divergence', [{{
  x: actionEpisodeKeys.map(v => Number(v)),
  y: l1ByEpisode,
  type:'scatter', mode:'lines+markers',
  marker:{{size:10, color:'#533483'}}, line:{{color:'#533483'}},
}}], {{
  title:'Distance from Random Action Distribution (higher = less random)',
  xaxis:{{title:'Episode seed'}}, yaxis:{{title:'L1 distance (0 to 2)', range:[0,2]}}
}}, {{responsive:true}});
</script>

</body></html>"""

    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150000)
    parser.add_argument("--train-cars", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.timesteps
    TRAIN_CARS = args.train_cars
    CAR_SPEEDS = (1, 2)
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

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available in torch. "
            "Likely driver/runtime mismatch. Run `python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'` "
            "and install a torch build compatible with your NVIDIA driver."
        )

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""

    run_id = datetime.now().strftime("dqn_%Y%m%d_%H%M%S") + run_suffix
    run_dir = os.path.join("runs", "dqn", run_id)
    os.makedirs(run_dir, exist_ok=True)

    params = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "algorithm": "DQN",
        "stage": 3,
        "total_timesteps": TOTAL_TIMESTEPS,
        "train_cars": TRAIN_CARS,
        "car_speeds": list(CAR_SPEEDS),
        "seed": args.seed,
        "device": device,
        "gpu_id": args.gpu_id,
        "grid_size": 8,
        "policy": "MlpPolicy (flattened observation)",
        "eval_car_counts": EVAL_CAR_COUNTS,
        "eval_episodes": EVAL_EPISODES,
        "key_episodes": KEY_EPISODES,
    }

    print(f"=== Run: {run_id} ===")
    print(f"Output: {run_dir}/\n")

    print("Training DQN...")
    t0 = time.time()
    model, cb = train_dqn(
        total_timesteps=TOTAL_TIMESTEPS,
        train_cars=TRAIN_CARS,
        car_speeds=CAR_SPEEDS,
        device=device,
        seed=args.seed,
        show_progress=(not args.no_progress),
    )
    train_time = time.time() - t0
    params["train_time_sec"] = round(train_time, 1)
    params["episodes_recorded"] = len(cb.ep_rewards)

    model_path = os.path.join(run_dir, f"dqn_stage3_cars{TRAIN_CARS}_mixed12_{TOTAL_TIMESTEPS}steps")
    model.save(model_path)
    np.savez(
        os.path.join(run_dir, f"training_curves_cars{TRAIN_CARS}_{TOTAL_TIMESTEPS}steps.npz"),
        rewards=np.array(cb.ep_rewards),
        wins=np.array(cb.ep_wins),
        lengths=np.array(cb.ep_lengths),
    )

    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("\nEvaluating across car counts...")
    eval_results = evaluate_across_car_counts(
        model,
        EVAL_CAR_COUNTS,
        num_episodes=EVAL_EPISODES,
        car_speeds=CAR_SPEEDS,
    )

    print("\nCollecting action-distribution evidence...")
    action_comparison = collect_action_distributions(
        model,
        num_cars=TRAIN_CARS,
        key_episodes=KEY_EPISODES,
        car_speeds=CAR_SPEEDS,
    )
    with open(os.path.join(run_dir, "action_distribution_key_episodes.json"), "w") as f:
        json.dump(action_comparison, f, indent=2)

    print("\nGenerating report...")
    generate_report(
        run_dir,
        params,
        cb.ep_rewards,
        cb.ep_wins,
        eval_results,
        action_comparison,
        TOTAL_TIMESTEPS,
    )

    print(f"\nDone in {train_time:.0f}s. Files in {run_dir}/")
