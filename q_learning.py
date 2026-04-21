"""Tabular Q-Learning agent for Quantum Frog.

References are to Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed).
"""

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from env import QuantumFrogEnv


ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class QLearningAgent:
    """Off-policy TD control — Q-Learning (Barto §6.5, Eq 6.8).

    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, n_actions=5, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.n_actions = n_actions
        self.alpha = alpha          # step size (Barto §2.4)
        self.gamma = gamma          # discount factor (Barto §3.3)
        self.epsilon = epsilon      # exploration rate (Barto §2.2, ε-greedy)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def _state_key(self, obs):
        return obs.tobytes()

    def select_action(self, obs, greedy=False):
        """ε-greedy action selection (Barto §2.2, Eq 2.2)."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_table[self._state_key(obs)]
        return int(np.argmax(q))

    def learn(self, obs, action, reward, next_obs, terminated):
        """One-step Q-Learning update (Barto §6.5, Eq 6.8)."""
        s = self._state_key(obs)
        s_next = self._state_key(next_obs)
        target = reward
        if not terminated:
            target += self.gamma * np.max(self.q_table[s_next])
        td_error = target - self.q_table[s][action]
        self.q_table[s][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(num_episodes=20000, num_cars=2, car_speeds=(1,), log_every=1000):
    env = QuantumFrogEnv(render_mode=None, num_cars=num_cars, car_speeds=car_speeds)
    agent = QLearningAgent()

    rewards_log = []
    wins_log = []
    epsilon_log = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 200:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total_reward += reward
            done = terminated or truncated
            steps += 1

        agent.decay_epsilon()
        rewards_log.append(total_reward)
        wins_log.append(1 if info["reached_top"] else 0)
        epsilon_log.append(agent.epsilon)

        if ep % log_every == 0:
            recent = slice(-log_every, None)
            avg_r = np.mean(rewards_log[recent])
            win_rate = np.mean(wins_log[recent]) * 100
            print(f"Ep {ep:>6} | ε={agent.epsilon:.3f} | Avg R={avg_r:>7.1f} | Win%={win_rate:.1f} | Q-states={len(agent.q_table)}")

    return agent, rewards_log, wins_log, epsilon_log


# ---------------------------------------------------------------------------
# Evaluation — answers Section 3 questions
# ---------------------------------------------------------------------------

def evaluate_across_car_counts(agent, car_counts, num_episodes=200):
    """Success rate & avg steps vs car count (§3: difficulty curve)."""
    results = {}
    for nc in car_counts:
        env = QuantumFrogEnv(render_mode=None, num_cars=nc)
        wins, total_steps, ep_steps = 0, 0, []
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep)
            done, steps = False, 0
            while not done and steps < 200:
                action = agent.select_action(obs, greedy=True)
                obs, reward, terminated, truncated, info = env.step(action)
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
        print(f"  cars={nc}: win={results[nc]['win_rate']*100:.1f}%, avg_steps={results[nc]['avg_steps']:.1f}")
    return results


def collect_action_distributions(agent, num_cars, key_episodes, max_steps=200, car_speeds=(1,)):
    """Compare learned greedy policy vs random baseline on key episode seeds."""
    env = QuantumFrogEnv(render_mode=None, num_cars=num_cars, car_speeds=car_speeds)
    out = {}

    for ep_seed in key_episodes:
        # Learned (greedy)
        obs, info = env.reset(seed=ep_seed)
        done = False
        steps = 0
        learned_counts = np.zeros(agent.n_actions, dtype=int)
        while not done and steps < max_steps:
            action = agent.select_action(obs, greedy=True)
            learned_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        learned_steps = steps
        learned_win = bool(info["reached_top"])

        # Random baseline
        obs, info = env.reset(seed=ep_seed)
        done = False
        steps = 0
        random_counts = np.zeros(agent.n_actions, dtype=int)
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


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    run_dir,
    params,
    rewards_log,
    wins_log,
    epsilon_log,
    eval_results,
    train_car_count,
    action_comparison,
):
    """Produce an interactive report.html with Plotly answering Section 3 questions."""

    window = 500
    episodes = list(range(1, len(rewards_log) + 1))

    def rolling(data, w):
        arr = np.array(data, dtype=float)
        cumsum = np.cumsum(arr)
        cumsum[w:] = cumsum[w:] - cumsum[:-w]
        out = cumsum[w - 1:] / w
        return [None] * (w - 1) + out.tolist()

    smooth_rewards = rolling(rewards_log, window)
    smooth_wins = [v * 100 if v is not None else None for v in rolling(wins_log, window)]

    car_counts = sorted(eval_results.keys())
    win_rates = [eval_results[c]["win_rate"] * 100 for c in car_counts]
    avg_steps = [eval_results[c]["avg_steps"] for c in car_counts]

    params_html = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in params.items())
    sorted_action_keys = sorted(action_comparison.keys(), key=lambda x: int(x))
    l1_by_episode = [action_comparison[k]["l1_distance"] for k in sorted_action_keys]

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Q-Learning Report — {params['run_id']}</title>
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

<h1>🐸 Quantum Frog — Q-Learning Report</h1>

<h2>Run Parameters</h2>
<table class="params">{params_html}</table>

<h2>1. Training Curves</h2>
<p>Rolling average (window={window}) of episode reward and win rate during training.</p>
<div id="training_reward" class="plot"></div>
<div id="training_winrate" class="plot"></div>
<div id="epsilon_decay" class="plot"></div>

<h2>2. Success Rate vs. Car Count (Difficulty Curve)</h2>
<p>Agent trained on {train_car_count} cars, then evaluated greedily across different car counts.
   Shows how difficulty scales — answers <b>§3 Q1</b>.</p>
<div id="difficulty_curve" class="plot"></div>

<h2>3. Average Episode Length vs. Car Count</h2>
<p>Measures policy efficiency. Shorter episodes on easy configs, longer (or capped) on hard ones — answers <b>§3 Q2</b>.</p>
<div id="avg_steps" class="plot"></div>

<h2>4. Episode Length Distribution per Car Count</h2>
<p>Box plot showing variance in episode length — reveals whether the agent is consistent or lucky.</p>
<div id="steps_box" class="plot"></div>

<h2>5. Learning Evidence: Action Distribution vs Random Guessing</h2>
<p>For key episodes (early/mid/late), compare action-choice distribution from the learned policy against a random-action baseline using the same episode seed. This directly shows non-random behavior.</p>
<label for="episodeSelector"><b>Episode seed:</b></label>
<select id="episodeSelector"></select>
<div id="action_compare" class="plot"></div>
<div id="action_divergence" class="plot"></div>

<script>
const eps = {json.dumps(episodes)};
const smooth_r = {json.dumps(smooth_rewards)};
const smooth_w = {json.dumps(smooth_wins)};
const epsilon = {json.dumps(epsilon_log)};
const car_counts = {json.dumps(car_counts)};
const win_rates = {json.dumps(win_rates)};
const avg_steps = {json.dumps(avg_steps)};
const steps_data = {json.dumps({str(c): eval_results[c]["steps_list"] for c in car_counts})};
const actionLabels = {json.dumps(ACTION_LABELS)};
const actionComparison = {json.dumps(action_comparison)};
const actionEpisodeKeys = {json.dumps(sorted_action_keys)};
const l1ByEpisode = {json.dumps(l1_by_episode)};

Plotly.newPlot('training_reward', [{{x: eps, y: smooth_r, type:'scatter', mode:'lines', name:'Avg Reward', line:{{color:'#0f3460'}}}}],
  {{title:'Rolling Avg Reward', xaxis:{{title:'Episode'}}, yaxis:{{title:'Reward'}}}}, {{responsive:true}});

Plotly.newPlot('training_winrate', [{{x: eps, y: smooth_w, type:'scatter', mode:'lines', name:'Win %', line:{{color:'#e94560'}}}}],
  {{title:'Rolling Win Rate %', xaxis:{{title:'Episode'}}, yaxis:{{title:'Win %', range:[0,100]}}}}, {{responsive:true}});

Plotly.newPlot('epsilon_decay', [{{x: eps, y: epsilon, type:'scatter', mode:'lines', name:'ε', line:{{color:'#533483'}}}}],
  {{title:'Exploration Rate (ε) Decay', xaxis:{{title:'Episode'}}, yaxis:{{title:'ε'}}}}, {{responsive:true}});

Plotly.newPlot('difficulty_curve', [{{x: car_counts, y: win_rates, type:'scatter', mode:'lines+markers',
  marker:{{size:10, color:'#0f3460'}}, line:{{color:'#0f3460'}}}}],
  {{title:'Success Rate vs Car Count', xaxis:{{title:'Number of Cars', dtick:1}}, yaxis:{{title:'Win %', range:[0,105]}}}}, {{responsive:true}});

Plotly.newPlot('avg_steps', [{{x: car_counts, y: avg_steps, type:'bar', marker:{{color:'#e94560'}}}}],
  {{title:'Avg Episode Length vs Car Count', xaxis:{{title:'Number of Cars', dtick:1}}, yaxis:{{title:'Avg Steps'}}}}, {{responsive:true}});

let box_traces = car_counts.map(c => ({{y: steps_data[String(c)], type:'box', name: c+' cars'}}));
Plotly.newPlot('steps_box', box_traces,
  {{title:'Episode Length Distribution', yaxis:{{title:'Steps'}}}}, {{responsive:true}});

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
        xaxis:{{title:'Action'}},
        annotations:[{{
            text:`Learned: steps=${{d.learned_steps}}, win=${{d.learned_win}} | Random: steps=${{d.random_steps}}, win=${{d.random_win}}`,
            showarrow:false,
            xref:'paper', yref:'paper', x:0, y:1.15, xanchor:'left'
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
    type:'scatter',
    mode:'lines+markers',
    marker:{{size:10, color:'#533483'}},
    line:{{color:'#533483'}},
}}], {{
    title:'Distance from Random Action Distribution (higher = less random)',
    xaxis:{{title:'Episode seed'}},
    yaxis:{{title:'L1 distance (0 to 2)', range:[0,2]}}
}}, {{responsive:true}});
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
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--train-cars", type=int, default=2)
    parser.add_argument("--car-speeds", type=int, nargs="+", default=[1])
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--run-tag", type=str, default="")
    args = parser.parse_args()

    NUM_EPISODES = args.episodes
    TRAIN_CARS = args.train_cars
    CAR_SPEEDS = tuple(args.car_speeds)
    EVAL_CAR_COUNTS = [1, 2, 3, 4, 5, 6]
    EVAL_EPISODES = args.eval_episodes
    KEY_EPISODES = [1, NUM_EPISODES // 2, NUM_EPISODES]

    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_id = datetime.now().strftime("qlearn_%Y%m%d_%H%M%S") + tag_suffix
    run_dir = os.path.join("runs", "q_learning", run_id)
    os.makedirs(run_dir, exist_ok=True)

    params = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "algorithm": "Tabular Q-Learning",
        "num_episodes": NUM_EPISODES,
        "train_cars": TRAIN_CARS,
        "car_speeds": list(CAR_SPEEDS),
        "grid_size": 8,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.9995,
        "max_steps_per_ep": 200,
        "eval_car_counts": EVAL_CAR_COUNTS,
        "eval_episodes": EVAL_EPISODES,
        "key_episodes": KEY_EPISODES,
    }

    print(f"=== Run: {run_id} ===")
    print(f"Output: {run_dir}/\n")

    # Train
    print("Training...")
    t0 = time.time()
    agent, rewards_log, wins_log, epsilon_log = train(num_episodes=NUM_EPISODES, num_cars=TRAIN_CARS, car_speeds=CAR_SPEEDS)
    train_time = time.time() - t0
    params["train_time_sec"] = round(train_time, 1)
    params["q_table_states"] = len(agent.q_table)

    # Save model & training data
    agent.save(os.path.join(run_dir, f"q_table_cars{TRAIN_CARS}_{NUM_EPISODES}ep.pkl"))
    np.savez(
        os.path.join(run_dir, f"training_curves_cars{TRAIN_CARS}_{NUM_EPISODES}ep.npz"),
        rewards=rewards_log, wins=wins_log, epsilon=epsilon_log,
    )
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Evaluate across car counts
    print("\nEvaluating across car counts...")
    eval_results = evaluate_across_car_counts(agent, EVAL_CAR_COUNTS, num_episodes=EVAL_EPISODES)

    print("\nCollecting action-distribution evidence...")
    action_comparison = collect_action_distributions(
        agent,
        num_cars=TRAIN_CARS,
        key_episodes=KEY_EPISODES,
        max_steps=200,
        car_speeds=CAR_SPEEDS,
    )
    with open(os.path.join(run_dir, "action_distribution_key_episodes.json"), "w") as f:
        json.dump(action_comparison, f, indent=2)

    # Generate report
    print("\nGenerating report...")
    generate_report(
        run_dir,
        params,
        rewards_log,
        wins_log,
        epsilon_log,
        eval_results,
        TRAIN_CARS,
        action_comparison,
    )

    print(f"\nDone in {train_time:.0f}s. Files in {run_dir}/")
