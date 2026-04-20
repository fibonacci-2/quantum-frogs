"""Tabular Q-Learning agent for Quantum Frog.

References are to Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed).
"""

import numpy as np
import pickle
from collections import defaultdict
from env import QuantumFrogEnv


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
        # Q-table: state_key → array of Q-values per action (Barto §6.5)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def _state_key(self, obs):
        """Flatten observation into a hashable key for tabular lookup."""
        return obs.tobytes()

    def select_action(self, obs, greedy=False):
        """ε-greedy action selection (Barto §2.2, Eq 2.2).

        With probability ε pick random action (explore),
        otherwise pick argmax Q(s,·) (exploit).
        """
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_table[self._state_key(obs)]
        return int(np.argmax(q))

    def learn(self, obs, action, reward, next_obs, terminated):
        """One-step Q-Learning update (Barto §6.5, Eq 6.8).

        Uses the max over next-state actions (off-policy),
        distinguishing it from SARSA which uses the actual next action.
        Bootstrap target: r + γ max_a' Q(s',a') when not terminal,
        just r when terminal (Barto §3.4, terminal states have value 0).
        """
        s = self._state_key(obs)
        s_next = self._state_key(next_obs)

        target = reward
        if not terminated:
            target += self.gamma * np.max(self.q_table[s_next])

        # TD error (Barto §6.1)
        td_error = target - self.q_table[s][action]
        self.q_table[s][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate after each episode (Barto §2.6, optimistic starts alternative)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data)


def train(num_episodes=10000, num_cars=2, log_every=1000):
    """Standard RL training loop (Barto §6.5, Figure 6.5)."""
    env = QuantumFrogEnv(render_mode=None, num_cars=num_cars)
    agent = QLearningAgent()

    rewards_log = []
    wins_log = []

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

        if ep % log_every == 0:
            recent = slice(-log_every, None)
            avg_r = np.mean(rewards_log[recent])
            win_rate = np.mean(wins_log[recent]) * 100
            print(f"Ep {ep:>6} | ε={agent.epsilon:.3f} | Avg R={avg_r:>7.1f} | Win%={win_rate:.1f} | Q-states={len(agent.q_table)}")

    agent.save()
    print(f"\nTraining done. Q-table saved ({len(agent.q_table)} states).")
    return agent, rewards_log, wins_log


def evaluate(agent, num_episodes=100, num_cars=2):
    """Greedy evaluation — no exploration (Barto §5.4, policy evaluation)."""
    env = QuantumFrogEnv(render_mode=None, num_cars=num_cars)
    wins = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        steps = 0
        while not done and steps < 200:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        if info["reached_top"]:
            wins += 1
        total_steps += steps

    print(f"Eval: {wins}/{num_episodes} wins ({wins}%), avg steps={total_steps / num_episodes:.1f}")
    return wins


def demo(agent, num_cars=2, seed=42):
    """Watch the trained agent play one episode."""
    env = QuantumFrogEnv(render_mode="ansi", num_cars=num_cars)
    obs, info = env.reset(seed=seed)
    env.render()
    done = False
    step = 0

    while not done and step < 50:
        action = agent.select_action(obs, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        name = QuantumFrogEnv.ACTION_NAMES[action]
        print(f"\nStep {step} | {name} | R={reward}")
        env.render()

    result = "WIN" if info["reached_top"] else "LOSS"
    print(f"\n{result} in {step} steps.")


if __name__ == "__main__":
    agent, rewards, wins = train(num_episodes=20000, num_cars=2)
    evaluate(agent, num_episodes=200, num_cars=2)
    demo(agent)
