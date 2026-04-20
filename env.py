"""Quantum Frog — Gymnasium environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class QuantumFrogEnv(gym.Env):

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    UP, DOWN, LEFT, RIGHT, STAY = range(5)
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}
    KEY_MAP = {"w": 0, "s": 1, "a": 2, "d": 3, " ": 4}

    def __init__(self, render_mode=None, num_cars=2, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.num_cars = num_cars
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(3, grid_size, grid_size), dtype=np.int8
        )
        self.action_space = spaces.Discrete(5)

        self.frog_pos = None
        self.cars = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frog_pos = (self.grid_size - 1, self.grid_size // 2)
        self.cars = []
        car_rows = self.np_random.choice(
            range(1, self.grid_size - 1), size=self.num_cars, replace=True
        )
        for row in car_rows:
            col = int(self.np_random.integers(0, self.grid_size))
            vel = int(self.np_random.choice([-1, 1]))
            self.cars.append({"row": int(row), "col": col, "vel": vel})
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action)

        dr, dc = self.ACTION_DELTAS[action]
        old_row = self.frog_pos[0]
        new_r = int(np.clip(old_row + dr, 0, self.grid_size - 1))
        new_c = int(np.clip(self.frog_pos[1] + dc, 0, self.grid_size - 1))
        self.frog_pos = (new_r, new_c)

        for car in self.cars:
            car["col"] = (car["col"] + car["vel"]) % self.grid_size

        hit = any(
            c["row"] == self.frog_pos[0] and c["col"] == self.frog_pos[1]
            for c in self.cars
        )
        reached_top = self.frog_pos[0] == 0

        reward = -1.0
        if hit:
            reward = -100.0
        elif reached_top:
            reward = 100.0
        elif self.frog_pos[0] < old_row:
            reward += 1.0

        terminated = hit or reached_top
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode != "ansi":
            return None
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for car in self.cars:
            grid[car["row"]][car["col"]] = "→" if car["vel"] > 0 else "←"
        fr, fc = self.frog_pos
        grid[fr][fc] = "F"
        border = "+" + "--" * self.grid_size + "+"
        lines = [border]
        for row in grid:
            lines.append("|" + " ".join(row) + "|")
        lines.append(border)
        out = "\n".join(lines)
        print(out)
        return out

    def play(self, seed=None):
        """Interactive manual play. WASD to move, space to stay, q to quit."""
        obs, info = self.reset(seed=seed)
        self.render()
        total_reward = 0.0
        step = 0

        while True:
            key = input("\nMove (w/a/s/d/space, q=quit): ").strip().lower()
            if key == "q":
                print(f"Quit. Total reward: {total_reward}, steps: {step}")
                break
            if key not in self.KEY_MAP:
                print("Invalid key. Use w=up, s=down, a=left, d=right, space=stay")
                continue

            action = self.KEY_MAP[key]
            obs, reward, terminated, truncated, info = self.step(action)
            step += 1
            total_reward += reward

            print(f"Step {step} | Action: {self.ACTION_NAMES[action]} | Reward: {reward}")
            self.render()

            if terminated:
                result = "YOU WIN!" if info["reached_top"] else "HIT BY CAR!"
                print(f"\n{result} Total reward: {total_reward}, steps: {step}")
                break

    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int8)
        obs[0, self.frog_pos[0], self.frog_pos[1]] = 1
        for car in self.cars:
            obs[1, car["row"], car["col"]] = 1
            obs[2, car["row"], car["col"]] = car["vel"]
        return obs

    def _get_info(self):
        return {"frog_pos": self.frog_pos, "reached_top": self.frog_pos[0] == 0}


if __name__ == "__main__":
    env = QuantumFrogEnv(render_mode="ansi", num_cars=5, grid_size=16)
    env.play()
