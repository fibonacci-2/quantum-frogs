"""Launch multiple Stage-4 Independent DQN runs, one per GPU."""

import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4, help="How many GPUs / parallel runs")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--train-cars", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=32,
                        help="Parallel envs per GPU run (improves GPU utilisation)")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    procs = []
    for gpu in range(args.gpus):
        cmd = [
            py,
            os.path.join(root, "idqn.py"),
            "--gpu-id", str(gpu),
            "--seed", str(100 + gpu),
            "--run-tag", f"gpu{gpu}",
            "--timesteps", str(args.timesteps),
            "--eval-episodes", str(args.eval_episodes),
            "--train-cars", str(args.train_cars),
            "--n-envs", str(args.n_envs),
            "--no-progress",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join(root, "runs", "idqn", f"launcher_gpu{gpu}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        f = open(log_path, "w")
        p = subprocess.Popen(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((gpu, p, f, log_path))
        print(f"Started GPU {gpu}: pid={p.pid}, log={log_path}")

    try:
        while True:
            alive = sum(1 for _, p, _, _ in procs if p.poll() is None)
            print(f"Running jobs: {alive}/{len(procs)}")
            if alive == 0:
                break
            time.sleep(15)
    finally:
        for _, p, f, _ in procs:
            if p.poll() is None:
                p.terminate()
            f.close()

    print("All runs finished.")
    for gpu, p, _, log_path in procs:
        print(f"GPU {gpu}: exit={p.returncode}, log={log_path}")


if __name__ == "__main__":
    main()
