"""
Train both agents with several seeds, evaluate them, and save everything the
README reports: the training log, the per-episode training rewards, a
100-episode greedy evaluation and the trained agent.

Each (agent, seed) run goes to results/<agent>/ (or results/<tag>/<agent>/) and
runs in its own process.

Usage:
    uv run python experiments/run_experiments.py                    # both agents, seeds 0 1 2
    uv run python experiments/run_experiments.py --agents dqn --seeds 0
    uv run python experiments/run_experiments.py --agents dqn --seeds 0 --explore-repeat 0.95 --tag tuning/p0.95
    uv run python experiments/run_experiments.py --seeds 0 1 2 3 4 5 6 7 8 9 --tag robustness
"""
import argparse
import contextlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"
RESULTS = Path("results")
EVAL_EPISODES = 100
EVAL_SEED = 10_000  # evaluation start states never overlap the training seeds

# Episode budgets follow the course README (~20k for Q-Learning, ~2500 for DQN).
CONFIGS = {
    "qlearning": {"cls": QLearningAgent, "episodes": 20_000, "log_interval": 500, "ext": "pkl"},
    "dqn": {"cls": DQNAgent, "episodes": 2_500, "log_interval": 50, "ext": "pt"},
}


def evaluate(agent, episodes: int = EVAL_EPISODES, first_seed: int = EVAL_SEED) -> tuple[list[float], int]:
    """Greedy episodes from fixed start states. Returns (rewards, flags reached)."""
    env = gym.make(ENV_ID)
    rewards, reached = [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=first_seed + i)
        total = 0.0
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            if terminated or truncated:
                reached += int(terminated)
                break
        rewards.append(total)
    env.close()
    return rewards, reached


def run(agent_name: str, seed: int, tag: str, kwargs: dict) -> dict:
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = CONFIGS[agent_name]
    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)
    agent = cfg["cls"](ENV_ID, **kwargs)

    start = time.perf_counter()
    with open(out / f"seed{seed}_train.log", "w") as log, contextlib.redirect_stdout(log):
        print(agent.info(), end="\n\n")
        rewards = agent.train(cfg["episodes"], cfg["log_interval"], seed=seed)
    minutes = (time.perf_counter() - start) / 60

    eval_rewards, reached = evaluate(agent)
    with contextlib.redirect_stdout(None):
        agent.save(out / f"seed{seed}.{cfg['ext']}")
    np.savetxt(out / f"seed{seed}_train_rewards.csv", rewards, fmt="%.0f", header="reward", comments="")
    np.savetxt(out / f"seed{seed}_eval_rewards.csv", eval_rewards, fmt="%.0f", header="reward", comments="")

    flags = [i + 1 for i, r in enumerate(rewards) if r > -200]
    summary = {
        "agent": agent_name,
        "tag": tag,
        "seed": seed,
        "hyperparameters": {k: getattr(agent, k) for k in agent._HPARAMS if k != "env_id"},
        "train_episodes": len(rewards),
        "train_env_steps": int(-sum(rewards)),
        "train_minutes": round(minutes, 2),
        "first_flag_episode": flags[0] if flags else None,
        "train_flags_reached": len(flags),
        "train_last100_mean": float(np.mean(rewards[-100:])),
        "eval_episodes": EVAL_EPISODES,
        "eval_mean": float(np.mean(eval_rewards)),
        "eval_std": float(np.std(eval_rewards)),
        "eval_best": float(np.max(eval_rewards)),
        "eval_worst": float(np.min(eval_rewards)),
        "eval_flags_reached": reached,
    }
    (out / f"seed{seed}_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[{tag} seed {seed}] {minutes:.1f} min | eval {summary['eval_mean']:.1f} "
        f"+/- {summary['eval_std']:.1f} | flag {reached}/{EVAL_EPISODES}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--explore-repeat", type=float, default=None, help="Override DQN explore_repeat")
    parser.add_argument("--tag", default=None, help="Results subfolder: results/<tag>/<agent> (default: results/<agent>)")
    args = parser.parse_args()

    jobs = []
    for agent_name in args.agents:
        kwargs = {}
        if agent_name == "dqn" and args.explore_repeat is not None:
            kwargs["explore_repeat"] = args.explore_repeat
        tag = f"{args.tag}/{agent_name}" if args.tag else agent_name
        jobs += [(agent_name, seed, tag, kwargs) for seed in args.seeds]

    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for future in [pool.submit(run, *job) for job in jobs]:
            future.result()


if __name__ == "__main__":
    main()
