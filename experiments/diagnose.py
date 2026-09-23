"""
Exercise 3 diagnosis: why textbook DQN never learns MountainCar.

Three measurements, all seeded so they can be reproduced:

  1. Exploration reach -- how many episodes reach the flag when actions are
     random, as a function of how likely the previous random action is to be
     repeated (0.0 = textbook uniform sampling at every step).
  2. Textbook DQN -- train with plain epsilon-greedy and track the mean
     Q-value and the spread between the three actions (EXERCISES.md, clue 3).
  3. Tabular optimism -- tabular Q-Learning learns with plain epsilon-greedy.
     Is that because its Q-table starts at 0, above every true value (all
     rewards are -1)? Compare against a table that starts at -100, the value
     of never reaching the flag.

Usage:
    uv run python experiments/diagnose.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"
OUT_DIR = Path("results/diagnosis")
SEED = 0

# Constructor arguments that give the original, per-step uniform exploration.
TEXTBOOK_DQN: dict = {"explore_repeat": 0.0}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exploration_reach(repeat_probs: list[float], episodes: int = 300) -> dict[str, int]:
    """Episodes (out of `episodes`) that reach the flag with purely random actions.

    With probability `p` the previous action is repeated, otherwise a fresh
    uniform action is drawn. p = 0 is the textbook exploratory action.
    """
    env = gym.make(ENV_ID)
    reached: dict[str, int] = {}
    for p in repeat_probs:
        rng = random.Random(SEED)
        count = 0
        for ep in range(episodes):
            env.reset(seed=SEED * 10_000 + ep)
            action = None
            while True:
                if action is None or rng.random() >= p:
                    action = rng.randrange(3)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    count += int(terminated)
                    break
        reached[str(p)] = count
        print(f"  repeat p={p:<5} -> reached the flag in {count}/{episodes} episodes")
    env.close()
    return reached


def q_stats(agent: DQNAgent, states: torch.Tensor) -> tuple[float, float]:
    """Mean Q-value and mean spread (max - min over the 3 actions) per state."""
    with torch.no_grad():
        q = agent.q_net(states).numpy()
    return float(q.mean()), float(np.abs(q.max(1) - q.min(1)).mean())


def textbook_dqn(checkpoints: list[int]) -> dict:
    seed_everything(SEED)
    agent = DQNAgent(ENV_ID, **TEXTBOOK_DQN)
    env = gym.make(ENV_ID)
    env.observation_space.seed(SEED)
    probe = torch.as_tensor(
        np.array([env.observation_space.sample() for _ in range(200)]), dtype=torch.float32
    )
    env.close()

    rewards: list[float] = []
    table = []
    done_so_far = 0
    for checkpoint in checkpoints:
        chunk = checkpoint - done_so_far
        # Every train() call builds a fresh env, so each chunk gets its own seed.
        rewards += agent.train(total_episodes=chunk, log_interval=chunk, seed=SEED + done_so_far)
        done_so_far = checkpoint
        mean_q, spread = q_stats(agent, probe)
        table.append({"episodes": checkpoint, "mean_q": mean_q, "spread": spread})
        print(f"  {checkpoint:>5} episodes | mean Q {mean_q:8.2f} | spread {spread:.4f}")

    flags = sum(r > -200 for r in rewards)
    print(f"  flags reached during {len(rewards)} training episodes: {flags}")
    np.savetxt(OUT_DIR / "dqn_textbook_rewards.csv", rewards, fmt="%.1f", header="reward", comments="")
    return {"q_table": table, "training_episodes": len(rewards), "flags_reached": int(flags)}


def tabular_optimism(episodes: int = 5000) -> dict:
    results = {}
    for name, init in [("zeros", 0.0), ("minus_100", -100.0)]:
        seed_everything(SEED)
        agent = QLearningAgent(ENV_ID)
        agent.q_table = defaultdict(lambda n=agent.n_actions, v=init: np.full(n, v))
        rewards = agent.train(total_episodes=episodes, log_interval=episodes, seed=SEED)
        first = next((i + 1 for i, r in enumerate(rewards) if r > -200), None)
        flags = int(sum(r > -200 for r in rewards))
        results[name] = {"first_flag_episode": first, "flags_reached": flags}
        print(f"  Q init {init:>6}: first flag at episode {first}, {flags}/{episodes} flags")
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    print("1. Exploration reach (300 random episodes each)")
    reach = exploration_reach([0.0, 0.5, 0.8, 0.9, 0.95, 0.98])

    print("\n2. Textbook DQN (plain epsilon-greedy)")
    dqn = textbook_dqn([250, 750, 1500])

    print("\n3. Tabular Q-Learning: optimistic vs pessimistic initial values (5000 episodes)")
    tab = tabular_optimism()

    report = {"exploration_reach_of_300": reach, "textbook_dqn": dqn, "tabular_optimism": tab}
    (OUT_DIR / "diagnosis.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {OUT_DIR / 'diagnosis.json'}")


if __name__ == "__main__":
    main()
