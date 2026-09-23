"""
Build the figures, GIFs and summary table used in the README from the files
written by run_experiments.py and diagnose.py.

The best agent of each method is picked by its 100-episode evaluation. Since
that same evaluation chose it, it is re-evaluated on 100 *new* start states
(confirmation set) that were used neither for training nor for picking.

Usage:
    uv run python experiments/report.py
"""
import json
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # render to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from PIL import Image
from run_experiments import evaluate

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"
RESULTS = Path("results")
FIGURES = RESULTS / "figures"
EVAL_SEED = 10_000
CONFIRM_SEED = 20_000  # start states for the confirmation evaluation
WINDOW = 100  # moving-average window for training curves
AGENTS = {
    "qlearning": {"label": "Q-Learning tabular", "cls": QLearningAgent, "ext": "pkl", "color": "#1f77b4"},
    "dqn": {"label": "DQN", "cls": DQNAgent, "ext": "pt", "color": "#d62728"},
}


def moving_average(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_runs(name: str) -> list[dict]:
    runs = []
    for path in (RESULTS / name).glob("seed*_summary.json"):
        summary = json.loads(path.read_text())
        seed = summary["seed"]
        summary["train"] = np.loadtxt(RESULTS / name / f"seed{seed}_train_rewards.csv", skiprows=1)
        summary["eval"] = np.loadtxt(RESULTS / name / f"seed{seed}_eval_rewards.csv", skiprows=1)
        runs.append(summary)
    return sorted(runs, key=lambda r: r["seed"])


def load_agent(name: str, seed: int):
    info = AGENTS[name]
    return info["cls"].load(RESULTS / name / f"seed{seed}.{info['ext']}")


def best_run(runs: list[dict]) -> dict:
    return max(runs, key=lambda r: (r["eval_flags_reached"], r["eval_mean"]))


def first_episode_at(curve: np.ndarray, level: float) -> int | None:
    """First episode whose trailing moving average is >= level."""
    hits = np.nonzero(curve >= level)[0]
    return int(hits[0] + WINDOW) if hits.size else None


def never_reached(runs: list[dict]) -> list[int]:
    return [r["seed"] for r in runs if r["train_flags_reached"] == 0]


def episode_axis(rewards: np.ndarray) -> np.ndarray:
    return np.arange(WINDOW, len(rewards) + 1)


# ── figures ───────────────────────────────────────────────────────────


def plot_training(name: str, runs: list[dict], best: dict, textbook: np.ndarray | None = None) -> None:
    info = AGENTS[name]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for run in runs:
        if run["seed"] == best["seed"]:
            continue
        failed = run["train_flags_reached"] == 0
        ax.plot(episode_axis(run["train"]), moving_average(run["train"]), lw=0.9, alpha=0.6,
                color="#ff7f0e" if failed else "#888888")
    ax.plot(episode_axis(best["train"]), moving_average(best["train"]), lw=2.2, color=info["color"])
    handles = [
        Line2D([], [], color=info["color"], lw=2.2, label=f"mejor agente (semilla {best['seed']})"),
        Line2D([], [], color="#888888", lw=0.9, label="otras semillas"),
    ]
    failed = never_reached(runs)
    if failed:
        seeds = ", ".join(map(str, failed))
        handles.append(Line2D([], [], color="#ff7f0e", lw=0.9, label=f"semillas {seeds}: nunca llegaron a la bandera"))
    if textbook is not None:
        ax.plot(episode_axis(textbook), moving_average(textbook), color="black", ls="--", lw=1.4)
        handles.append(Line2D([], [], color="black", ls="--", lw=1.4, label="DQN con ε-greedy de libro (sin corrección)"))
    ax.axhline(-110, color="gray", ls=":", lw=1)
    handles.append(Line2D([], [], color="gray", ls=":", lw=1, label="umbral 'resuelto' (−110)"))
    ax.set_title(f"{info['label']}: recompensa de entrenamiento, {len(runs)} semillas (media móvil de {WINDOW})")
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Recompensa por episodio")
    ax.set_ylim(-205, -90)
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_training.png", dpi=130)
    plt.close(fig)


def plot_comparison_steps(all_runs: dict[str, list[dict]]) -> None:
    """Both methods on a common x-axis: environment steps (the real cost of experience)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    handles = []
    for name, runs in all_runs.items():
        info = AGENTS[name]
        for run in runs:
            steps = np.cumsum(-run["train"])[WINDOW - 1:]
            ax.plot(steps / 1e6, moving_average(run["train"]), color=info["color"], lw=0.9, alpha=0.45)
        handles.append(Line2D([], [], color=info["color"], lw=1.5, label=f"{info['label']} ({len(runs)} semillas)"))
    ax.axhline(-110, color="gray", ls=":", lw=1)
    handles.append(Line2D([], [], color="gray", ls=":", lw=1, label="umbral 'resuelto' (−110)"))
    ax.set_title("Q-Learning vs DQN según la experiencia usada")
    ax.set_xlabel("Pasos de entorno acumulados (millones)")
    ax.set_ylabel(f"Recompensa (media móvil {WINDOW})")
    ax.set_ylim(-205, -90)
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "comparison_env_steps.png", dpi=130)
    plt.close(fig)


def plot_stability(all_runs: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (name, runs) in enumerate(all_runs.items()):
        info = AGENTS[name]
        means = np.array([r["eval_mean"] for r in runs])
        flags = np.array([r["eval_flags_reached"] for r in runs])
        x = i + (np.array([r["seed"] for r in runs]) - 4.5) * 0.05
        ax.scatter(x, means, s=70, color=info["color"], alpha=0.75, edgecolor="black", zorder=3)
        # Label only the agents that do not always reach the flag; the ones that never do share a point.
        for xi, m, r in zip(x, means, runs):
            if 0 < r["eval_flags_reached"] < 100:
                left = r["seed"] < 5  # write the label on the free side of the point
                ax.annotate(f"semilla {r['seed']}: {r['eval_flags_reached']}/100", (xi, m),
                            textcoords="offset points", xytext=(-9 if left else 9, -3),
                            ha="right" if left else "left", fontsize=7.5)
        zero = [r["seed"] for r in runs if r["eval_flags_reached"] == 0]
        if zero:
            ax.annotate(f"semillas {', '.join(map(str, zero))}: 0/100", (x[runs.index(
                next(r for r in runs if r["eval_flags_reached"] == 0))], -200), textcoords="offset points",
                xytext=(9, 6), fontsize=7.5)
        solved, failed = int((flags == 100).sum()), int((flags == 0).sum())
        ax.text(i, -72, f"{solved}/{len(runs)} llegan siempre · {failed}/{len(runs)} nunca llegan",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(all_runs)), [AGENTS[n]["label"] for n in all_runs])
    ax.axhline(-110, color="gray", ls=":", lw=1)
    ax.set_xlim(-0.6, len(all_runs) - 0.4)
    ax.set_ylim(-205, -65)
    ax.set_ylabel("Recompensa media en evaluación (100 episodios)")
    ax.set_title("Estabilidad: evaluación de los 10 agentes entrenados por método (un punto por semilla)\n"
                 "etiquetados: agentes que no llegan a la bandera en todos los episodios", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "stability_seeds.png", dpi=130)
    plt.close(fig)


def plot_best_evaluation(best: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data, labels, colors = [], [], []
    for name, run in best.items():
        head = f"{AGENTS[name]['label']} (semilla {run['seed']})"
        data += [run["eval"], run["confirm"]]
        labels += [f"{head}\nevaluación", f"{head}\nconfirmación\n(estados iniciales nuevos)"]
        colors += [AGENTS[name]["color"]] * 2
    parts = ax.boxplot(data, patch_artist=True, widths=0.55)
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for i, (values, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(values))
        ax.scatter(i + jitter, values, s=8, color=color, alpha=0.6)
        ax.text(i, -73, f"media {np.mean(values):.1f}\nbanderas {int(np.sum(values > -200))}/100",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7.5)
    ax.axhline(-110, color="gray", ls=":", lw=1)
    ax.set_ylim(-205, -62)
    ax.set_ylabel("Recompensa por episodio")
    ax.set_title("Mejor agente de cada método: 100 episodios greedy en cada conjunto de estados iniciales")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "best_evaluation.png", dpi=130)
    plt.close(fig)


def greedy_grids(name: str, agent, n: int = 150) -> tuple[np.ndarray, np.ndarray, list]:
    """Greedy action and max-Q over a (velocity x position) grid of the state space."""
    env = gym.make(ENV_ID)
    low, high = env.observation_space.low, env.observation_space.high
    env.close()
    pos = np.linspace(low[0], high[0], n)
    vel = np.linspace(low[1], high[1], n)
    grid = np.array([[p, v] for v in vel for p in pos], dtype=np.float32)
    if name == "dqn":
        with torch.no_grad():
            q = agent.q_net(torch.as_tensor(grid)).numpy()
    else:
        keys = [agent.discretize(s) for s in grid]
        q = np.array([agent.q_table[k] if k in agent.q_table else np.full(agent.n_actions, np.nan) for k in keys])
    unseen = np.isnan(q).any(1)
    filled = np.nan_to_num(q, nan=-np.inf)
    action = np.where(unseen, np.nan, filled.argmax(1))
    value = np.where(unseen, np.nan, filled.max(1))
    extent = [low[0], high[0], low[1], high[1]]
    return action.reshape(n, n), value.reshape(n, n), extent


def plot_policies(best: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    cmap = ListedColormap(["#4c72b0", "#e8d98a", "#dd8452"])
    cmap.set_bad("white")
    value_cmap = matplotlib.colormaps["viridis"].copy()
    value_cmap.set_bad("white")
    for row, (name, run) in enumerate(best.items()):
        info = AGENTS[name]
        action, value, extent = greedy_grids(name, load_agent(name, run["seed"]))
        ax = axes[row, 0]
        ax.imshow(action, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest")
        ax.set_title(f"{info['label']} (semilla {run['seed']}): acción greedy")
        ax = axes[row, 1]
        im = ax.imshow(value, origin="lower", extent=extent, aspect="auto", cmap=value_cmap,
                       interpolation="nearest")
        ax.set_title(f"{info['label']}: valor estimado max_a Q(s, a)")
        fig.colorbar(im, ax=ax)
        for ax in axes[row]:
            ax.axvline(0.5, color="black", ls="--", lw=1)
            ax.set_xlabel("Posición")
            ax.set_ylabel("Velocidad")
    fig.suptitle("Lo que aprendió cada agente (blanco en Q-Learning = celdas nunca visitadas; "
                 "línea punteada = bandera)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(3)]
    fig.legend(handles, ["empujar izquierda", "no empujar", "empujar derecha"], loc="lower center",
               ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "policies_and_values.png", dpi=120)
    plt.close(fig)


def save_gif(name: str, run: dict) -> dict:
    agent = load_agent(name, run["seed"])
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=EVAL_SEED)
    frames, total, steps = [env.render()], 0.0, 0
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total += float(reward)
        steps += 1
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    images = [Image.fromarray(f).resize((360, 240)) for f in frames[::2]]
    images[0].save(FIGURES / f"{name}_best.gif", save_all=True, append_images=images[1:], duration=40, loop=0)
    return {"seed": run["seed"], "reward": total, "steps": steps, "reached_flag": bool(terminated)}


# ── summary table ─────────────────────────────────────────────────────


def aggregate(runs: list[dict]) -> str:
    means = [r["eval_mean"] for r in runs]
    solved = [r for r in runs if r["eval_flags_reached"] == r["eval_episodes"]]
    failed = [r for r in runs if r["eval_flags_reached"] == 0]
    text = (
        f"{len(solved)}/{len(runs)} semillas llegan a la bandera en los 100 episodios de evaluación y "
        f"{len(failed)}/{len(runs)} nunca llegan; evaluación media {np.mean(means):.1f} "
        f"(mediana {np.median(means):.1f}, rango {min(means):.1f} a {max(means):.1f})"
    )
    if solved:
        text += f"; media de las semillas que llegan siempre: {np.mean([r['eval_mean'] for r in solved]):.1f}"
    return text + "."


def summary_table(all_runs: dict[str, list[dict]], best: dict[str, dict]) -> str:
    lines = [
        (
            "| Método | Semilla | Pasos de entorno | 1.ª bandera (episodio) | Media móvil ≥ −150 (episodio) | "
            "Media últimos 100 (entrenamiento) | Evaluación 100 ep. (media ± desv.) | Mejor / peor | "
            "Llega a la bandera |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, runs in all_runs.items():
        for r in runs:
            reach = first_episode_at(moving_average(r["train"]), -150)
            mark = " **(mejor)**" if r["seed"] == best[name]["seed"] else ""
            lines.append(
                f"| {AGENTS[name]['label']}{mark} | {r['seed']} | {r['train_env_steps']} | "
                f"{r['first_flag_episode'] or 'nunca'} | {reach or 'nunca'} | {r['train_last100_mean']:.1f} | "
                f"{r['eval_mean']:.1f} ± {r['eval_std']:.1f} | {r['eval_best']:.0f} / {r['eval_worst']:.0f} | "
                f"{r['eval_flags_reached']}/{r['eval_episodes']} |"
            )
    for name, runs in all_runs.items():
        lines.append(f"\n**{AGENTS[name]['label']}**: {aggregate(runs)}")
    for name, run in best.items():
        c = run["confirm"]
        lines.append(
            f"\n**Confirmación del mejor {AGENTS[name]['label']} (semilla {run['seed']})** en 100 estados "
            f"iniciales nuevos: {c.mean():.1f} ± {c.std():.1f}, mejor {c.max():.0f}, peor {c.min():.0f}, "
            f"banderas {int((c > -200).sum())}/100."
        )
    return "\n".join(lines)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    all_runs = {name: load_runs(name) for name in AGENTS}
    best = {name: best_run(runs) for name, runs in all_runs.items()}
    for name, run in best.items():
        run["confirm"] = np.array(evaluate(load_agent(name, run["seed"]), first_seed=CONFIRM_SEED)[0])

    textbook_csv = RESULTS / "diagnosis" / "dqn_textbook_rewards.csv"
    textbook = np.loadtxt(textbook_csv, skiprows=1) if textbook_csv.exists() else None
    plot_training("qlearning", all_runs["qlearning"], best["qlearning"])
    plot_training("dqn", all_runs["dqn"], best["dqn"], textbook)
    plot_comparison_steps(all_runs)
    plot_stability(all_runs)
    plot_best_evaluation(best)
    plot_policies(best)
    gifs = {name: save_gif(name, run) for name, run in best.items()}

    table = summary_table(all_runs, best)
    (RESULTS / "summary.md").write_text(table + "\n")
    confirm = {name: {"seed": run["seed"], "rewards": run["confirm"].tolist()} for name, run in best.items()}
    (RESULTS / "best_agents.json").write_text(json.dumps({"gif_episode": gifs, "confirmation": confirm}, indent=1))
    print(table)
    print("\nGIF episodes:", json.dumps(gifs))


if __name__ == "__main__":
    main()
