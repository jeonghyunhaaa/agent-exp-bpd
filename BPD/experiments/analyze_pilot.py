"""Analysis for epsilon_scaling_pilot.py results.

Run AFTER experiments/epsilon_scaling_pilot.py has produced its output files.
Reads:
  - experiments/results/epsilon_pilot_raw.pkl
  - experiments/results/epsilon_pilot_summary.csv

Produces:
  - experiments/results/table_best_epsilon.csv
  - experiments/results/fig_youden_by_epsilon.png
  - experiments/results/fig_gap_distribution.png
"""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load() -> tuple[list[dict], pd.DataFrame]:
    with open(RESULTS_DIR / "epsilon_pilot_raw.pkl", "rb") as f:
        raw = pickle.load(f)
    summary = pd.read_csv(RESULTS_DIR / "epsilon_pilot_summary.csv")
    return raw, summary


def table_best_epsilon(summary: pd.DataFrame) -> pd.DataFrame:
    """N별로 Youden's J(recall - FPR)를 최대화하는 epsilon을 뽑는다."""
    best = summary.loc[summary.groupby("N")["youden_j"].idxmax()].reset_index(drop=True)
    best = best[["N", "epsilon", "recall", "fpr", "youden_j"]].rename(columns={"epsilon": "best_epsilon"})
    best.to_csv(RESULTS_DIR / "table_best_epsilon.csv", index=False)
    return best


def plot_youden_by_epsilon(summary: pd.DataFrame) -> None:
    """N별로 epsilon 스윕에 따른 Youden's J 곡선. 정점 위치가 N에 따라 이동하는지가 핵심."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, g in summary.groupby("N"):
        g = g.sort_values("epsilon")
        ax.plot(g["epsilon"], g["youden_j"], marker="o", label=f"N={n}")
    ax.axvline(1.5, color="gray", linestyle="--", linewidth=1, label="epsilon=1.5 (paper default)")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Youden's J (recall - FPR)")
    ax.set_title("Detection quality vs epsilon, by agent count N")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_youden_by_epsilon.png", dpi=150)
    plt.close(fig)


def plot_gap_distribution(raw: list[dict]) -> None:
    """attack 조건에서 malicious agent의 gap vs benign agent들의 gap을 N별로 비교."""
    rows = []
    for r in raw:
        if not r["attack"]:
            continue
        agent_scores = r["agent_scores"]
        for agent_id, score in agent_scores.items():
            others = [s for aid, s in agent_scores.items() if aid != agent_id]
            if not others:
                continue
            gap = sum(abs(score - o) for o in others) / len(others)
            role = "malicious" if agent_id == r["malicious_agent_id"] else "benign"
            rows.append({"N": r["N"], "gap": gap, "role": role})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ns = sorted(df["N"].unique())
    for i, n in enumerate(ns):
        sub = df[df["N"] == n]
        benign = sub[sub["role"] == "benign"]["gap"]
        malicious = sub[sub["role"] == "malicious"]["gap"]
        ax.scatter([i - 0.08] * len(benign), benign, color="steelblue", alpha=0.6, label="benign" if i == 0 else None)
        ax.scatter([i + 0.08] * len(malicious), malicious, color="firebrick", alpha=0.9, label="malicious" if i == 0 else None)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("gap (mean pairwise |score diff|)")
    ax.set_title("Malicious vs benign agent gap, by N (attack rows only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_gap_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    raw, summary = load()
    best = table_best_epsilon(summary)
    print(best.to_string(index=False))
    plot_youden_by_epsilon(summary)
    plot_gap_distribution(raw)
    print(f"\nSaved: {RESULTS_DIR / 'table_best_epsilon.csv'}")
    print(f"Saved: {RESULTS_DIR / 'fig_youden_by_epsilon.png'}")
    print(f"Saved: {RESULTS_DIR / 'fig_gap_distribution.png'}")


if __name__ == "__main__":
    main()
