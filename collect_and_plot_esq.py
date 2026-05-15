"""Collect per-task JSON results and plot E_sq upper bound vs alpha.

Reads results_esq/task_*.json, picks the best (lowest) bound per (NE, alpha),
and plots one line per NE value.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect(results_dir: str) -> dict[int, dict[float, float]]:
    """Return {NE: {alpha: best_cmi}} from task JSON files."""
    best: dict[int, dict[float, float]] = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(results_dir, "task_*.json")))
    if not files:
        raise SystemExit(f"No task_*.json files found in {results_dir}/")

    for fp in files:
        with open(fp) as f:
            doc = json.load(f)
        NE = int(doc["NE"])
        alpha = round(float(doc["alpha"]), 10)
        cmi = float(doc["cmi_upper_bound_bits"])
        if alpha not in best[NE] or cmi < best[NE][alpha]:
            best[NE][alpha] = cmi

    return dict(best)


def plot(data: dict[int, dict[float, float]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for NE in sorted(data):
        alphas = sorted(data[NE])
        vals = [data[NE][a] for a in alphas]
        ax.plot(alphas, vals, "o-", label=f"$N_E = {NE}$", markersize=4)

    ax.set_xlabel(r"$\alpha$", fontsize=13)
    ax.set_ylabel(r"$E_\mathrm{sq}$ upper bound (bits)", fontsize=13)
    ax.set_title(r"Squashed entanglement of the Werner state near $\alpha = -1/3$",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def print_table(data: dict[int, dict[float, float]]) -> None:
    NEs = sorted(data)
    all_alphas = sorted(set(a for d in data.values() for a in d))
    header = f"{'alpha':>16}" + "".join(f"{'NE='+str(NE):>14}" for NE in NEs)
    print(header)
    print("-" * len(header))
    for a in all_alphas:
        row = f"{a:>16.10f}"
        for NE in NEs:
            v = data[NE].get(a)
            if v is not None:
                row += f"{v:>14.6f}"
            else:
                row += f"{'---':>14}"
        print(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default="results_esq")
    p.add_argument("--out-plot", type=str, default="esq_vs_alpha.png")
    args = p.parse_args()

    data = collect(args.in_dir)
    print(f"Loaded data for NE = {sorted(data.keys())}\n")
    print_table(data)
    print()
    plot(data, args.out_plot)


if __name__ == "__main__":
    main()
