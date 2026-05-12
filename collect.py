"""Aggregate per-task JSON results into a single CSV table."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


FIELDS = [
    "task_id", "alpha", "NE", "n_restarts", "max_iter", "seed",
    "cmi_upper_bound_bits", "elapsed_seconds",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, required=True,
                   help="Directory containing task_*.json files.")
    p.add_argument("--out", type=str, default="results.csv")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "task_*.json")))
    if not files:
        raise SystemExit(f"No task_*.json files found in {args.in_dir}")

    rows = []
    for fp in files:
        with open(fp) as f:
            rows.append(json.load(f))

    # Sort by (NE, alpha) for a tidy plot-friendly table.
    rows.sort(key=lambda r: (r["NE"], r["alpha"]))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print(f"Collected {len(rows)} results -> {args.out}")
    # Quick textual summary.
    print(f"\n{'alpha':>8} {'NE':>4} {'E_sq (bits)':>14} {'time (s)':>10}")
    for r in rows:
        print(f"{r['alpha']:>8.3f} {r['NE']:>4d} "
              f"{r['cmi_upper_bound_bits']:>14.6f} {r['elapsed_seconds']:>10.1f}")


if __name__ == "__main__":
    main()
