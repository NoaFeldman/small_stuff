"""Aggregate per-task JSON sweeps into a single CSV table."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


FIELDS = ["NE", "alpha", "cmi_upper_bound_bits", "direction",
          "task_id", "n_restarts", "max_iter", "seed", "elapsed_seconds"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="results.csv")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "task_*.json")))
    if not files:
        raise SystemExit(f"No task_*.json files found in {args.in_dir}")

    rows = []
    for fp in files:
        with open(fp) as f:
            doc = json.load(f)
        for r in doc.get("results", []):
            rows.append({
                "NE": doc["NE"],
                "alpha": r["alpha"],
                "cmi_upper_bound_bits": r["cmi_upper_bound_bits"],
                "direction": r.get("direction", ""),
                "task_id": doc["task_id"],
                "n_restarts": doc["n_restarts"],
                "max_iter": doc["max_iter"],
                "seed": doc["seed"],
                "elapsed_seconds": doc["elapsed_seconds"],
            })

    rows.sort(key=lambda r: (r["NE"], r["alpha"]))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Collected {len(rows)} (NE, alpha) points -> {args.out}\n")
    print(f"{'alpha':>8} {'NE':>4} {'E_sq (bits)':>14}  dir")
    for r in rows:
        print(f"{r['alpha']:>8.3f} {r['NE']:>4d} "
              f"{r['cmi_upper_bound_bits']:>14.6f}  {r['direction']}")


if __name__ == "__main__":
    main()
