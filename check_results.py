"""Sanity-check the Werner squashed-entanglement upper bound results.

The data is split between:
  * old warm-started full sweeps (task_1/2/3.json), and
  * new per-(NE, alpha) cluster jobs (task_4..63.json).

We merge them by keeping the tightest bound for each (NE, alpha) -- any
optimization run is a valid variational upper bound, so the minimum across
runs is always at least as good.

Checks (only properties that are guaranteed to hold for *any* valid run):

  1. Non-negativity (CMI >= 0, modulo numerical slack).
  2. Universal upper bound: E_sq(rho_AB) <= log_2(d) for d = 3.
  3. Exactness at alpha = -1: rho_AB is pure on the antisymmetric subspace
     of dimension 3, so E_sq = log_2(3).
  4. Regression vs. the old warm-started sweep: report where the new
     per-point runs are looser than the old sweep (the merged bound is
     still the min, but this signals the warm-start advantage was lost).
  5. Improvement statistics: how often did the new runs strictly improve
     on the old sweep.

We deliberately do *not* require the bound to be ~0 in the separable region
(alpha >= -1/3) or to be monotone in NE.  The bound is well-defined for any
parameter vector but the underlying L-BFGS-B optimization is non-convex and
even the old warm-started sweep does not get close to 0 in the separable
region (e.g. NE=3, alpha=+1 gives 0.53 bits).
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

RESULTS_DIR = "results"

D = 3
LOG2_D = math.log2(D)
TOL_NONNEG = 1e-6
TOL_UPPER = 1e-6
TOL_AT_MINUS_ONE = 5e-3


def load() -> tuple[dict, dict, dict, dict]:
    merged: dict[int, dict[float, float]] = defaultdict(dict)
    old: dict[int, dict[float, float]] = defaultdict(dict)
    new: dict[int, dict[float, float]] = defaultdict(dict)
    n_new_runs: dict[int, dict[float, int]] = defaultdict(lambda: defaultdict(int))
    elapsed_old: dict[int, float] = defaultdict(float)
    elapsed_new: dict[int, float] = defaultdict(float)
    new_tasks: dict[int, int] = defaultdict(int)

    for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, "task_*.json"))):
        with open(fp) as f:
            doc = json.load(f)
        NE = int(doc["NE"])
        is_sweep = "results" in doc
        rows = doc["results"] if is_sweep else [doc]
        for r in rows:
            # np.linspace and the per-point cluster grid agree only up to 1e-16.
            a = round(float(r["alpha"]), 6)
            v = float(r["cmi_upper_bound_bits"])
            if is_sweep:
                old[NE][a] = v
            else:
                if a not in new[NE] or v < new[NE][a]:
                    new[NE][a] = v
                n_new_runs[NE][a] += 1
            if a not in merged[NE] or v < merged[NE][a]:
                merged[NE][a] = v
        if is_sweep:
            elapsed_old[NE] += float(doc.get("elapsed_seconds", 0.0))
        else:
            elapsed_new[NE] += float(doc.get("elapsed_seconds", 0.0))
            new_tasks[NE] += 1

    meta = {NE: {"elapsed_old": elapsed_old[NE],
                 "elapsed_new": elapsed_new[NE],
                 "n_new_tasks": new_tasks[NE],
                 "n_new_runs": dict(n_new_runs[NE])}
            for NE in merged}
    return dict(merged), dict(old), dict(new), meta


def main() -> None:
    merged, old, new, meta = load()
    if not merged:
        raise SystemExit(f"No task_*.json files in {RESULTS_DIR}/")

    NEs = sorted(merged)
    print(f"Found NE values: {NEs}")
    for NE in NEs:
        m = meta[NE]
        missing = sorted(a for a in merged[NE] if a not in new.get(NE, {}))
        print(f"  NE={NE}: {len(merged[NE])} alphas merged. "
              f"old sweep: {len(old.get(NE, {}))} pts in {m['elapsed_old']:.0f}s. "
              f"new cluster: {len(new.get(NE, {}))} pts, "
              f"{m['n_new_tasks']} tasks, {m['elapsed_new']:.0f}s total.")
        if missing:
            print(f"    new runs do not cover: "
                  + ", ".join(f"{a:+.2f}" for a in missing))
    print()

    failures: list[str] = []

    # 1) Non-negativity.
    for NE in NEs:
        for a, v in sorted(merged[NE].items()):
            if v < -TOL_NONNEG:
                failures.append(f"[NE={NE}] alpha={a:+.4f}: bound = {v:.3e} < 0")

    # 2) Universal upper bound: bound <= log_2(d).
    print(f"Universal upper bound  E_sq <= log_2({D}) = {LOG2_D:.6f}:")
    for NE in NEs:
        worst_a = max(merged[NE], key=lambda a: merged[NE][a])
        worst_v = merged[NE][worst_a]
        ok = worst_v <= LOG2_D + TOL_UPPER
        print(f"  NE={NE}: max merged bound = {worst_v:.6f} at alpha={worst_a:+.4f}"
              f"   {'OK' if ok else 'BAD'}")
        if not ok:
            failures.append(f"[NE={NE}] alpha={worst_a:+.4f}: bound "
                            f"{worst_v:.6f} > log_2({D}) = {LOG2_D:.6f}")
    print()

    # 3) Exactness at alpha = -1.
    print(f"At alpha = -1, expected  E_sq = log_2({D}) = {LOG2_D:.6f}:")
    for NE in NEs:
        if -1.0 not in merged[NE]:
            continue
        v = merged[NE][-1.0]
        diff = v - LOG2_D
        ok = abs(diff) < TOL_AT_MINUS_ONE
        print(f"  NE={NE}: bound = {v:.6f}  (diff = {diff:+.2e})  "
              f"{'OK' if ok else 'BAD'}")
        if not ok:
            failures.append(f"[NE={NE}] alpha=-1: bound {v:.6f} deviates "
                            f"from log_2({D})")
    print()

    # 4) Regression: where did the new runs come out looser than the old sweep?
    print("New per-point runs vs. old warm-started sweep (shared alphas):")
    notable = []
    for NE in NEs:
        if NE not in old or NE not in new:
            continue
        deltas = []
        for a in sorted(set(old[NE]) & set(new[NE])):
            d = new[NE][a] - old[NE][a]
            deltas.append((a, old[NE][a], new[NE][a], d))
            if d > 0.05:
                notable.append((NE, a, old[NE][a], new[NE][a], d))
        if not deltas:
            continue
        better = sum(1 for *_, d in deltas if d < -1e-6)
        worse  = sum(1 for *_, d in deltas if d >  1e-6)
        worst_better = min((d for *_, d in deltas), default=0.0)
        worst_worse  = max((d for *_, d in deltas), default=0.0)
        print(f"  NE={NE} ({len(deltas)} shared alphas): "
              f"new tighter at {better}, looser at {worse}, "
              f"best improvement = {worst_better:+.4f} bits, "
              f"worst regression = {worst_worse:+.4f} bits.")
    if notable:
        print("  Looser by > 0.05 bits at:")
        for NE, a, vo, vn, d in notable:
            print(f"    NE={NE} alpha={a:+.4f}: old {vo:.4f}  ->  new {vn:.4f}  "
                  f"(delta = {d:+.4f})")
    print()

    # 5) Strict improvements over the old sweep.
    print("Strict improvements from new cluster runs (>= 1e-4 bits):")
    n_improved = 0
    for NE in NEs:
        if NE not in old:
            continue
        for a in sorted(merged[NE]):
            vo = old[NE].get(a)
            if vo is None:
                continue
            vm = merged[NE][a]
            if vm < vo - 1e-4:
                n_improved += 1
                print(f"  NE={NE} alpha={a:+.4f}: {vo:.6f}  ->  {vm:.6f}  "
                      f"(delta = {vm - vo:+.4f})")
    print(f"  total improvements: {n_improved}")
    print()

    if failures:
        print(f"FAIL: {len(failures)} hard check(s) failed:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)
    print("All hard sanity checks PASSED.")


if __name__ == "__main__":
    main()
