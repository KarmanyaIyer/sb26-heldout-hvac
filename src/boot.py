"""Paired building-level bootstrap CI of the relative score change between two runs.

python src/boot.py <ref_run> <run> [n_boot]
"""
import json
import pathlib
import sys

import numpy as np

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def means(name):
    d = json.loads((RES / f"{name}.json").read_text())
    return {r["scenario"]: -r["mean"] for r in d["scenarios"]}


ref, run = means(sys.argv[1]), means(sys.argv[2])
n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
assert set(ref) == set(run)
blds = sorted({k.split("/")[0] for k in ref})
by = {b: [k for k in ref if k.startswith(b + "/")] for b in blds}
a = np.array([[sum(ref[k] for k in by[b]), sum(run[k] for k in by[b]), len(by[b])] for b in blds])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(blds), size=(n_boot, len(blds)))
s = a[idx].sum(axis=1)
rel = 100 * (s[:, 1] - s[:, 0]) / s[:, 0]
pt = 100 * (a[:, 1].sum() - a[:, 0].sum()) / a[:, 0].sum()
lo, hi = np.percentile(rel, [2.5, 97.5])
print(f"{sys.argv[2]} vs {sys.argv[1]}: {pt:+.2f}% [{lo:+.2f}, {hi:+.2f}] (95% paired building bootstrap, {len(blds)} buildings)")
