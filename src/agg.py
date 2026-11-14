"""Aggregate rollout result JSONs into one table.

python src/agg.py --ref <run>
Results are collected in results/*.json (one file per run name). Prints score, decomposition,
per-climate means, and paired deltas vs a reference on common scenarios.
"""
import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]



RES = ROOT / "results"
RES.mkdir(exist_ok=True)


runs = {}
for f in sorted(RES.glob("*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if "scenarios" not in d or d["summary"]["steps"] < 500:
        continue
    runs[f.stem] = d


def climate(s):
    w = s.split("/")[1]
    return w.split("_")[0]


ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
print(f"{'run':34s} {'n':>3s} {'steps':>5s} {'score':>9s} {'comf':>8s} {'energy':>8s} {'carbon':>8s}  per-climate")
for name, d in runs.items():
    sc = d["scenarios"]
    s = d["summary"]
    byc = defaultdict(list)
    for r in sc:
        byc[climate(r["scenario"])].append(-r["mean"])
    cl = " ".join(f"{k[:4]}={sum(v)/len(v):.5f}" for k, v in sorted(byc.items()))
    print(f"{name:34s} {len(sc):3d} {s['steps']:5d} {s['score']:.6f} {s['comfort']:.6f} {s['energy']:.6f} {s['carbon']:.6f}  {cl}")
    if ref and ref in runs and name != ref:
        a = {r["scenario"]: -r["mean"] for r in runs[ref]["scenarios"]}
        b = {r["scenario"]: -r["mean"] for r in sc}
        common = sorted(set(a) & set(b))
        if common and runs[ref]["summary"]["steps"] == s["steps"]:
            da = sum(a[k] for k in common) / len(common)
            db = sum(b[k] for k in common) / len(common)
            wins = sum(b[k] < a[k] for k in common)
            print(f"{'':34s}   vs {ref}: {db:.6f} vs {da:.6f} ({100*(db-da)/da:+.2f}%), better on {wins}/{len(common)}")
