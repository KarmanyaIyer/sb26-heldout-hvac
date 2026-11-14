"""Lean rollout harness mirroring the Kaggle gateway.

Same loader, metadata builder and occupancy constant as kaggle_evaluation/gateway.py
(which itself calls smart_buildings.evaluate), but no tracker/charts. Records the
per-step reward decomposition so we can see which term dominates.

Usage:
  python src/rollout.py --policy policies.mine:MinePolicy --set train30 [--steps 8640] [--jobs 8]
Scenario sets: train30 (shipped), or a directory containing building/*/weather/config.gin.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
KIT = pathlib.Path(os.environ.get("SB_KIT", ROOT / "ext" / "starter-kit"))
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402


def load_policy_class(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def run_one(args):
    scen_dir, policy_spec, steps, policy_kwargs = args
    import matplotlib

    matplotlib.use("Agg")
    from smart_buildings import evaluate

    scen_dir = pathlib.Path(scen_dir)
    cwd = os.getcwd()
    os.chdir(KIT)
    try:
        env = evaluate.load_hybrid_environment(scen_dir / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0
        meta = evaluate.create_building_meta_data(env, scen_dir)
    finally:
        os.chdir(cwd)
    spec = env.action_spec()
    cls = load_policy_class(policy_spec)
    t0 = time.time()
    policy = cls(meta, spec, **policy_kwargs) if policy_kwargs else cls(meta, spec)
    t_init = time.time() - t0
    # ANALYSIS ONLY (never in a submission): oracle policies may ask for the live env,
    # e.g. to replay the occupancy model and measure what seed replay is worth.
    if getattr(policy, "wants_env", False):
        policy.env = env

    comps = {"prod": 0.0, "energy": 0.0, "carbon": 0.0, "occ_steps": 0}
    # loss by (Mon | Tue-Fri | weekend, hour): [comfort, energy+carbon] in reward units
    by_hour = np.zeros((3, 24, 2))
    rf = env.reward_function
    orig = rf.compute_reward

    def wrapped(info):
        r = orig(info)
        comps["prod"] += r.normalized_productivity_regret
        comps["energy"] += r.normalized_energy_cost
        comps["carbon"] += r.normalized_carbon_emission
        comps["occ_steps"] += int(r.total_occupancy > 0)
        try:
            ts = env.current_simulation_timestamp
            wk = 0 if ts.dayofweek == 0 else (2 if ts.dayofweek >= 5 else 1)
            by_hour[wk, ts.hour, 0] += -0.6 * r.normalized_productivity_regret
            by_hour[wk, ts.hour, 1] += 0.2 * (r.normalized_energy_cost + r.normalized_carbon_emission)
        except Exception:
            pass
        return r

    rf.compute_reward = wrapped

    obs, _ = env.reset()
    total = 0.0
    n = 0
    t_pol = 0.0
    t1 = time.time()
    for _ in range(steps):
        ta = time.time()
        a = policy.get_action(obs)
        t_pol += time.time() - ta
        act = {
            "continuous_action": np.asarray(a["continuous_action"], dtype=np.float32),
            "discrete_action": np.asarray(a["discrete_action"], dtype=np.int32),
        }
        assert np.all(np.isfinite(act["continuous_action"]))
        assert act["continuous_action"].min() >= -1.0 - 1e-6 and act["continuous_action"].max() <= 1.0 + 1e-6
        obs, r, term, trunc, _ = env.step(act)
        total += float(r)
        n += 1
        if term or trunc:
            break
    wall = time.time() - t1
    name = f"{scen_dir.parent.name}/{scen_dir.name}"
    out = {
        "scenario": name,
        "steps": n,
        "total": total,
        "mean": total / n,
        "comfort_term": 0.6 * comps["prod"] / n,
        "energy_term": -0.2 * comps["energy"] / n,
        "carbon_term": -0.2 * comps["carbon"] / n,
        "occ_frac": comps["occ_steps"] / n,
        "by_hour": (by_hour / n).round(7).tolist(),
        "sec": wall,
        "policy_sec": t_pol,
        "init_sec": t_init,
        "n_ahu": len(meta["temp_indices"]),
        "n_zone": len(meta["zone_temp_indices"]),
    }
    return out


def scenario_dirs(which: str):
    if which == "train30":
        base = KIT / "smart_buildings" / "training_scenarios"
    else:
        base = pathlib.Path(which)
    return sorted(p.parent for p in base.glob("*/*/config.gin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--set", default="train30")
    ap.add_argument("--filter", default="")
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--kwargs", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    kw = json.loads(a.kwargs) if a.kwargs else {}
    dirs = [d for d in scenario_dirs(a.set) if all(f in str(d) for f in a.filter.split(",") if f)]
    jobs = [(str(d), a.policy, a.steps, kw) for d in dirs]
    t0 = time.time()
    res = []
    if a.jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                res.append(r)
                print(json.dumps(r), flush=True)
    else:
        for j in jobs:
            r = run_one(j)
            res.append(r)
            print(json.dumps(r), flush=True)
    res.sort(key=lambda r: r["scenario"])
    score = -float(np.mean([r["mean"] for r in res]))
    summ = {
        "policy": a.policy,
        "kwargs": kw,
        "set": a.set,
        "steps": a.steps,
        "n": len(res),
        "score": score,
        "comfort": -float(np.mean([r["comfort_term"] for r in res])),
        "energy": -float(np.mean([r["energy_term"] for r in res])),
        "carbon": -float(np.mean([r["carbon_term"] for r in res])),
        "wall_min": (time.time() - t0) / 60,
    }
    print("SUMMARY", json.dumps(summ), flush=True)
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps({"summary": summ, "scenarios": res}, indent=1))


if __name__ == "__main__":
    main()
