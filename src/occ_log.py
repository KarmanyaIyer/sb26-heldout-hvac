"""ANALYSIS: log per-zone model residuals next to true zone occupancy, to check if occupant heat is detectable.

SB_KIT=... python src/occ_log.py <scenario_dir> <steps> <out.npz>
"""
import os
import pathlib
import sys

import numpy as np

sys.argv, args = sys.argv[:1], sys.argv[1:]
import rollout  # noqa: E402  (sets sys.path / KIT)
from smart_buildings import evaluate  # noqa: E402
from policies.occ import _OccOracle  # noqa: E402

scen, steps, out = pathlib.Path(args[0]), int(args[1]), args[2]
cwd = os.getcwd()
os.chdir(rollout.KIT)
env = evaluate.load_hybrid_environment(scen / "config.gin")
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = evaluate.create_building_meta_data(env, scen)
os.chdir(cwd)
p = _OccOracle(meta, env.action_spec())
p.env = env
obs, _ = env.reset()
R, TH, OCC, N, H = [], [], [], [], []
ni = p.field_index["num_occupants"]
for i in range(steps):
    a = p.get_action(obs)
    R.append(p.residual.copy())
    TH.append(p.theta[:, 5].copy())
    OCC.append(p._current())
    N.append(float(obs[ni]))
    H.append(p._hour(obs))
    obs, *_ = env.step({"continuous_action": np.asarray(a["continuous_action"], np.float32),
                        "discrete_action": np.asarray(a["discrete_action"], np.int32)})
np.savez(out, R=np.array(R), TH=np.array(TH), OCC=np.array(OCC), N=np.array(N), H=np.array(H))
print("saved", out)
