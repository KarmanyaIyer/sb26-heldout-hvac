"""ANALYSIS: how often does the AHU candidate choice change with gas_cost_ratio / comfort_cost?

SB_KIT=... python src/lambda_check.py <scenario_dir> <steps>
Runs MinePolicy; at every AHU decision also evaluates the same search with alternative weights.
"""
import os
import pathlib
import sys

import numpy as np

sys.argv, args = sys.argv[:1], sys.argv[1:]
import rollout  # noqa: E402
from smart_buildings import evaluate  # noqa: E402
from policies.mine import MinePolicy  # noqa: E402

ALTS = {"gas0.3": {"gas_cost_ratio": 0.3}, "gas1.0": {"gas_cost_ratio": 1.0}, "gas0": {"gas_cost_ratio": 0.0},
        "cc2x": {"comfort_cost": 130000.0}, "cc0.25x": {"comfort_cost": 16250.0}}


class Probe(MinePolicy):
    def reset(self):
        super().reset()
        self.n_dec = 0
        self.diff = {k: 0 for k in ALTS}
        self.n_heat = 0

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        base = super()._choose_ahu(group, temperature, free, low, high, max_flow, outside, water)
        self.n_dec += 1
        self.n_heat += int(np.sum(base["reheat"]) > 1.0)
        saved = dict(self.parameters)
        req = self.required_water_target
        for k, alt in ALTS.items():
            self.parameters.update(alt)
            c = MinePolicy._choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water)
            self.parameters.clear()
            self.parameters.update(saved)
            if (c["run"] != base["run"] or abs(c["setpoint"] - base["setpoint"]) > 1e-6
                    or abs(c["recirc"] - base["recirc"]) > 1e-6 or abs(c["pressure"] - base["pressure"]) > 1e-6):
                self.diff[k] += 1
        self.required_water_target = req
        return base


scen, steps = pathlib.Path(args[0]), int(args[1])
cwd = os.getcwd()
os.chdir(rollout.KIT)
env = evaluate.load_hybrid_environment(scen / "config.gin")
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = evaluate.create_building_meta_data(env, scen)
os.chdir(cwd)
p = Probe(meta, env.action_spec())
obs, _ = env.reset()
for _ in range(steps):
    a = p.get_action(obs)
    obs, *_ = env.step({"continuous_action": np.asarray(a["continuous_action"], np.float32),
                        "discrete_action": np.asarray(a["discrete_action"], np.int32)})
print(scen.parent.name + "/" + scen.name, "decisions", p.n_dec, "with_reheat", p.n_heat,
      {k: f"{v / max(p.n_dec, 1):.4f}" for k, v in p.diff.items()}, flush=True)
