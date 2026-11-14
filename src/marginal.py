"""Marginal reward per watt of electricity vs natural gas, straight from the scenario's reward function.

python src/marginal.py [scenario_dir]
Prints d(-reward)/dW per AHU for each fuel by hour of a weekday and the gas/electric ratio.
"""
import os
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
KIT = pathlib.Path(os.environ.get("SB_KIT", HERE.parent / "ext" / "kit-fixed"))
sys.path.insert(0, str(KIT))
from smart_buildings import evaluate  # noqa: E402

scen = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else KIT / "smart_buildings/training_scenarios/building_train_1/atlanta_2"
os.chdir(KIT)
env = evaluate.load_hybrid_environment(scen / "config.gin")
rf = env.reward_function
n_ahu = rf._max_electricity_rate / 80000.0
E, G = rf._electricity_energy_cost, rf._natural_gas_energy_cost
rows = []
for h in range(24):
    t0 = pd.Timestamp("2026-01-14 00:00", tz="UTC") + pd.Timedelta(hours=h)  # a Wednesday
    t1 = t0 + pd.Timedelta(minutes=5)
    me_c, mg_c = E.cost(t0, t1, rf._max_electricity_rate), G.cost(t0, t1, rf._max_natural_gas_rate)
    me_k, mg_k = E.carbon(t0, t1, rf._max_electricity_rate), G.carbon(t0, t1, rf._max_natural_gas_rate)
    d = 1000.0  # +1 kW
    de = 0.2 * (E.cost(t0, t1, d) / (me_c + mg_c) + E.carbon(t0, t1, d) / (me_k + mg_k)) / d
    dg = 0.2 * (G.cost(t0, t1, d) / (me_c + mg_c) + G.carbon(t0, t1, d) / (me_k + mg_k)) / d
    rows.append((h, de * n_ahu, dg * n_ahu, dg / de))
print(f"n_ahu={n_ahu:.0f}  (per-AHU marginal reward per W; ratio gas/electric)")
for h, de, dg, r in rows:
    print(f"{h:2d}h  elec={de:.3e}  gas={dg:.3e}  ratio={r:.3f}")
rs = [r for *_, r in rows]
print(f"ratio range {min(rs):.3f}-{max(rs):.3f}")
