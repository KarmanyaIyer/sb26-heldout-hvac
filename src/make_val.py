"""Build the held-out validation set: generated buildings rescaled into the shipped size
range, paired with weather strips that no shipped training scenario uses.

Step 1 (raw buildings, run inside the kit):
  python -m smart_buildings.run_floor_generator --start-seed 600 --count 16 --no-pdf \
      --output-dir smart_buildings/buildings_val
Step 2: python src/make_val.py  (uses the kit venv; SB_KIT selects the kit)
Writes building YAMLs to <kit>/smart_buildings/buildings_v/ and scenarios to
<kit>/smart_buildings/val_scenarios/<building>/<weather>/.
"""
import os
import pathlib
import subprocess
import sys

import yaml

KIT = pathlib.Path(os.environ.get("SB_KIT", pathlib.Path(__file__).resolve().parents[1] / "ext" / "starter-kit"))
SRC = KIT / "smart_buildings" / "buildings_val"
DST = KIT / "smart_buildings" / "buildings_v"
OUT = KIT / "smart_buildings" / "val_scenarios"
DST.mkdir(exist_ok=True)

# Shipped six: width 318-430, height 188-301. Generator default: 400-700, 300-500.
def rescale(v, lo0, hi0, lo1, hi1):
    return int(round(lo1 + (v - lo0) / (hi0 - lo0) * (hi1 - lo1)))

WEATHER_PAIRS = [
    ["fairbanks_6", "phoenix_7"],
    ["minneapolis_1", "singapore_5"],
    ["denver_4", "atlanta_7"],
    ["fairbanks_3", "minneapolis_5"],
    ["atlanta_4", "minneapolis_8"],
    ["phoenix_2", "minneapolis_3"],
    ["singapore_2", "denver_1"],
    ["minneapolis_6", "minneapolis_2"],
]

candidates = sorted(p.stem for p in SRC.glob("building_*.yaml"))
pi = 0
for name in candidates:
    if pi >= len(WEATHER_PAIRS):
        break
    cfg = yaml.safe_load((SRC / f"{name}.yaml").read_text())
    fp = cfg["floor_plan"]
    fp["width"] = rescale(fp["width"], 400, 700, 318, 430)
    fp["height"] = rescale(fp["height"], 300, 500, 188, 301)
    (DST / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ok = True
    for w in WEATHER_PAIRS[pi]:
        if (OUT / name / w / "config.gin").exists():
            continue
        r = subprocess.run(
            [sys.executable, "-m", "smart_buildings.run_scenario_generator",
             "--buildings", str(DST / f"{name}.yaml"), "--weathers", w,
             "--output-dir", str(OUT)],
            cwd=KIT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            ok = False
            print(name, "FAILED:", r.stderr.strip().splitlines()[-1][:160], flush=True)
            break
    if ok:
        # generated configs embed absolute kit paths; make them kit-relative so the set is portable
        for w in WEATHER_PAIRS[pi]:
            g = OUT / name / w / "config.gin"
            g.write_text(g.read_text().replace(str(KIT) + "/", ""))
        print(name, fp["width"], fp["height"], fp["num_floors"], WEATHER_PAIRS[pi], flush=True)
        pi += 1
print("built", pi, "of", len(WEATHER_PAIRS))
