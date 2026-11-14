# Controller Improvement or Simulator Artifact? Held-Out Evaluation of One-Step MPC for HVAC

Code and per-scenario results for the paper of the same title (NeurIPS 2026 Smart Buildings Challenge, Round 1). The PDF is in `paper/paper.pdf`.

## What is here

| Path | Purpose |
|---|---|
| `src/rollout.py` | Rollout harness that mirrors the Kaggle gateway (same scenario loader, metadata builder and occupancy normalization), records the comfort / energy / carbon terms and an hour-of-week loss profile. |
| `src/make_val.py` | Builds **val16**: 8 generated buildings rescaled into the public size range x 2 weather strips from station months no public scenario uses. |
| `src/marginal.py` | Marginal reward per watt of electricity vs natural gas, read from a scenario's reward function. |
| `src/lambda_check.py` | Counts how often the controller's AHU choice changes when its cost weights change (output in `results/probes/lam_*.log`). |
| `src/occ_log.py`, `src/occ_analyze.py` | Tests whether occupant sensible heat is visible in the per-zone model residual (output in `results/probes/occ_analysis.txt`). |
| `src/agg.py`, `src/boot.py` | Paired per-scenario deltas against a reference run, and paired building-bootstrap intervals. |
| `src/tables.py`, `src/figures.py` | The paper's tables and Figure 1, generated from `results/`. |
| `src/policies/penguin.py` | The public controller chain we start from (see Attribution). |
| `src/policies/mine.py` | Our variants: `MinePolicy` (gas price 0.5, parameterized hot-water hold and floor), `OptStartPolicy` (adds deficit-scaled start-up), and the paper's two final controllers `V2aPolicy` and `V2bPolicy`. |
| `src/policies/occ.py`, `src/policies/exploit_oracles.py` | **Analysis only**: occupancy oracles that read the simulator's internal occupancy model. They cannot run in a submission. |
| `src/policies/exploit_controls.py`, `src/policies/optprobe.py` | Analysis controls: the schedule-only control matched to the future-occupancy oracle, and the start-up decomposition probe. |
| `patches/kit-fixed.patch` | The exact simulator change used for the "corrected" simulator (see below). |
| `results/` | One JSON per run: summary plus per-scenario score, reward decomposition, runtime and hour-of-week loss. `exploit_aggregates.json` holds the Table 3 numbers. |
| `paper/` | LaTeX source, tables, figure and PDF. |

### Run names in `results/`

| Name | Meaning |
|---|---|
| `penguin_*` | The public chain (`policies.penguin:NearMinimumPumpPressurePolicy`). |
| `g50_val_fixed` | v1: the public chain with gas price 0.5 (equivalent to `MinePolicy`), the controller of our first leaderboard submission. |
| `v2a_*`, `v2b_*` | The final controllers (`V2aPolicy`, `V2bPolicy`). |
| `cc2x_val_fixed` | The public chain with twice the comfort price (cost-weight ablation). |
| `pre2_val_fixed` | The public chain with 2 h pre-conditioning (explored, not reported in the paper). |
| `*_val_fixed`, `*_val_shipped` | Full 30-day val16 runs on the corrected and the shipped simulator. |
| `*_train30_*` | The 30 public scenarios. |
| `s10_*` | 10-day val16 screens on the corrected simulator (ablations). |
| `s10sh_*` | 10-day val16 screens on the shipped simulator. |
| `fid_atl2` | Harness fidelity check against a Kaggle-scored run. |
| `v2a307_train30_fixed` | v2a without the 303 K ceiling change (post-hoc diagnosis on the public scenarios). |
| `ph_*` | Phoenix-only attribution runs on the public scenarios: ceiling change alone (`ph_smax303`), start-up rule alone (`ph_opt`), v2a without the ceiling (`ph_v2a307`). |

The `summary.kwargs` field of every JSON records the exact parameters of that run.

## Reproducing

All commands use the starter kit's own environment (Python 3.12 or newer, created by `uv sync`). Below, `$REPO` is this repository and `$KIT` / `$KITFIX` are the two simulator checkouts.

1. Simulators. Both are built from starter-kit commit `95cea5b1c57b5291f44baf0283e0a081d53f5b16`.
   ```
   git clone https://github.com/smart-buildings-2026/starter-kit $KIT
   git -C $KIT checkout 95cea5b1c57b5291f44baf0283e0a081d53f5b16
   (cd $KIT && uv sync)
   cp -r $KIT $KITFIX
   git -C $KITFIX apply $REPO/patches/kit-fixed.patch
   ```
   `kit-fixed.patch` is the combined source change of the starter kit's open pull requests #2 (energy-balanced hot-water tank, head `ec27d9e`) and #3 (airflow-weighted return air, head `52b6f7d`), without their test files.
2. val16, once per kit (shown for `$KITFIX`; repeat with `$KIT`):
   ```
   cd $KITFIX
   .venv/bin/python -m smart_buildings.run_floor_generator --start-seed 600 --count 16 --no-pdf --output-dir smart_buildings/buildings_val
   cd $REPO
   SB_KIT=$KITFIX $KITFIX/.venv/bin/python src/make_val.py
   ```
   This writes `smart_buildings/val_scenarios/` inside the kit (8 buildings that pass the generator's minimum-zone check, 2 weather strips each).
3. Run a policy (from `$REPO`). The public chain and our final controller (v2a):
   ```
   SB_KIT=$KITFIX $KITFIX/.venv/bin/python src/rollout.py --policy policies.penguin:NearMinimumPumpPressurePolicy \
       --set $KITFIX/smart_buildings/val_scenarios --jobs 8 --out results/penguin_val_fixed.json
   SB_KIT=$KITFIX $KITFIX/.venv/bin/python src/rollout.py --policy policies.mine:V2aPolicy \
       --set $KITFIX/smart_buildings/val_scenarios --jobs 8 --out results/v2a_val_fixed.json
   ```
   Use `policies.mine:V2bPolicy` for v2b. `--steps 2880` gives the 10-day screens used for ablations; `--set train30` runs the 30 public scenarios of the selected kit.
4. Compare and tabulate (from `$REPO`):
   ```
   $KITFIX/.venv/bin/python src/agg.py --ref penguin_val_fixed
   $KITFIX/.venv/bin/python src/boot.py penguin_val_fixed v2a_val_fixed
   $KITFIX/.venv/bin/python src/tables.py
   $KITFIX/.venv/bin/python src/figures.py
   ```

The simulator is deterministic, so a run reproduces exactly given the same kit commit, patch and scenario files. Table 3 (occupancy oracles and the tank-sawtooth rule) ships as aggregates in `results/exploit_aggregates.json`; the oracles are included as code, and the sawtooth rule is specified in the paper (Section 6) rather than shipped.

## Attribution

`src/policies/penguin.py` contains a public controller by Daniel Wu (https://www.kaggle.com/code/danielwuu/cpu-b-conditional-hotwater-20261022) and the supervisory extensions from the public notebook by penguin069 (https://www.kaggle.com/code/penguin069/smart-buildings-best-water-pump-safety001). Both were shared publicly on Kaggle for this competition, where the competition rules license publicly shared competition code under an Open Source Initiative approved license that does not limit commercial use, and Kaggle publishes public notebooks under Apache-2.0 by default. They are included unmodified apart from packaging, and remain the work of their authors. `patches/kit-fixed.patch` is derived from the Apache-2.0 starter kit and the pull requests named above; see `NOTICE`.

## License

Apache-2.0 for the code in this repository (see `LICENSE` and `NOTICE`). The starter kit is Apache-2.0. Competition data and generated scenarios are not redistributed; they are rebuilt from the starter kit, and the competition data is licensed CC BY-NC-SA 4.0.
