"""ANALYSIS ONLY: occupancy oracles on top of MinePolicy, to measure what specific uses of per-zone occupancy are worth.
They read the simulator's occupancy model through rollout.py's wants_env hook and are never submitted.
"""
import copy

import numpy as np
import pandas as pd

from policies.mine import MinePolicy


class _OccOracle(MinePolicy):
    wants_env = True

    def _current(self):
        model = self.env.building.occupancy
        now = self.env.building.current_timestamp
        dt = pd.Timedelta(minutes=5)
        ids = [z.replace("VAV_", "room_") for z in self.zone_ids]
        if model._occupancy_cache_key == (now, now + dt) and all(z in model._occupancy_cache for z in ids):
            return np.array([model._occupancy_cache[z] for z in ids])
        model = copy.deepcopy(model)
        return np.array([model.average_zone_occupancy(z, now, now + dt) for z in ids])


class DepartureOracle(_OccOracle):
    """Relax a zone to standby once its occupants have left for the day (perfect departure detection)."""

    name = "oracle_departure"

    def _band(self, obs, temperature):
        low, high, mode = super()._band(obs, temperature)
        hour = self._hour(obs)
        if not hasattr(self, "_seen") or hour < 1.0:
            self._seen = np.zeros(self.n_zones, bool)
        if not mode:
            return low, high, mode
        occ = self._current() > 0
        self._seen |= occ
        gone = self._seen & ~occ & (hour >= 13.0)
        if np.any(gone):
            low = low.copy()
            high = high.copy()
            low[gone] = self.parameters["standby_minimum"]
            high[gone] = self.parameters["standby_maximum"]
        return low, high, mode
