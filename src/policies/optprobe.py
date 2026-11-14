"""ANALYSIS ONLY: decompose where the optimal-start rule changes the target band.

OptStartProbe behaves exactly like OptStartPolicy and counts zone-steps whose band it changes,
split into steps inside the published one-hour comfort-soon window and steps earlier than it.
OptStartEarlyOnly applies the rule only earlier than the one-hour window.
"""
import sys

import numpy as np

from policies.mine import MinePolicy, OptStartPolicy


class OptStartProbe(OptStartPolicy):
    name = "mine_optstart_probe"

    def _band(self, obs, temperature):
        b_low, b_high, b_mode = MinePolicy._band(self, obs, temperature)
        low, high, mode = super()._band(obs, temperature)
        if not hasattr(self, "_cnt"):
            self._cnt = {"steps": 0, "in_hour": 0, "early": 0, "early_steps": 0}
        c = self._cnt
        c["steps"] += 1
        changed = (low != b_low) | (high != b_high)
        n = int(np.count_nonzero(changed))
        if n:
            h = self._hours_to_comfort(obs)
            if h is not None and h > 1.0:
                c["early"] += n
                c["early_steps"] += 1
            else:
                c["in_hour"] += n
        if c["steps"] % 288 == 0:
            print("OPTPROBE", c["steps"], c["in_hour"], c["early"], c["early_steps"], file=sys.stderr, flush=True)
        return low, high, mode


class OptStartEarlyOnly(OptStartPolicy):
    name = "mine_optstart_early_only"

    def _band(self, obs, temperature):
        h = self._hours_to_comfort(obs)
        if h is not None and 0.0 < h <= 1.0:
            return MinePolicy._band(self, obs, temperature)
        return super()._band(obs, temperature)
