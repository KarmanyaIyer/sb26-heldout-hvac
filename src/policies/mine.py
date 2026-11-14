"""Our variants on top of the public controller chain in penguin.py (authors credited there and in NOTICE).

- MinePolicy: gas_cost_ratio 0.5 (the reward's marginal gas-to-electricity price is 0.48 to 0.55 per watt)
  and the hot-water plant constants (hold time after heat demand, reservoir floor) exposed as parameters.
- OptStartPolicy: deficit-scaled start-up. The `comfort_mode_soon` flag only looks 60 min ahead; we compute
  the hours to the next weekday comfort period from the cyclic time features and hold each zone to the
  comfort band (full margin) once that time is within os_lead0 + os_lead_per_k * deficit, capped at os_lead_max.
  Its defaults are the steep setting from the paper's ablation; V2aPolicy / V2bPolicy carry the final values.
- V2aPolicy: the paper's controller (start-up 0.25 h + 0.05 h/K capped at 3 h, unoccupied cooling ceiling
  303 K, 2 h hot-water hold). V2bPolicy: the same with the public chain's 72 h hold.
"""
import math

import numpy as np

from policies.penguin import NearMinimumPumpPressurePolicy


class MinePolicy(NearMinimumPumpPressurePolicy):
    """Exposes the hot-water plant constants (hold time after heat demand, reservoir floor) as parameters."""

    name = "mine"
    DEFAULTS = dict(NearMinimumPumpPressurePolicy.DEFAULTS, gas_cost_ratio=0.5,
                    water_hold_hours=72.0, water_floor_k=320.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.HOLD_STEPS = int(round(float(self.parameters["water_hold_hours"]) * 12))
        self.MIN_HEATING_TARGET_K = float(self.parameters["water_floor_k"])


class OptStartPolicy(MinePolicy):
    """Deficit-proportional optimal start beyond the 60-minute look-ahead flag."""

    name = "mine_optstart"
    DEFAULTS = dict(MinePolicy.DEFAULTS, os_lead0=0.5, os_lead_per_k=0.35, os_lead_max=10.0)

    def _dow(self, obs):
        s = self.field_index.get("dow_sin_000")
        c = self.field_index.get("dow_cos_000")
        if s is None or c is None:
            return None
        return int(round((math.atan2(float(obs[s]), float(obs[c])) % (2.0 * math.pi)) * 7.0 / (2.0 * math.pi))) % 7

    def _hours_to_comfort(self, obs):
        dow = self._dow(obs)
        if dow is None:
            return None
        hour = self._hour(obs)
        start, end = self.start_hour, self.end_hour
        for k in range(8):
            d = (dow + k) % 7
            if d >= 5:
                continue
            if k == 0:
                if hour < start:
                    return start - hour
                if hour < end:
                    return 0.0
                continue
            return k * 24.0 + start - hour
        return None

    def _band(self, obs, temperature):
        low, high, mode = super()._band(obs, temperature)
        if mode:
            return low, high, mode
        h = self._hours_to_comfort(obs)
        if h is None or h <= 0.0 or h > float(self.parameters["os_lead_max"]):
            return low, high, mode
        margin = float(self.parameters["comfort_margin"])
        cl = self.comfort_band[0] + margin
        ch = self.comfort_band[1] - margin
        deficit = np.maximum(np.maximum(cl - temperature, temperature - ch), 0.0)
        lead = float(self.parameters["os_lead0"]) + float(self.parameters["os_lead_per_k"]) * deficit
        active = lead >= h
        if np.any(active):
            low = low.copy()
            high = high.copy()
            low[active] = np.maximum(low[active], cl)
            high[active] = np.minimum(high[active], ch)
        return low, high, mode


class V2aPolicy(OptStartPolicy):
    """The paper's final controller (v2a)."""

    name = "v2a"
    DEFAULTS = dict(OptStartPolicy.DEFAULTS, os_lead0=0.25, os_lead_per_k=0.05, os_lead_max=3.0,
                    standby_maximum=303.0, water_hold_hours=2.0)


class V2bPolicy(V2aPolicy):
    """v2a with the public chain's 72 h hot-water hold (v2b)."""

    name = "v2b"
    DEFAULTS = dict(V2aPolicy.DEFAULTS, water_hold_hours=72.0)
