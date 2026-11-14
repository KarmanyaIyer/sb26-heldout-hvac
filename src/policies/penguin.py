# Public controller by Daniel Wu, reviewed 2026-11-03.
# Source: https://www.kaggle.com/code/danielwuu/cpu-b-conditional-hotwater-20261022
# Embedded policy.py SHA256: 185eccbe60fdfc3728b03ecf834de9a2e58943d241d25ce16a1012a043bb33f6
# Preserved unmodified below; reference-only benchmark, no learned weights.
"""Adaptive, physics-informed HVAC policy for the Smart Buildings challenge.

The only runtime dependency is NumPy.  All learned state belongs to the current
episode.  No building IDs, station IDs, hidden files, or weather traces are used.

The controller identifies a small thermal model for every zone, compensates for
the simulator's delayed zone-temperature observation, then searches feasible
air-handler operating points.  Airflow and reheat are allocated independently to
each zone; fan and hot-water pressure are reset to the smallest useful values.
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Any, Mapping

import numpy as np


class TransferPolicy:
    """A transferable controller implementing the official two-method API."""

    name = "adaptive_physics"
    DEFAULTS = {
        "comfort_margin": 0.12,
        "precondition_hours": 1.0,
        "recovery_steps": 1.5,
        "forgetting_factor": 0.997,
        "model_residual_gain": 0.16,
        "comfort_cost": 65000.0,
        "gas_cost_ratio": 0.30,
        "hot_water_target": 345.0,
        "standby_minimum": 282.0,
        "standby_maximum": 307.0,
        "observation_delay": 1,
        "response_scale": 0.75,
        "disturbance_rate": 0.35,
    }
    AIR_CAPACITY = 1.2 * 1006.0  # volumetric heat capacity, J/(m3 K)
    WATER_VALVE_CAPACITY = 4180.0 * 1000.0 * 0.108 / 3600.0
    DESIGN_PRESSURE = 2000.0
    MIN_PRESSURE = 200.0
    MIN_DAMPER = 0.10
    MAX_RECIRCULATION = 0.70
    MAX_DISCHARGE_K = 308.15

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any = None,
        parameters: Mapping[str, float] | None = None,
    ) -> None:
        self.metadata = building_meta_data
        self.action_spec = action_spec
        self.parameters = dict(self.DEFAULTS)
        if parameters:
            unknown = set(parameters) - set(self.parameters)
            if unknown:
                raise ValueError(f"Unknown policy parameters: {sorted(unknown)}")
            self.parameters.update(parameters)
        self.n_continuous = int(building_meta_data["n_continuous_action"])
        self.n_discrete = int(building_meta_data["n_discrete_action"])
        self.field_names = [str(x) for x in building_meta_data["observation_field_names"]]
        self.field_index = {name: i for i, name in enumerate(self.field_names)}
        self.obs_norm = building_meta_data.get("observation_normalization_metadata", {})
        self.action_norm = building_meta_data.get("action_normalization_metadata", {})
        self.temp_indices = list(building_meta_data["temp_indices"])
        self.pressure_indices = list(building_meta_data["pressure_indices"])
        self.recirculation_indices = list(building_meta_data["recirculation_indices"])
        self.water_indices = list(building_meta_data["water_indices"])
        self.num_ahus = len(self.temp_indices)
        zone_actions = building_meta_data["zone_action_indices"]
        self.zone_ids = sorted(zone_actions, key=self._natural_key)
        self.n_zones = len(self.zone_ids)
        if not self.n_zones:
            raise ValueError("Building metadata contains no zone actuators")
        self.zone_number = {name: i for i, name in enumerate(self.zone_ids)}
        self.zone_obs = np.array([
            self.field_index.get(f"{z}_zone_air_temperature_sensor", -1)
            for z in self.zone_ids
        ], dtype=int)
        if np.any(self.zone_obs < 0):
            raise ValueError("A zone actuator has no matching temperature observation")
        self.zone_low_obs = np.array([
            self.field_index.get(f"{z}_zone_air_heating_temperature_setpoint_sensor", -1)
            for z in self.zone_ids
        ], dtype=int)
        self.zone_high_obs = np.array([
            self.field_index.get(f"{z}_zone_air_cooling_temperature_setpoint_sensor", -1)
            for z in self.zone_ids
        ], dtype=int)
        self.zone_flow_obs = np.array([
            self.field_index.get(f"{z}_supply_air_flowrate_setpoint", -1)
            for z in self.zone_ids
        ], dtype=int)
        self.damper_indices = np.array([
            zone_actions[z]["supply_air_damper_percentage_command"] for z in self.zone_ids
        ], dtype=int)
        self.valve_indices = np.array([
            zone_actions[z]["reheat_valve_percentage_command"] for z in self.zone_ids
        ], dtype=int)

        assignments = building_meta_data.get(
            "ahu_zone_assignments", building_meta_data.get("aha_zone_assignments", {})
        )
        self.ahu_ids = sorted(assignments, key=self._natural_key)
        if len(self.ahu_ids) != self.num_ahus:
            raise ValueError("AHU action indices and zone assignments disagree")
        self.groups = []
        self.zone_ahu = np.zeros(self.n_zones, dtype=int)
        self.outside_obs = []
        for i, ahu in enumerate(self.ahu_ids):
            group = np.array([
                self.zone_number[str(room).replace("room_", "VAV_")]
                for room in assignments[ahu]
                if str(room).replace("room_", "VAV_") in self.zone_number
            ], dtype=int)
            self.groups.append(group)
            self.zone_ahu[group] = i
            self.outside_obs.append(self._find_field(
                f"ahs_{ahu}_outside_air_temperature_sensor",
                f"{ahu}_outside_air_temperature_sensor",
                suffix="outside_air_temperature_sensor",
            ))
        self.water_sensor = self._find_field(suffix="supply_water_temperature_sensor")
        self.comfort_now_index = self.field_index.get("comfort_mode_now")
        self.comfort_soon_index = self.field_index.get("comfort_mode_soon")
        self.occupancy_index = self.field_index.get("num_occupants")
        self.comfort_band = tuple(map(float, building_meta_data["comfort_temp_window"]))
        self.eco_band = tuple(map(float, building_meta_data["eco_temp_window"]))
        self.start_hour = float(building_meta_data.get("morning_start_hour", 6))
        self.end_hour = float(building_meta_data.get("evening_start_hour", 19))

        self.zone_area, self.zone_ua = self._geometry_priors(building_meta_data.get("zone_map"))
        # The five-minute effective capacitance includes air, near-surface
        # furnishings, and the part of the fabric coupled over one time step.
        effective_capacity = 3800.0 * self.zone_area + 21000.0 * np.sqrt(self.zone_area)
        self.initial_gain = np.clip(300000.0 / effective_capacity, 0.08, 2.5)
        self.reset()

    @staticmethod
    def _natural_key(value: str) -> tuple:
        return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(value)))

    def _find_field(self, *exact: str, suffix: str = "") -> int | None:
        for name in exact:
            if name in self.field_index:
                return self.field_index[name]
        if suffix:
            for name, index in self.field_index.items():
                if name.endswith(suffix):
                    return index
        return None

    def _geometry_priors(self, zone_map: Any) -> tuple[np.ndarray, np.ndarray]:
        """Read public floor geometry without importing a vision dependency.

        Official maps encode interior=0, wall=1, exterior=2; room numbering is
        the row-major order of four-connected interior components.
        """
        areas = np.full(self.n_zones, 55.0)
        ua = np.full(self.n_zones, 18.0)
        if zone_map is None:
            return areas, ua
        grid = np.asarray(zone_map)
        if grid.ndim != 2 or grid.size > 5000000:
            return areas, ua
        height, width = grid.shape
        inside = (grid == 0).ravel()
        visited = np.zeros(grid.size, dtype=bool)
        flat = grid.ravel()
        counts, external_edges = [], []
        for start in np.flatnonzero(inside):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            count = 0
            edge = 0
            while stack:
                p = stack.pop()
                count += 1
                row, col = divmod(p, width)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    q = nr * width + nc
                    if inside[q] and not visited[q]:
                        visited[q] = True
                        stack.append(q)
                    elif flat[q] == 2:
                        edge += 1
                    elif flat[q] == 1:
                        rr, cc = row + 2 * dr, col + 2 * dc
                        if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] == 2:
                            edge += 1
            counts.append(count)
            external_edges.append(edge)
        if len(counts) != self.n_zones:
            return areas, ua
        for j, name in enumerate(self.zone_ids):
            match = re.search(r"(\d+)$", name)
            label = int(match.group(1)) - 1 if match else j
            if 0 <= label < len(counts):
                areas[j] = max(1.0, 0.04 * counts[label])
                ua[j] = max(0.5, external_edges[label] * 0.2 * 3.0 * 1.3)
        return areas, ua

    def reset(self) -> None:
        self.step_count = 0
        self.history: deque[dict[str, Any]] = deque(maxlen=3)
        self.previous_observed = None
        self.slow_temperature = None
        self.residual = np.zeros(self.n_zones)
        self.model_error = np.full(self.n_zones, 0.08)
        self.theta = np.zeros((self.n_zones, 6))
        self.theta[:, 0] = self.initial_gain * self.zone_area * 2.0 / 1000.0
        self.theta[:, 1] = self.initial_gain * self.zone_ua * 20.0 / 1000.0
        self.theta[:, 2] = 0.18
        self.theta[:, 3] = 0.08
        self.theta[:, 4] = self.initial_gain * self.zone_area * 8.0 / 1000.0
        self.theta[:, 5] = self.initial_gain * self.parameters["response_scale"]
        self.theta[:, :2] *= self.parameters["response_scale"]
        self.theta[:, 4] *= self.parameters["response_scale"]
        prior = np.array([0.12, 0.20, 0.15, 0.15, 0.16, 0.8])
        self.covariance = np.repeat(np.diag(prior)[None, :, :], self.n_zones, axis=0)
        self.last_heating_step = -1000
        self.last_action = None

    def _temperature(self, obs: np.ndarray, index: int | None, kind: str, fallback: float) -> float:
        if index is None:
            return fallback
        value = float(obs[index])
        # In the published normalizer, AHU-prefixed measurement names are
        # exact-match misses and remain native.  Physical-range detection also
        # supports a future gateway that normalizes those fields correctly.
        if 150.0 < value < 450.0:
            return value
        norm = self.obs_norm.get(kind, {})
        mean = float(norm.get("sample_mean", fallback))
        std = float(norm.get("sample_std", math.sqrt(norm.get("sample_variance", 1.0))))
        value = value * std + mean
        return value if np.isfinite(value) and 150.0 < value < 450.0 else fallback

    def _zone_temperatures(self, obs: np.ndarray) -> np.ndarray:
        norm = self.obs_norm.get("zone_air_temperature_sensor", {})
        mean = float(norm.get("sample_mean", 295.0))
        std = float(norm.get("sample_std", math.sqrt(norm.get("sample_variance", 25.0))))
        raw = obs[self.zone_obs].astype(float)
        value = np.where((raw > 150.0) & (raw < 450.0), raw, raw * std + mean)
        fallback = self.previous_observed if self.previous_observed is not None else mean
        return np.where(np.isfinite(value) & (value > 150.0) & (value < 450.0), value, fallback)

    def _hour(self, obs: np.ndarray) -> float:
        sine = self.field_index.get("hod_sin_000")
        cosine = self.field_index.get("hod_cos_000")
        if sine is None or cosine is None:
            return (self.start_hour + self.step_count / 12.0) % 24.0
        return (math.atan2(float(obs[sine]), float(obs[cosine])) % (2.0 * math.pi)) * 12.0 / math.pi

    def _band(self, obs: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        now = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else (
            self.start_hour <= self._hour(obs) < self.end_hour
        )
        soon = bool(obs[self.comfort_soon_index] > 0.5) if self.comfort_soon_index is not None else now
        default = self.comfort_band if now else self.eco_band
        low = np.full(self.n_zones, default[0])
        high = np.full(self.n_zones, default[1])
        valid = (self.zone_low_obs >= 0) & (self.zone_high_obs >= 0)
        if np.any(valid):
            sensor_low = obs[self.zone_low_obs[valid]].astype(float)
            sensor_high = obs[self.zone_high_obs[valid]].astype(float)
            good = (sensor_low > 150.0) & (sensor_high < 450.0) & (sensor_high > sensor_low)
            indices = np.flatnonzero(valid)[good]
            low[indices] = sensor_low[good]
            high[indices] = sensor_high[good]
        # Band sensors, like zone temperatures, are one step old at transitions.
        if now:
            low = np.maximum(low, self.comfort_band[0])
            high = np.minimum(high, self.comfort_band[1])
        elif soon:
            hour = self._hour(obs)
            remaining = max(0.0, (self.start_hour - hour) % 24.0)
            progress = np.clip(1.0 - remaining / max(0.1, self.parameters["precondition_hours"]), 0.0, 1.0)
            low = self.eco_band[0] + progress * (self.comfort_band[0] - self.eco_band[0]) + np.zeros(self.n_zones)
            high = self.eco_band[1] + progress * (self.comfort_band[1] - self.eco_band[1]) + np.zeros(self.n_zones)
        else:
            # With no occupants, comfort does not contribute to reward. Keep a
            # recoverable building but permit useful night/weekend setback.
            low[:] = self.parameters["standby_minimum"]
            high[:] = self.parameters["standby_maximum"]
        margin = self.parameters["comfort_margin"] if (now or soon) else 0.0
        low += margin
        high -= margin
        # Do not request the entire recovery in a single five-minute step.
        recovery = max(1.0, self.parameters["recovery_steps"])
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
        return low, high, now

    def _group_temperature(self, temperature: np.ndarray) -> np.ndarray:
        means = np.zeros(self.n_zones)
        for group in self.groups:
            if group.size:
                means[group] = np.average(temperature[group], weights=self.zone_area[group])
        return means

    def _features(self, temperature: np.ndarray, outside: np.ndarray, mode: bool, slow: np.ndarray) -> np.ndarray:
        return np.column_stack((
            np.ones(self.n_zones),
            np.clip((outside - temperature) / 20.0, -5.0, 5.0),
            np.clip((slow - temperature) / 5.0, -5.0, 5.0),
            np.clip((self._group_temperature(temperature) - temperature) / 5.0, -5.0, 5.0),
            np.full(self.n_zones, float(mode)),
        ))

    def _learn(self, observed: np.ndarray) -> None:
        delay = int(self.parameters["observation_delay"])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        old = self.previous_observed
        power = (applied["conductance"] * (applied["supply"] - observed) + applied["reheat"]) / 1000.0
        features = self._features(old, applied["outside"], applied["mode"], applied["slow"])
        x = np.column_stack((features, power))
        delta = np.clip(observed - old, -15.0, 15.0)
        prediction = np.einsum("ij,ij->i", self.theta, x)
        remaining = delta - prediction
        rate = float(self.parameters["disturbance_rate"])
        self.model_error = 0.94 * self.model_error + 0.06 * np.minimum(np.abs(remaining - self.residual), 2.0)
        self.residual = np.clip((1.0 - rate) * self.residual + rate * remaining, -5.0, 5.0)

    def _free_drift(self, temperature: np.ndarray, outside: np.ndarray, mode: bool) -> np.ndarray:
        features = self._features(temperature, outside, mode, self.slow_temperature)
        return np.clip(np.einsum("ij,ij->i", self.theta[:, :5], features) + self.residual, -5.0, 5.0)

    def _native_action(self, kind: str, value: Any, low: float, high: float) -> Any:
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((v for k, v in self.action_norm.items() if str(k).endswith(kind)), None)
        if spec:
            low = float(spec.get("min_native_value", low))
            high = float(spec.get("max_native_value", high))
            out_low = float(spec.get("min_normalized_value", -1.0))
            out_high = float(spec.get("max_normalized_value", 1.0))
        else:
            out_low, out_high = -1.0, 1.0
        return np.clip(out_low + (np.asarray(value) - low) * (out_high - out_low) / max(high - low, 1e-12), out_low, out_high)

    def _choose_ahu(self, group: np.ndarray, temperature: np.ndarray, free: np.ndarray,
                    low: np.ndarray, high: np.ndarray, max_flow: np.ndarray,
                    outside: float, water: float) -> dict[str, Any]:
        """Joint supply-temperature, economizer, airflow, and reheat search."""
        t, d, lo, hi = temperature[group], free[group], low[group], high[group]
        response = self.theta[group, 5]
        capacity = max_flow[group]
        inside = float(np.average(t, weights=self.zone_area[group]))
        targets = np.array([285.0, 287.0, 289.0, 291.0, 293.0, 295.0, 297.0, 299.0, 302.0, 305.0])
        targets = np.concatenate((targets, [float(np.clip(np.min(hi) - 1.5, 285.0, 305.0))]))
        setpoint = np.repeat(targets, 2)
        recirc = np.full_like(setpoint, self.MAX_RECIRCULATION)
        if outside < inside - 0.5:
            recirc[1::2] = np.clip((targets - outside) / max(inside - outside, 1e-6), 0.0, self.MAX_RECIRCULATION)
        mixed = recirc * inside + (1.0 - recirc) * outside
        supply = np.minimum(mixed, setpoint)
        sa = supply[:, None]
        # Invert the implicit thermal model at the two comfort boundaries.
        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / response
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / response
        cooling_flow = cool_need[None, :] / (self.AIR_CAPACITY * np.maximum(hi[None, :] - sa, 0.05))
        heating_flow = heat_need[None, :] / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - lo[None, :], 0.5))
        # Once the water coil saturates, more cold supply air reduces net
        # heating. The maximum useful heating flow occurs where the coil can
        # just lift the supply stream to the discharge-temperature limit.
        maximum_water_heat = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * np.maximum(water - sa, 0.0)
        maximum_heating_flow = maximum_water_heat / (
            self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - sa, 0.1)
        )
        heating_flow = np.minimum(heating_flow, maximum_heating_flow)
        needed_flow = np.maximum(cooling_flow, heating_flow)
        fraction = np.max(needed_flow / capacity[None, :], axis=1)
        pressure = np.clip(self.DESIGN_PRESSURE * fraction ** 2, self.MIN_PRESSURE, self.DESIGN_PRESSURE)
        pressure_ratio = np.sqrt(pressure / self.DESIGN_PRESSURE)
        dampers = np.clip(needed_flow / (capacity[None, :] * pressure_ratio[:, None]), self.MIN_DAMPER, 1.0)
        flow = dampers * capacity[None, :] * pressure_ratio[:, None]
        conductance = flow * self.AIR_CAPACITY
        water_gap = np.maximum(water - sa, 0.0)
        wanted_reheat = np.maximum((lo[None, :] - t[None, :] - d[None, :]) * 1000.0 / response[None, :]
                                    + conductance * (lo[None, :] - sa), 0.0)
        reheat_limit = np.minimum(
            conductance * np.maximum(self.MAX_DISCHARGE_K - sa, 0.0),
            self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * water_gap,
        )
        reheat = np.minimum(wanted_reheat, reheat_limit)
        predicted = (t[None, :] + d[None, :] + response[None, :] * (conductance * sa + reheat) / 1000.0) / (
            1.0 + response[None, :] * conductance / 1000.0
        )
        violation = np.maximum(lo[None, :] - predicted, 0.0) + np.maximum(predicted - hi[None, :], 0.0)
        thermal_penalty = np.sum(violation + 2.0 * violation ** 2, axis=1)
        total_flow = flow.sum(axis=1)
        fan = total_flow * pressure * (2.0 - recirc) / 0.9
        chiller = total_flow * self.AIR_CAPACITY * np.maximum(mixed - supply, 0.0) / 3.5
        gas = reheat.sum(axis=1)
        objective = fan + chiller + self.parameters["gas_cost_ratio"] * gas
        objective += self.parameters["comfort_cost"] * thermal_penalty
        # The off candidate removes the enforced ventilation floor when the
        # building's own thermal inertia can keep this AHU's zones comfortable.
        off_prediction = t + d
        off_violation = np.maximum(lo - off_prediction, 0.0) + np.maximum(off_prediction - hi, 0.0)
        off_objective = self.parameters["comfort_cost"] * np.sum(off_violation + 2.0 * off_violation ** 2)
        best = int(np.argmin(objective))
        if off_objective <= objective[best]:
            return {"run": 0, "pressure": 0.0, "setpoint": 305.0, "recirc": 0.7,
                    "damper": np.full(group.size, self.MIN_DAMPER), "conductance": np.zeros(group.size),
                    "supply": np.full(group.size, mixed[best]), "reheat": np.zeros(group.size)}
        return {"run": 1, "pressure": float(pressure[best]), "setpoint": float(setpoint[best]),
                "recirc": float(recirc[best]), "damper": dampers[best],
                "conductance": conductance[best], "supply": np.full(group.size, supply[best]),
                "reheat": reheat[best]}

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=float).reshape(-1)
        if obs.size != len(self.field_names):
            raise ValueError(f"Observation has {obs.size} fields; expected {len(self.field_names)}")
        observed = self._zone_temperatures(obs)
        if self.slow_temperature is None:
            self.slow_temperature = observed.copy()
        self._learn(observed)
        outside_ahu = np.array([
            self._temperature(obs, index, "outside_air_temperature_sensor", 290.0)
            for index in self.outside_obs
        ])
        outside = outside_ahu[self.zone_ahu]
        mode = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else True
        temperature = observed.copy()
        if self.history and int(self.parameters["observation_delay"]) == 1:
            last = self.history[-1]
            drift = self._free_drift(observed, last["outside"], last["mode"])
            gain = self.theta[:, 5] / 1000.0
            temperature = (observed + drift + gain * (last["conductance"] * last["supply"] + last["reheat"])) / (
                1.0 + gain * last["conductance"]
            )
            temperature = np.clip(temperature, observed - 8.0, observed + 8.0)
        low, high, mode = self._band(obs, temperature)
        free = self._free_drift(temperature, outside, mode)
        max_flow = np.full(self.n_zones, 2.0)
        valid = self.zone_flow_obs >= 0
        if np.any(valid):
            flow_reading = obs[self.zone_flow_obs[valid]]
            max_flow[valid] = np.where(np.isfinite(flow_reading) & (flow_reading > 0.0), flow_reading, 2.0)
        max_flow = np.clip(max_flow, 0.01, 100.0)
        water_observed = self._temperature(obs, self.water_sensor, "supply_water_temperature_sensor", 340.0)
        heat_demand = np.any(temperature + free < low)
        if heat_demand:
            self.last_heating_step = self.step_count
        recent_heat = self.step_count - self.last_heating_step < 24
        water_target = self.parameters["hot_water_target"] if recent_heat else 310.0
        water_available = water_observed
        continuous = np.full(self.n_continuous, -1.0, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        conductance = np.zeros(self.n_zones)
        supply = outside.copy()
        reheat = np.zeros(self.n_zones)
        for i, group in enumerate(self.groups):
            if not group.size:
                continue
            choice = self._choose_ahu(group, temperature, free, low, high, max_flow,
                                      float(outside_ahu[i]), float(water_available))
            discrete[i] = choice["run"]
            continuous[self.temp_indices[i]] = self._native_action(
                f"{self.ahu_ids[i]}_supply_air_temperature_setpoint", choice["setpoint"], 285.0, 305.0)
            continuous[self.pressure_indices[i]] = self._native_action(
                f"{self.ahu_ids[i]}_supply_air_static_pressure_setpoint", choice["pressure"], 0.0, 2000.0)
            continuous[self.recirculation_indices[i]] = self._native_action(
                f"{self.ahu_ids[i]}_recirculation_percentage_command", choice["recirc"], 0.0, 1.0)
            continuous[self.damper_indices[group]] = self._native_action(
                "supply_air_damper_percentage_command", choice["damper"], 0.0, 1.0)
            conductance[group] = choice["conductance"]
            supply[group] = choice["supply"]
            reheat[group] = choice["reheat"]
        if np.any(reheat > 10.0):
            # Reheat can also be needed to protect a satisfied zone from the
            # cold air supplied to its neighbour. Keep the boiler available in
            # that case even when there was no free-floating heating demand.
            self.last_heating_step = self.step_count
            water_target = self.parameters["hot_water_target"]
        # The valve law is Q = Kv * sqrt(dP) * rho_w * cp_w * (Tw - Ta).
        # Choose pressure from the most demanding valve, then allocate each
        # valve its own heat.  This avoids running the pump at 20 bar by habit.
        valve_need = reheat / (self.WATER_VALVE_CAPACITY * np.maximum(water_available - supply, 1.0))
        pressure_water = float(np.clip(np.max(valve_need) ** 2 * 1.05, 0.0, 20.0))
        if pressure_water > 1e-8:
            valves = np.clip(valve_need / math.sqrt(pressure_water), 0.0, 1.0)
            discrete[-1] = 1
        else:
            valves = np.zeros(self.n_zones)
        continuous[self.valve_indices] = self._native_action("reheat_valve_percentage_command", valves, 0.0, 1.0)
        if self.water_indices:
            continuous[self.water_indices[0]] = self._native_action("supply_water_temperature_setpoint", water_target, 310.0, 350.0)
        if len(self.pressure_indices) > self.num_ahus:
            continuous[self.pressure_indices[-1]] = self._native_action("differential_pressure_setpoint", pressure_water, 0.0, 20.0)
        # Record the physically delivered quantity, including actuator caps.
        actual_reheat = np.minimum(
            self.WATER_VALVE_CAPACITY * math.sqrt(pressure_water) * valves * np.maximum(water_available - supply, 0.0),
            conductance * np.maximum(self.MAX_DISCHARGE_K - supply, 0.0),
        )
        self.history.append({"conductance": conductance, "supply": supply,
                             "reheat": actual_reheat, "outside": outside.copy(),
                             "mode": mode, "slow": self.slow_temperature.copy()})
        self.slow_temperature += (temperature - self.slow_temperature) / 48.0
        self.previous_observed = observed.copy()
        self.step_count += 1
        action = {"continuous_action": np.clip(np.nan_to_num(continuous, nan=-1.0), -1.0, 1.0).astype(np.float32),
                  "discrete_action": discrete.astype(np.int32)}
        self.last_action = action
        return action



"""Deterministic thermal-control adaptations, developed against frozen V35.

Attribution: Daniel Wu's public analytical controller supplies the zone model,
observer, SAT/mixing search, fan reset and hydraulic allocation, unchanged in
public_daniel_thermal.py. No neural weights or RL are loaded here.
https://www.kaggle.com/code/danielwuu/cpu-b-conditional-hotwater-20261022
https://www.kaggle.com/code/danielwuu/cpu-a-alpha075-control-20261022
V35's telemetry principle is retained: reject nonphysical data and keep acting.
https://github.com/smart-buildings-2026/starter-kit

The public observer assumes T_next=(T+d+g*(G*T_supply+H))/(1+g*G).
Here G=rho*cp*q [W/K], H [W], g [K/W per decision], d [K/decision].
V41 tests bounded recursive least squares for the six observable heat-balance
coefficients, with exponential forgetting, instead of fixed geometry priors.
P <- (P-K*x.T*P)/lambda; K=P*x/(lambda+x.T*P*x).
Bounds and clipped innovations are engineering hypotheses, not calibrated
confidence intervals. Native five-minute reward, physics and limits decide gains.
V42 tests a modest reserve derived from past prediction error. That error is
not calibrated uncertainty and this reserve is not a comfort guarantee.
"""
import numpy as np


class GuardedJointThermalPolicy(TransferPolicy):
    name = 'facilities_v40_joint_thermal'

    def _zone_temperatures(self, obs):
        value = super()._zone_temperatures(obs)
        previous = self.previous_observed if self.previous_observed is not None else np.mean(self.comfort_band)
        return np.where(np.isfinite(value) & (value >= 150.) & (value <= 400.), value, previous)

    def get_action(self, obs):
        action = super().get_action(obs)
        self.last_diagnostics = dict(model_error_mean_K=float(np.mean(self.model_error)),
            model_residual_mean_K=float(np.mean(self.residual)),
            thermal_response_mean_K_per_kW=float(np.mean(self.theta[:, 5])))
        return action


def bounded_rls(theta, covariance, x, target, lower, upper, forgetting=.997):
    """Independent per-zone least-squares updates; caller supplies past data only."""
    px = np.einsum('nij,nj->ni', covariance, x)
    denominator = forgetting + np.einsum('ni,ni->n', x, px)
    gain = px / np.maximum(denominator[:, None], 1e-9)
    innovation = np.clip(target - np.einsum('ni,ni->n', theta, x), -2., 2.)
    fitted = np.clip(theta + gain * innovation[:, None], lower, upper)
    updated = (covariance - np.einsum('ni,nj->nij', gain, px)) / forgetting
    updated = .5 * (updated + updated.transpose(0, 2, 1))
    return fitted, updated


class IdentifiedJointThermalPolicy(GuardedJointThermalPolicy):
    name = 'facilities_v41_identified_thermal'

    def reset(self):
        super().reset()
        self._thermal_prior = self.theta[:, 5].copy()

    def _learn(self, observed):
        delay = int(self.parameters['observation_delay'])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        power = (applied['conductance'] * (applied['supply'] - observed) + applied['reheat']) / 1000.
        features = self._features(self.previous_observed, applied['outside'], applied['mode'], applied['slow'])
        x = np.column_stack((features, power))
        delta = observed - self.previous_observed
        valid = np.isfinite(x).all(axis=1) & np.isfinite(delta) & (np.abs(delta) <= 3.)
        lower = np.zeros_like(self.theta)
        upper = np.ones_like(self.theta) * np.array([2., 5., 2., 2., 2., 1.])
        lower[:, 0] = -2.
        lower[:, 5] = .5 * self._thermal_prior
        upper[:, 5] = 1.5 * self._thermal_prior
        if np.any(valid):
            fit, covariance = bounded_rls(self.theta[valid], self.covariance[valid], x[valid],
                delta[valid] - self.residual[valid], lower[valid], upper[valid],
                float(self.parameters['forgetting_factor']))
            self.theta[valid], self.covariance[valid] = fit, covariance
        remaining = np.clip(delta - np.einsum('ni,ni->n', self.theta, x), -5., 5.)
        self.model_error = .94 * self.model_error + .06 * np.minimum(np.abs(remaining-self.residual), 2.)
        rate = float(self.parameters['disturbance_rate'])
        self.residual = np.clip((1-rate)*self.residual + rate*remaining, -5., 5.)


class ErrorReserveJointThermalPolicy(GuardedJointThermalPolicy):
    name = 'facilities_v42_error_reserve_thermal'

    def _band(self, obs, temperature):
        low, high, mode = super()._band(obs, temperature)
        if mode:
            extra = np.minimum(np.clip(.5*(self.model_error-.08), 0., .25), .25*np.maximum(high-low, 0.))
            low, high = low+extra, high-extra
        return low, high, mode


NOV03_MODELS = {c.name: c for c in (GuardedJointThermalPolicy, IdentifiedJointThermalPolicy,
                                   ErrorReserveJointThermalPolicy)}


"""Four isolated operating-point variations of frozen V42, authorized Nov 3.

Analytical controller attribution and native model equations are preserved in
public_daniel_thermal.py and facilities_nov03_models.py.
https://www.kaggle.com/code/danielwuu/cpu-a-alpha075-control-20261022
These constants are control hypotheses evaluated with the native reward.
"""


class LeanReserveThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v43_lean_reserve_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, comfort_margin=.08)


class WiderReserveThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v44_wider_reserve_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, comfort_margin=.20)


class GradualRecoveryThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v45_gradual_recovery_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, recovery_steps=2.)


class HeatingCostThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v46_heating_cost_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, gas_cost_ratio=.45)


NOV03_RELEASE = {42: ErrorReserveJointThermalPolicy, 43: LeanReserveThermalPolicy,
                 44: WiderReserveThermalPolicy, 45: GradualRecoveryThermalPolicy,
                 46: HeatingCostThermalPolicy}

class IdentifiedReservePolicy(IdentifiedJointThermalPolicy, ErrorReserveJointThermalPolicy):
    name = "identified_reserve"

import math
import numpy as np
class CausalIdentifiedReservePolicy(IdentifiedReservePolicy):
    """Keep the supplied bounded fit; measure actual forecast error first.

    Computing reserve from the already-fitted equation residual can conceal
    prediction error. Use the old coefficients to predict the newly received
    temperature, then fit them and update the reserve from that prior forecast.
    All inputs are observations and actions already available in this episode.
    """

    name = "causal_identified"

    def _learn(self, observed):
        delay = int(self.parameters["observation_delay"])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        old = self.previous_observed
        features = self._features(old, applied["outside"], applied["mode"], applied["slow"])
        drift = np.einsum("ij,ij->i", self.theta[:, :5], features) + self.residual
        gain = self.theta[:, 5] / 1000.0
        prediction = (old + drift + gain * (applied["conductance"] * applied["supply"] + applied["reheat"])) / (
            1.0 + gain * applied["conductance"])
        error = np.where(np.isfinite(observed - prediction), np.minimum(np.abs(observed - prediction), 2.0), 2.0)
        previous_error = self.model_error.copy()
        super()._learn(observed)
        self.model_error = 0.94 * previous_error + 0.06 * error

class WarmStandbyPolicy(CausalIdentifiedReservePolicy):
    """Keep the existing hot setpoint for 36 hours after measured heat demand.

    The public controller drops the requested water temperature after two
    hours without demand. Reheating that reservoir the following morning can
    dominate short cold-period losses. A longer hold costs tank dissipation
    but avoids daily forced cooling and repeated recovery of stored heat.
    Demand timing is episode-local; no future weather or station ID is read.
    """

    name = "causal_warm_standby"
    HOLD_STEPS = 36 * 12

    def _native_action(self, kind, value, low, high):
        if (kind == "supply_water_temperature_setpoint"
                and self.step_count - self.last_heating_step < self.HOLD_STEPS):
            value = self.parameters["hot_water_target"]
        return super()._native_action(kind, value, low, high)

class WeekendWaterHoldPolicy(WarmStandbyPolicy):
    """Carry the warm reservoir through a longer pause after heat demand."""

    name = "warm_hold72"
    HOLD_STEPS = 72 * 12

class DeepFastRecoveryPolicy(WeekendWaterHoldPolicy):
    """Use a 274 K empty-period floor and one-step requested band recovery."""

    name = "hold72_deep_fast_recovery"
    DEFAULTS = dict(WeekendWaterHoldPolicy.DEFAULTS, standby_minimum=274.0, recovery_steps=1.0)

class EmptyDayConsistentPolicy(DeepFastRecoveryPolicy):
    """Use the empty-day band but keep the real schedule mode in the model."""

    name = "deep_empty_day_consistent_model"
    EMPTY_CONFIRMATION_HOURS = 2.0

    def reset(self):
        super().reset()
        self.empty_day_relaxed_steps = 0

    def _band(self, obs, temperature):
        low, high, schedule_mode = super()._band(obs, temperature)
        if not schedule_mode or self.occupancy_index is None:
            return low, high, schedule_mode
        occupants = float(obs[self.occupancy_index])
        elapsed = (self._hour(obs) - self.start_hour) % 24.0
        scheduled_duration = (self.end_hour - self.start_hour) % 24.0
        if (not np.isfinite(occupants) or occupants != 0.0
                or not self.EMPTY_CONFIRMATION_HOURS <= elapsed < scheduled_duration):
            return low, high, schedule_mode
        self.empty_day_relaxed_steps += 1
        # Empty low/high limits control the AHU allocation independently of
        # this bit. Schedule-driven internal heat is still physically active,
        # so keep its real mode for free drift and history['mode']. Historical
        # learning/observer calls then naturally use their own stored modes.
        return (np.full(self.n_zones, self.parameters["standby_minimum"]),
                np.full(self.n_zones, self.parameters["standby_maximum"]), schedule_mode)

    def experiment_diagnostics(self):
        return {"empty_day_relaxed_steps": self.empty_day_relaxed_steps}

class FineSupplySearchPolicy(WarmStandbyPolicy):
    """Enrich the supplied SAT search from coarse points to a 0.5 K grid."""

    name = "warm_fine_sat"

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        # The physical equations, limits and cost weights below match V42.
        # The 0.5 K grid includes every original fixed setpoint candidate.
        t, d, lo, hi = temperature[group], free[group], low[group], high[group]
        response = self.theta[group, 5]
        capacity = max_flow[group]
        inside = float(np.average(t, weights=self.zone_area[group]))
        targets = np.arange(285.0, 305.01, 0.5)
        targets = np.concatenate((targets, [float(np.clip(np.min(hi) - 1.5, 285.0, 305.0))]))
        setpoint = np.repeat(targets, 2)
        recirc = np.full_like(setpoint, self.MAX_RECIRCULATION)
        if outside < inside - 0.5:
            recirc[1::2] = np.clip((targets - outside) / max(inside - outside, 1e-6), 0.0, self.MAX_RECIRCULATION)
        mixed = recirc * inside + (1.0 - recirc) * outside
        supply = np.minimum(mixed, setpoint)
        sa = supply[:, None]
        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / response
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / response
        cooling_flow = cool_need[None, :] / (self.AIR_CAPACITY * np.maximum(hi[None, :] - sa, 0.05))
        heating_flow = heat_need[None, :] / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - lo[None, :], 0.5))
        maximum_water_heat = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * np.maximum(water - sa, 0.0)
        maximum_heating_flow = maximum_water_heat / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - sa, 0.1))
        heating_flow = np.minimum(heating_flow, maximum_heating_flow)
        needed_flow = np.maximum(cooling_flow, heating_flow)
        fraction = np.max(needed_flow / capacity[None, :], axis=1)
        pressure = np.clip(self.DESIGN_PRESSURE * fraction ** 2, self.MIN_PRESSURE, self.DESIGN_PRESSURE)
        pressure_ratio = np.sqrt(pressure / self.DESIGN_PRESSURE)
        dampers = np.clip(needed_flow / (capacity[None, :] * pressure_ratio[:, None]), self.MIN_DAMPER, 1.0)
        flow = dampers * capacity[None, :] * pressure_ratio[:, None]
        conductance = flow * self.AIR_CAPACITY
        water_gap = np.maximum(water - sa, 0.0)
        wanted_reheat = np.maximum((lo[None, :] - t[None, :] - d[None, :]) * 1000.0 / response[None, :]
                                    + conductance * (lo[None, :] - sa), 0.0)
        reheat_limit = np.minimum(conductance * np.maximum(self.MAX_DISCHARGE_K - sa, 0.0),
                                  self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * water_gap)
        reheat = np.minimum(wanted_reheat, reheat_limit)
        predicted = (t[None, :] + d[None, :] + response[None, :] * (conductance * sa + reheat) / 1000.0) / (
            1.0 + response[None, :] * conductance / 1000.0)
        violation = np.maximum(lo[None, :] - predicted, 0.0) + np.maximum(predicted - hi[None, :], 0.0)
        thermal_penalty = np.sum(violation + 2.0 * violation ** 2, axis=1)
        total_flow = flow.sum(axis=1)
        fan = total_flow * pressure * (2.0 - recirc) / 0.9
        chiller = total_flow * self.AIR_CAPACITY * np.maximum(mixed - supply, 0.0) / 3.5
        gas = reheat.sum(axis=1)
        objective = fan + chiller + self.parameters["gas_cost_ratio"] * gas
        objective += self.parameters["comfort_cost"] * thermal_penalty
        off_prediction = t + d
        off_violation = np.maximum(lo - off_prediction, 0.0) + np.maximum(off_prediction - hi, 0.0)
        off_objective = self.parameters["comfort_cost"] * np.sum(off_violation + 2.0 * off_violation ** 2)
        best = int(np.argmin(objective))
        if off_objective <= objective[best]:
            return {"run": 0, "pressure": 0.0, "setpoint": 305.0, "recirc": 0.7,
                    "damper": np.full(group.size, self.MIN_DAMPER), "conductance": np.zeros(group.size),
                    "supply": np.full(group.size, mixed[best]), "reheat": np.zeros(group.size)}
        return {"run": 1, "pressure": float(pressure[best]), "setpoint": float(setpoint[best]),
                "recirc": float(recirc[best]), "damper": dampers[best], "conductance": conductance[best],
                "supply": np.full(group.size, supply[best]), "reheat": reheat[best]}

class EmptyDayConsistentFineLeanPolicy(EmptyDayConsistentPolicy):
    """Combine consistent empty-day features with fine SAT and a lean margin."""

    name = "deep_empty_consistent_fine_margin08"
    DEFAULTS = dict(DeepFastRecoveryPolicy.DEFAULTS, comfort_margin=.08)

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        return FineSupplySearchPolicy._choose_ahu(
            self, group, temperature, free, low, high, max_flow, outside, water)

class StableMarginPolicy(EmptyDayConsistentFineLeanPolicy):
    """Halve fixed padding only for zones with small causal forecast error."""

    name = "deep_empty_stable_margin04"
    ERROR_THRESHOLD_K = .01
    FIXED_MARGIN_FRACTION = .5

    def reset(self):
        super().reset()
        self.adaptive_margin_band_calls = 0
        self.adaptive_margin_zone_decisions = 0

    def _eligible_margin_zones(self):
        error = self.model_error
        return np.isfinite(error) & (error >= 0.0) & (error <= self.ERROR_THRESHOLD_K)

    def _band(self, obs, temperature):
        previous_empty_count = self.empty_day_relaxed_steps
        low, high, physical_mode = super()._band(obs, temperature)
        # EmptyDayConsistentPolicy deliberately replaces both bands with
        # exact standby limits. A changed counter identifies this call, even
        # if earlier observations already triggered empty-day relaxation.
        if self.empty_day_relaxed_steps != previous_empty_count:
            return low, high, physical_mode
        soon = (bool(obs[self.comfort_soon_index] > .5)
                if self.comfort_soon_index is not None else bool(physical_mode))
        if not (physical_mode or soon):
            return low, high, physical_mode
        reduction = max(0.0, float(self.parameters["comfort_margin"])) * (1.0 - self.FIXED_MARGIN_FRACTION)
        eligible = self._eligible_margin_zones()
        if not reduction or not np.any(eligible):
            return low, high, physical_mode
        # Eligibility is below the inherited .08 K error-reserve threshold,
        # so the reserve is exactly zero for these zones. Keep all other
        # bands untouched. Undo/reapply only the inherited recovery transform
        # to remove fixed padding at its original point before recovery; this
        # also preserves prestart semantics for configured recovery >1.
        recovery = max(1.0, float(self.parameters["recovery_steps"]))
        t = temperature[eligible]
        old_low, old_high = low[eligible], high[eligible]
        target_low = np.where(old_low > t, t + recovery * (old_low - t), old_low) - reduction
        target_high = np.where(old_high < t, t + recovery * (old_high - t), old_high) + reduction
        low = low.copy()
        high = high.copy()
        low[eligible] = np.minimum(target_low, t + np.maximum((target_low - t) / recovery, 0.0))
        high[eligible] = np.maximum(target_high, t - np.maximum((t - target_high) / recovery, 0.0))
        self.adaptive_margin_band_calls += 1
        self.adaptive_margin_zone_decisions += int(np.count_nonzero(eligible))
        return low, high, physical_mode

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "adaptive_margin_band_calls": self.adaptive_margin_band_calls,
                "adaptive_margin_zone_decisions": self.adaptive_margin_zone_decisions}

class VacancyBoundaryTolerancePolicy(EmptyDayConsistentFineLeanPolicy):
    """Correct only roundoff at a vacancy boundary, leaving prestart intact."""

    CLOCK_TOLERANCE_HOURS = 1e-6  # 3.6 ms; far below a five-minute decision.

    def _band(self, obs, temperature):
        before = self.empty_day_relaxed_steps
        low, high, mode = super()._band(obs, temperature)
        if (not mode or self.empty_day_relaxed_steps != before
                or self.occupancy_index is None):
            return low, high, mode
        occupants = float(obs[self.occupancy_index])
        if not np.isfinite(occupants) or occupants != 0.0:
            return low, high, mode
        raw_elapsed = self._hour(obs) - self.start_hour
        elapsed = 0.0 if abs(raw_elapsed) <= self.CLOCK_TOLERANCE_HOURS else raw_elapsed % 24.0
        duration = (self.end_hour - self.start_hour) % 24.0
        if (abs(elapsed - self.EMPTY_CONFIRMATION_HOURS) <= self.CLOCK_TOLERANCE_HOURS
                and elapsed < duration):
            self.empty_day_relaxed_steps += 1
            return (np.full(self.n_zones, self.parameters["standby_minimum"]),
                    np.full(self.n_zones, self.parameters["standby_maximum"]), mode)
        return low, high, mode

class StableHalfHourVacancyPolicy(StableMarginPolicy, VacancyBoundaryTolerancePolicy):
    """Compose stable half-padding with thirty-minute current-zero timing."""

    name = "deep_empty_confirm30m_stable_margin04"
    EMPTY_CONFIRMATION_HOURS = .5

class DemandSized340Policy(StableHalfHourVacancyPolicy):
    """Revisit the 340 K demand-sized reserve around the new best policy."""

    name = "best_water_demand340"
    MIN_HEATING_TARGET_K = 340.0
    COIL_CAPACITY_HEADROOM = .95
    TARGET_HEADROOM_K = 2.0

    def reset(self):
        super().reset()
        self.held_water_target = None
        self.observed_water_for_reserve = 340.0
        self.required_water_target = 310.0
        self.initial_water_temperature = None
        self.maximum_water_target = 310.0
        self.maximum_required_water_target = 310.0
        self.capacity_limited_target_requests = 0
        self.demand_sized_water_steps = 0

    def get_action(self, obs):
        self.observed_water_for_reserve = self._temperature(
            np.asarray(obs), self.water_sensor, "supply_water_temperature_sensor", 340.0)
        self.required_water_target = 310.0
        return super().get_action(obs)

    def _required_target(self, group, temperature, free, low, max_flow, choice):
        if not choice["run"]:
            return 310.0
        supply = choice["supply"]
        heating_need = np.maximum(low[group] - temperature[group] - free[group], 0.0) * 1000.0 / self.theta[group, 5]
        # Recover useful airflow before the inherited search clips it to the
        # current water capacity. Inspecting only selected reheat would never
        # request sufficient recovery from a cold, water-limited reservoir.
        desired_flow = heating_need / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - low[group], .5))
        desired_conductance = np.maximum(
            choice["conductance"], self.AIR_CAPACITY * np.minimum(desired_flow, max_flow[group]))
        wanted_reheat = np.maximum((low[group] - temperature[group] - free[group]) * 1000.0 / self.theta[group, 5] + desired_conductance * (low[group] - supply), 0.0)
        air_limit = desired_conductance * np.maximum(self.MAX_DISCHARGE_K - supply, 0.0)
        feasible_reheat = np.minimum(wanted_reheat, air_limit)
        active = feasible_reheat > 10.0
        if not np.any(active):
            return 310.0
        maximum_water_conductance = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * self.COIL_CAPACITY_HEADROOM
        targets = supply + feasible_reheat / maximum_water_conductance + self.TARGET_HEADROOM_K
        uncapped_target = float(np.max(targets[active]))
        self.maximum_required_water_target = max(self.maximum_required_water_target, uncapped_target)
        if uncapped_target > 350.0:
            self.capacity_limited_target_requests += 1
        return float(np.clip(uncapped_target, 310.0, 350.0))

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        choice = super()._choose_ahu(group, temperature, free, low, high, max_flow, outside, water)
        self.required_water_target = max(
            self.required_water_target,
            self._required_target(group, temperature, free, low, max_flow, choice))
        return choice

    def _native_action(self, kind, value, low, high):
        if kind != "supply_water_temperature_setpoint":
            return super()._native_action(kind, value, low, high)
        if self.step_count - self.last_heating_step < self.HOLD_STEPS:
            if self.held_water_target is None:
                self.initial_water_temperature = self.observed_water_for_reserve
                self.held_water_target = float(np.clip(
                    max(self.MIN_HEATING_TARGET_K, self.observed_water_for_reserve), 310.0, 350.0))
            self.held_water_target = min(350.0, max(self.held_water_target, self.required_water_target))
            value = self.held_water_target
            self.demand_sized_water_steps += 1
        else:
            value = 310.0
            self.held_water_target = None
        self.maximum_water_target = max(self.maximum_water_target, float(value))
        # Encode directly with the inherited normalizer law. Calling the
        # ancestor water-hold override here would replace the sized target
        # with 345 K. Non-water actuators still use the exact parent path.
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((v for k, v in self.action_norm.items() if str(k).endswith(kind)), None)
        if spec:
            low = float(spec.get("min_native_value", low))
            high = float(spec.get("max_native_value", high))
            out_low = float(spec.get("min_normalized_value", -1.0))
            out_high = float(spec.get("max_normalized_value", 1.0))
        else:
            out_low, out_high = -1.0, 1.0
        return np.clip(out_low + (np.asarray(value) - low) * (out_high - out_low) / max(high - low, 1e-12), out_low, out_high)

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "initial_reserve_water_k": self.initial_water_temperature,
                "maximum_water_target_k": self.maximum_water_target,
                "maximum_required_water_target_k": self.maximum_required_water_target,
                "capacity_limited_target_requests": self.capacity_limited_target_requests,
                "demand_sized_water_steps": self.demand_sized_water_steps,
                "held_water_target_k": self.held_water_target}

class SignedDemandSized320Policy(DemandSized340Policy):
    """Screen signed feasible duty with a 320 K qualified reservoir floor."""

    name = "best_water_signed320"
    MIN_HEATING_TARGET_K = 320.0

class NearMinimumPumpPressurePolicy(SignedDemandSized320Policy):
    """Retain 0.1 percent pressure reserve above the feasible valve minimum."""

    name = "best_water_pump_safety001"
    PRESSURE_SAFETY_FACTOR = 1.001
    VALVE_KV_M3_PER_HOUR = .108
    FLOW_PRODUCT_TOLERANCE = 8.0 * np.finfo(np.float32).eps

    def reset(self):
        super().reset()
        self.hydraulic_adjusted_steps = 0
        self.hydraulic_capacity_tie_steps = 0
        self.hydraulic_quantization_fallback_steps = 0
        self.hydraulic_power_wsteps_before = 0.0
        self.hydraulic_power_wsteps_after = 0.0
        self.hydraulic_maximum_flow_product_error = 0.0
        self.hydraulic_maximum_history_heat_correction_w = 0.0

    def _normalization_limits(self, kind, native_low, native_high):
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((value for key, value in self.action_norm.items()
                         if str(key).endswith(kind)), None)
        if spec:
            return (float(spec.get("min_native_value", native_low)),
                    float(spec.get("max_native_value", native_high)),
                    float(spec.get("min_normalized_value", -1.0)),
                    float(spec.get("max_normalized_value", 1.0)))
        return float(native_low), float(native_high), -1.0, 1.0

    @staticmethod
    def _decode_action(value, limits):
        native_low, native_high, out_low, out_high = limits
        return native_low + (np.asarray(value, dtype=float) - out_low) * (
            native_high - native_low) / (out_high - out_low)

    def _encode_action_upward(self, kind, desired, limits, default_low, default_high):
        # Float32 action rounding must not silently remove the planned flow.
        encoded = np.asarray(self._native_action(kind, desired, default_low, default_high), dtype=np.float32)
        decoded = self._decode_action(encoded, limits)
        encoded = np.where(decoded < desired,
                           np.nextafter(encoded, np.float32(np.inf)), encoded).astype(np.float32)
        return np.clip(encoded, -1.0, 1.0).astype(np.float32)

    def get_action(self, obs):
        action = super().get_action(obs)
        if (len(self.pressure_indices) <= self.num_ahus
                or not action["discrete_action"][-1]):
            return action
        pressure_kind = "differential_pressure_setpoint"
        valve_kind = "reheat_valve_percentage_command"
        pressure_limits = self._normalization_limits(pressure_kind, 0.0, 20.0)
        valve_limits = self._normalization_limits(valve_kind, 0.0, 1.0)
        if (not np.isfinite(pressure_limits + valve_limits).all()
                or pressure_limits[1] <= pressure_limits[0]
                or pressure_limits[3] <= pressure_limits[2]
                or valve_limits[1] <= valve_limits[0]
                or valve_limits[3] <= valve_limits[2]):
            self.hydraulic_quantization_fallback_steps += 1
            return action
        continuous = action["continuous_action"]
        pressure_index = self.pressure_indices[-1]
        old_pressure = float(self._decode_action(continuous[pressure_index], pressure_limits))
        old_valves = np.clip(self._decode_action(continuous[self.valve_indices], valve_limits), 0.0, 1.0)
        if not np.isfinite(old_pressure) or old_pressure <= 0.0 or not np.isfinite(old_valves).all():
            return action
        # Respect the actual metadata valve cap, not an assumed full opening.
        valve_cap = min(1.0, valve_limits[1])
        if valve_cap <= 0.0:
            return action
        flow_product = old_valves * math.sqrt(old_pressure)
        hydraulic_scale = 100000.0 * self.VALVE_KV_M3_PER_HOUR / 3600.0
        old_power = hydraulic_scale * old_pressure * float(np.sum(flow_product))
        self.hydraulic_power_wsteps_before += old_power
        minimum_pressure = (float(np.max(flow_product)) / valve_cap) ** 2
        target_pressure = min(old_pressure, max(0.0, pressure_limits[0],
                                               minimum_pressure * self.PRESSURE_SAFETY_FACTOR))
        if target_pressure <= 0.0 or target_pressure >= old_pressure:
            self.hydraulic_capacity_tie_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        encoded_pressure = self._encode_action_upward(
            pressure_kind, target_pressure, pressure_limits, 0.0, 20.0)
        new_pressure = float(self._decode_action(encoded_pressure, pressure_limits))
        if not (0.0 < new_pressure < old_pressure and new_pressure <= 20.0):
            self.hydraulic_quantization_fallback_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        desired_valves = flow_product / math.sqrt(new_pressure)
        encoded_valves = self._encode_action_upward(
            valve_kind, desired_valves, valve_limits, 0.0, 1.0)
        new_valves = np.clip(self._decode_action(encoded_valves, valve_limits), 0.0, 1.0)
        new_product = new_valves * math.sqrt(new_pressure)
        difference = new_product - flow_product
        tolerance = self.FLOW_PRODUCT_TOLERANCE * np.maximum(1.0, flow_product)
        if (np.any(desired_valves > valve_cap)
                or np.any(new_valves > valve_cap + self.FLOW_PRODUCT_TOLERANCE)
                or np.any(np.abs(difference) > tolerance)
                or np.any(difference < -tolerance)):
            self.hydraulic_quantization_fallback_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        new_power = hydraulic_scale * new_pressure * float(np.sum(new_product))
        if new_power >= old_power:
            self.hydraulic_quantization_fallback_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        continuous[pressure_index] = encoded_pressure
        continuous[self.valve_indices] = encoded_valves
        # History must describe the new normalized/quantized commands. This
        # also accounts for metadata caps that clipped the parent's planned
        # pressure or valves. Every non-reheat history field stays unchanged.
        history = self.history[-1]
        delivered_reheat = np.minimum(
            self.WATER_VALVE_CAPACITY * new_product
            * np.maximum(self.observed_water_for_reserve - history["supply"], 0.0),
            history["conductance"] * np.maximum(self.MAX_DISCHARGE_K - history["supply"], 0.0))
        self.hydraulic_maximum_history_heat_correction_w = max(
            self.hydraulic_maximum_history_heat_correction_w,
            float(np.max(np.abs(delivered_reheat - history["reheat"]))))
        history["reheat"] = delivered_reheat
        self.last_action = action
        self.hydraulic_adjusted_steps += 1
        self.hydraulic_power_wsteps_after += new_power
        self.hydraulic_maximum_flow_product_error = max(
            self.hydraulic_maximum_flow_product_error, float(np.max(np.abs(difference))))
        return action

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "hydraulic_adjusted_steps": self.hydraulic_adjusted_steps,
                "hydraulic_capacity_tie_steps": self.hydraulic_capacity_tie_steps,
                "hydraulic_quantization_fallback_steps": self.hydraulic_quantization_fallback_steps,
                "hydraulic_power_wsteps_before": self.hydraulic_power_wsteps_before,
                "hydraulic_power_wsteps_after": self.hydraulic_power_wsteps_after,
                "hydraulic_maximum_flow_product_error": self.hydraulic_maximum_flow_product_error,
                "hydraulic_maximum_history_heat_correction_w": self.hydraulic_maximum_history_heat_correction_w}


