from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from eco_routing.data.loaders import data_repository


@dataclass
class BehaviorProfiles:
    vehicle_efficiency: Dict[str, float]
    driver_aggressiveness: Dict[str, float]
    congestion_by_road_type: Dict[str, float]
    idle_penalty_by_road_type: Dict[str, float]
    weather_risk_index: float


def build_behavior_profiles() -> BehaviorProfiles:
    vehicles = data_repository.vehicles
    drivers = data_repository.drivers
    env = data_repository.environment
    trips = data_repository.trips
    insights = data_repository.insights

    vehicle_eff = {}
    if not vehicles.empty and "vehicle_id" in vehicles.columns:
        if "fuel_efficiency_l_per_100km" in vehicles.columns:
            eff_series = vehicles.set_index("vehicle_id")["fuel_efficiency_l_per_100km"]
            vehicle_eff = eff_series.to_dict()

    driver_aggr = {}
    if not drivers.empty and "driver_id" in drivers.columns:
        col = None
        for c in drivers.columns:
            if "aggress" in c.lower():
                col = c
                break
        if col:
            driver_aggr = drivers.set_index("driver_id")[col].to_dict()

    congestion_by_road_type: Dict[str, float] = {}
    idle_penalty_by_road_type: Dict[str, float] = {}
    if not trips.empty and "road_type" in trips.columns:
        grouped = trips.groupby("road_type")
        if "avg_congestion" in trips.columns:
            congestion_by_road_type = grouped["avg_congestion"].mean().to_dict()
        if "idle_ratio" in trips.columns:
            idle_penalty_by_road_type = grouped["idle_ratio"].mean().to_dict()

    weather_risk_index = 1.0
    if not env.empty:
        cols = [c for c in env.columns if "rain" in c.lower() or "visibility" in c.lower()]
        if cols:
            weather_risk_index = float(env[cols].mean().mean())

    return BehaviorProfiles(
        vehicle_efficiency=vehicle_eff,
        driver_aggressiveness=driver_aggr,
        congestion_by_road_type=congestion_by_road_type,
        idle_penalty_by_road_type=idle_penalty_by_road_type,
        weather_risk_index=weather_risk_index,
    )


behavior_profiles: Optional[BehaviorProfiles] = None


def get_behavior_profiles() -> BehaviorProfiles:
    global behavior_profiles
    if behavior_profiles is None:
        behavior_profiles = build_behavior_profiles()
    return behavior_profiles


