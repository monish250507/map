from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from eco_routing.core.cost_function import SegmentCostContext, segment_cost_calculator
from eco_routing.core.pathfinder import PathResult
from eco_routing.core.road_graph import road_graph


@dataclass
class SegmentMetrics:
    u: int
    v: int
    key: int
    distance_km: float
    time_min: float
    fuel_liters: float
    co2_kg: float


class FuelEmissionAgent:
    """Computes fuel and CO₂ emissions per segment."""

    def compute_for_path(
        self,
        path: PathResult,
        ctx: SegmentCostContext,
    ) -> Tuple[List[SegmentMetrics], float, float]:
        segments: List[SegmentMetrics] = []
        total_fuel = 0.0
        total_co2 = 0.0
        for u, v, key in path.edges:
            seg = road_graph.get_segment(u, v, key)
            fuel_l, co2_kg = segment_cost_calculator.fuel_and_co2_estimate(seg, ctx)
            time_min = 0.0
            if seg.speed_kph > 0:
                time_min = seg.distance_km / seg.speed_kph * 60.0

            metrics = SegmentMetrics(
                u=u,
                v=v,
                key=key,
                distance_km=seg.distance_km,
                time_min=time_min,
                fuel_liters=fuel_l,
                co2_kg=co2_kg,
            )
            segments.append(metrics)
            total_fuel += fuel_l
            total_co2 += co2_kg
        return segments, total_fuel, total_co2


