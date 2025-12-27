from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from eco_routing.core.fuel_emission_agent import SegmentMetrics
from eco_routing.core.pathfinder import PathResult
from eco_routing.data.loaders import data_repository

RouteLabel = Literal["fastest", "greenest", "balanced"]


@dataclass
class RouteMetrics:
    path: PathResult
    segments: List[SegmentMetrics]
    total_fuel_l: float
    total_co2_kg: float
    eco_score: float
    label: RouteLabel


class EcoScoreAgent:
    """Aggregates route metrics and computes normalized EcoScore (0–100)."""

    def __init__(self) -> None:
        self._init_normalizers()

    def _init_normalizers(self) -> None:
        fuel_df = data_repository.fuel_predictions
        co2_df = data_repository.co2_aggregates
        sims = data_repository.simulations

        self.mean_fuel = float(fuel_df["fuel_liters"].mean()) if "fuel_liters" in fuel_df.columns else 5.0
        self.mean_co2 = float(co2_df["co2_kg"].mean()) if "co2_kg" in co2_df.columns else 12.0
        self.mean_time_min = float(sims["time_min"].mean()) if "time_min" in sims.columns else 30.0

    def _score_single(self, fuel_l: float, co2_kg: float, time_min: float) -> float:
        fuel_component = max(0.0, 1.5 - fuel_l / (self.mean_fuel or 1.0))
        co2_component = max(0.0, 1.5 - co2_kg / (self.mean_co2 or 1.0))
        time_component = max(0.0, 1.0 - time_min / (self.mean_time_min or 1.0))
        raw = 0.4 * fuel_component + 0.4 * co2_component + 0.2 * time_component
        return max(0.0, min(100.0, raw * 100.0 / 3.0))

    def rank_routes(
        self,
        paths: List[PathResult],
        segments_per_path: List[List[SegmentMetrics]],
        fuels: List[float],
        co2s: List[float],
    ) -> List[RouteMetrics]:
        metrics: List[RouteMetrics] = []
        for i, path in enumerate(paths):
            fuel_l = fuels[i]
            co2_kg = co2s[i]
            eco_score = self._score_single(fuel_l, co2_kg, path.total_time_min)
            metrics.append(
                RouteMetrics(
                    path=path,
                    segments=segments_per_path[i],
                    total_fuel_l=fuel_l,
                    total_co2_kg=co2_kg,
                    eco_score=eco_score,
                    label="balanced",
                )
            )

        if not metrics:
            return metrics

        fastest = min(metrics, key=lambda m: m.path.total_time_min)
        greenest = min(metrics, key=lambda m: (m.total_co2_kg, m.total_fuel_l))
        best_eco = max(metrics, key=lambda m: m.eco_score)

        for m in metrics:
            if m is fastest:
                m.label = "fastest"
            if m is greenest:
                m.label = "greenest"
            if m is best_eco:
                m.label = "balanced"

        return metrics


eco_score_agent = EcoScoreAgent()


