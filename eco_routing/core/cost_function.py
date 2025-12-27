from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict
import logging

from eco_routing.core.road_graph import RoadSegment
from eco_routing.data.preprocess import get_behavior_profiles
from eco_routing.ml.fuel_predictor import predict_fuel
from eco_routing.ml.traffic_predictor import predict_traffic
from eco_routing.ml.driver_profiler import infer_driver_profile

Preference = Literal["green", "fast", "balanced"]


@dataclass
class SegmentCostContext:
    vehicle_id: str
    driver_id: str
    preference: Preference


class SegmentCostCalculator:
    """
    Computes sustainability-aware cost per road segment.
    """

    def __init__(self) -> None:
        self.behavior = get_behavior_profiles()

    def _base_weights(self, preference: Preference) -> tuple[float, float, float, float, float]:
        if preference == "green":
            return 0.4, 0.3, 0.15, 0.1, 0.05
        if preference == "fast":
            return 0.15, 0.1, 0.1, 0.35, 0.3
        return 0.25, 0.2, 0.15, 0.25, 0.15

    def _vehicle_factor(self, vehicle_id: str) -> float:
        eff = self.behavior.vehicle_efficiency.get(vehicle_id)
        if eff is None or eff <= 0:
            return 1.0
        return 6.5 / float(eff)

    def _driver_factor(self, driver_id: str) -> float:
        # Try to use ML to infer driver profile and adjust factor accordingly
        try:
            driver_profile = infer_driver_profile({'driver_id': driver_id})
            if driver_profile == 'eco':
                # Eco drivers are more efficient, so return a factor < 1.0
                return 0.8
            elif driver_profile == 'aggressive':
                # Aggressive drivers are less efficient, so return a factor > 1.0
                return 1.3
            else:
                # Normal driver profile
                return 1.0
        except:
            # Fallback to the original behavior
            aggr = self.behavior.driver_aggressiveness.get(driver_id)
            if aggr is None:
                return 1.0
            return 1.0 + 0.3 * float(aggr)

    def _congestion_factor(self, road_type: str) -> float:
        # Try to use ML to predict traffic conditions
        try:
            traffic_features = {
                'road_type': road_type,
                'hour_of_day': 12,  # Default to midday if not available
                'day_of_week': 1,   # Default to Monday if not available
                'avg_speed': 40.0   # Default average speed
            }
            traffic_factor = predict_traffic(traffic_features)
            # Scale the base congestion factor by the ML-predicted traffic factor
            base_factor = float(self.behavior.congestion_by_road_type.get(road_type, 1.0))
            return base_factor * traffic_factor
        except:
            # Fallback to original behavior
            return float(self.behavior.congestion_by_road_type.get(road_type, 1.0))

    def _idle_factor(self, road_type: str) -> float:
        return float(self.behavior.idle_penalty_by_road_type.get(road_type, 0.1))

    def fuel_and_co2_estimate(self, segment: RoadSegment, ctx: SegmentCostContext) -> tuple[float, float]:
        distance_km = segment.distance_km
        base_l_per_100km = 7.0
        vehicle_factor = self._vehicle_factor(ctx.vehicle_id)
        driver_factor = self._driver_factor(ctx.driver_id)
        congestion_factor = self._congestion_factor(segment.road_type)
        weather_factor = self.behavior.weather_risk_index

        # Try to use ML prediction for fuel consumption
        try:
            features_dict = {
                'distance': distance_km,
                'speed': segment.speed_kph if hasattr(segment, 'speed_kph') and segment.speed_kph > 0 else 60.0,
                'elevation_change': segment.elevation_delta if hasattr(segment, 'elevation_delta') else 0.0,
                'road_type_factor': self._congestion_factor(segment.road_type),
                'vehicle_factor': vehicle_factor,
                'driver_factor': driver_factor,
                'weather_factor': weather_factor
            }
            fuel_l = predict_fuel(features_dict)
            logging.info("Fuel prediction: ML model used")
        except Exception as e:
            # Fallback to the existing formula
            l_per_100km = base_l_per_100km * vehicle_factor * driver_factor
            l_per_100km *= (0.7 + 0.6 * congestion_factor)
            l_per_100km *= (0.8 + 0.4 * weather_factor)

            fuel_l = l_per_100km * distance_km / 100.0
            logging.info("Fuel prediction: Fallback formula used")
            
        co2_kg = fuel_l * 2.31
        return fuel_l, co2_kg

    def segment_cost(self, segment: RoadSegment, ctx: SegmentCostContext) -> float:
        fuel_l, co2_kg = self.fuel_and_co2_estimate(segment, ctx)
        w1, w2, w3, w4, w5 = self._base_weights(ctx.preference)

        elevation_penalty = max(0.0, segment.elevation_delta) * segment.distance_km
        congestion_penalty = self._congestion_factor(segment.road_type) * segment.distance_km
        idle_penalty = self._idle_factor(segment.road_type) * segment.distance_km

        cost = (
            w1 * fuel_l
            + w2 * co2_kg
            + w3 * elevation_penalty
            + w4 * congestion_penalty
            + w5 * idle_penalty
        )
        return float(cost)

    def compute_context_factors(self, ctx: SegmentCostContext) -> Tuple[float, float, float, float, float, float, float, float]:
        w1, w2, w3, w4, w5 = self._base_weights(ctx.preference)
        v_f = self._vehicle_factor(ctx.vehicle_id)
        d_f = self._driver_factor(ctx.driver_id)
        w_f = self.behavior.weather_risk_index
        return w1, w2, w3, w4, w5, v_f, d_f, w_f

    def segment_cost_fast(
        self,
        segment: RoadSegment,
        factors: Tuple[float, float, float, float, float, float, float, float],
        congestion_map: Dict[str, float],
        idle_map: Dict[str, float],
    ) -> float:
        w1, w2, w3, w4, w5, v_f, d_f, w_f = factors
        dist = segment.distance_km
        rtype = segment.road_type
        
        # Use ML-enhanced congestion factor
        try:
            traffic_features = {
                'road_type': rtype,
                'hour_of_day': 12,  # Default to midday if not available
                'day_of_week': 1,   # Default to Monday if not available
                'avg_speed': 40.0   # Default average speed
            }
            traffic_factor = predict_traffic(traffic_features)
            cong_f = congestion_map.get(rtype, 1.0) * traffic_factor
        except:
            cong_f = congestion_map.get(rtype, 1.0)
        
        # Combined fuel logic: 7.0 * v_f * d_f * (0.7 + 0.6*c) * (0.8 + 0.4*w) * dist / 100.0
        # Re-arrange for fewer ops if possible, but linearity is fine.
        l_per_100km = 7.0 * v_f * d_f * (0.7 + 0.6 * cong_f) * (0.8 + 0.4 * w_f)
        fuel_l = l_per_100km * dist * 0.01
        co2_kg = fuel_l * 2.31
        
        elev_p = max(0.0, segment.elevation_delta) * dist
        cong_p = cong_f * dist
        idle_p = idle_map.get(rtype, 0.1) * dist
        
        return (
            w1 * fuel_l +
            w2 * co2_kg +
            w3 * elev_p +
            w4 * cong_p +
            w5 * idle_p
        )


segment_cost_calculator = SegmentCostCalculator()