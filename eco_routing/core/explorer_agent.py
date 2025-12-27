from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from eco_routing.core.cost_function import Preference, SegmentCostContext
from eco_routing.core.eco_score import RouteMetrics, eco_score_agent
from eco_routing.core.explainability_agent import RouteExplanation, explainability_agent
from eco_routing.core.fuel_emission_agent import FuelEmissionAgent
from eco_routing.core.route_explorer_agent import RouteExplorerAgent, RouteExplorerRequest
from eco_routing.core.road_graph import road_graph
from eco_routing.ml.traffic_predictor import predict_traffic
from eco_routing.ml.driver_profiler import infer_driver_profile
from eco_routing.ml.route_risk_detector import predict_route_risk


@dataclass
class ExplorerRequest:
    source_lat: float
    source_lon: float
    dest_lat: float
    dest_lon: float
    vehicle_id: str
    driver_id: str
    preference: Preference


class ExplorerAgent:
    """High-level orchestrator calling specialized agents."""

    def __init__(self) -> None:
        self.route_explorer = RouteExplorerAgent()
        self.fuel_agent = FuelEmissionAgent()

    def _geojson_for_path(self, path: RouteMetrics) -> Dict[str, Any]:
        if road_graph.graph is None:
            raise RuntimeError("Road graph not built")
        coords: List[List[float]] = []
        for node in path.path.nodes:
            data = road_graph.graph.nodes[node]
            x = data.get("x")
            y = data.get("y")
            if x is not None and y is not None:
                coords.append([float(x), float(y)])
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {},
        }

    def optimize(self, req: ExplorerRequest) -> Dict[str, Any]:
        explorer_req = RouteExplorerRequest(
            source_lat=req.source_lat,
            source_lon=req.source_lon,
            dest_lat=req.dest_lat,
            dest_lon=req.dest_lon,
            vehicle_id=req.vehicle_id,
            driver_id=req.driver_id,
            preference=req.preference,
        )
        paths = self.route_explorer.explore(explorer_req, max_paths=3)
        if not paths:
            raise RuntimeError("No routes found")

        ctx = SegmentCostContext(
            vehicle_id=req.vehicle_id,
            driver_id=req.driver_id,
            preference=req.preference,
        )

        segments_per_path: List[List] = []
        fuels: List[float] = []
        co2s: List[float] = []
        for path in paths:
            segs, fuel_l, co2_kg = self.fuel_agent.compute_for_path(path, ctx)
            segments_per_path.append(segs)
            fuels.append(fuel_l)
            co2s.append(co2_kg)

        ranked: List[RouteMetrics] = eco_score_agent.rank_routes(paths, segments_per_path, fuels, co2s)
        if not ranked:
            raise RuntimeError("No ranked routes available")

        # Single-route policy: strict lexicographic priority
        # 1. Lowest emissions (CO₂ kg)
        # 2. Lowest travel_time (minutes)
        # 3. Shortest distance (km)
        chosen = min(
            ranked,
            key=lambda r: (
                r.total_co2_kg,  # Priority 1: emissions
                r.path.total_time_min,  # Priority 2: travel time
                r.path.total_distance_km,  # Priority 3: distance
            ),
        )
        
        # Get ML-based information for the chosen route
        ml_outputs = {}
        
        # Traffic factor prediction
        try:
            traffic_features = {
                'hour_of_day': 12,  # Default to midday if not available
                'day_of_week': 1,   # Default to Monday if not available
                'avg_speed': chosen.path.total_distance_km / (chosen.path.total_time_min / 60) if chosen.path.total_time_min > 0 else 60.0
            }
            ml_outputs['traffic_factor'] = predict_traffic(traffic_features)
        except:
            ml_outputs['traffic_factor'] = 1.0  # Default neutral traffic factor
        
        # Driver profile prediction
        try:
            ml_outputs['driver_profile'] = infer_driver_profile({'driver_id': req.driver_id})
        except:
            ml_outputs['driver_profile'] = 'normal'  # Default profile
        
        # Route risk prediction
        try:
            route_features = {
                'distance': chosen.path.total_distance_km,
                'num_segments': len(chosen.segments),
                'avg_speed': chosen.path.total_distance_km / (chosen.path.total_time_min / 60) if chosen.path.total_time_min > 0 else 60.0,
                'elevation_change': sum(abs(seg.distance_km) for seg in chosen.segments)  # Using distance as a proxy
            }
            ml_outputs['route_risk'] = predict_route_risk(route_features)
        except:
            ml_outputs['route_risk'] = 0.0  # Default zero risk
        
        # No alternatives needed for single-route policy
        alternatives: List[RouteMetrics] = []
        explanation: RouteExplanation = explainability_agent.explain(
            chosen, 
            alternatives, 
            driver_id=req.driver_id,
            vehicle_id=req.vehicle_id,
            traffic_factor=ml_outputs['traffic_factor']
        )

        return {
            "geojson": self._geojson_for_path(chosen),
            "distance_km": chosen.path.total_distance_km,
            "time_min": chosen.path.total_time_min,
            "fuel_liters": chosen.total_fuel_l,
            "co2_kg": chosen.total_co2_kg,
            "eco_score": chosen.eco_score,
            "ml_outputs": ml_outputs,
            "explanation": {
                "top_reasons": explanation.reasons,
                "rejected_alternatives": explanation.rejected_alternatives,
            },
            "segment_breakdown": explanation.segment_breakdown,
        }


explorer_agent = ExplorerAgent()