from __future__ import annotations

from typing import Any, Dict, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from eco_routing.api.auth import require_api_key
from eco_routing.core.cost_function import Preference
from eco_routing.core.explorer_agent import ExplorerRequest, explorer_agent


class Point(BaseModel):
    lat: float
    lon: float


class OptimizeRequest(BaseModel):
    source: Point
    destination: Point
    vehicle_id: str = Field(..., description="Vehicle identifier")
    driver_id: str = Field(..., description="Driver identifier")
    preference: Preference


router = APIRouter()


@router.post("/eco-route")
def get_eco_route(body: OptimizeRequest, api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
    """
    Eco-routing endpoint that returns route with ML predictions.
    Accepts origin and destination coordinates and returns eco-friendly route with ML insights.
    """
    try:
        req = ExplorerRequest(
            source_lat=body.source.lat,
            source_lon=body.source.lon,
            dest_lat=body.destination.lat,
            dest_lon=body.destination.lon,
            vehicle_id=body.vehicle_id,
            driver_id=body.driver_id,
            preference=body.preference,
        )
        result = explorer_agent.optimize(req)
        
        # Format the response to match the requirements
        response = {
            "route_coordinates": [],  # Extract coordinates from geojson
            "fuel_consumption": result["fuel_liters"],
            "co2_emissions": result["co2_kg"],
            "eco_score": result["eco_score"],
            "ml_outputs": result["ml_outputs"],
            "explanation": result["explanation"]["top_reasons"][0] if result["explanation"]["top_reasons"] else "Route optimized based on eco criteria",
        }
        
        # Extract coordinates from the geojson
        if "geojson" in result and result["geojson"]:
            coordinates = result["geojson"]["geometry"]["coordinates"]
            response["route_coordinates"] = [{"lat": coord[1], "lng": coord[0]} for coord in coordinates]
        
        # Add other metrics
        response["distance_km"] = result["distance_km"]
        response["time_min"] = result["time_min"]
        response["segment_breakdown"] = result["segment_breakdown"]
        
        return response
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/optimize")
def optimize_route(body: OptimizeRequest, api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
    """
    Legacy optimization endpoint for backward compatibility.
    """
    try:
        req = ExplorerRequest(
            source_lat=body.source.lat,
            source_lon=body.source.lon,
            dest_lat=body.destination.lat,
            dest_lon=body.destination.lon,
            vehicle_id=body.vehicle_id,
            driver_id=body.driver_id,
            preference=body.preference,
        )
        result = explorer_agent.optimize(req)
        return {
            "geojson": result["geojson"],
            "distance_km": result["distance_km"],
            "time_min": result["time_min"],
            "fuel_liters": result["fuel_liters"],
            "co2_kg": result["co2_kg"],
            "eco_score": result["eco_score"],
            "explanation": result["explanation"],
            "segment_breakdown": result["segment_breakdown"],
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc