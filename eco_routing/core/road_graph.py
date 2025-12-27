from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon

from eco_routing.config.settings import settings


# Authoritative bbox format: (min_lon, min_lat, max_lon, max_lat)
# This is the standard GeoJSON/OSM bbox representation.
Bbox = Tuple[float, float, float, float]


def bbox_to_poly(bbox: Bbox) -> Polygon:
    """
    Pure utility function: Convert bbox tuple to Shapely Polygon.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)

    Returns:
        Shapely Polygon covering the bbox.

    Raises:
        ValueError: If bbox does not have exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements (min_lon, min_lat, max_lon, max_lat), got {len(bbox)}")
    min_lon, min_lat, max_lon, max_lat = bbox
    return Polygon(
        [
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, max_lat),
            (max_lon, min_lat),
            (min_lon, min_lat),
        ]
    )


def bbox_contains(outer: Bbox, inner: Bbox) -> bool:
    """
    Check if outer bbox completely contains inner bbox.

    Args:
        outer: (min_lon, min_lat, max_lon, max_lat)
        inner: (min_lon, min_lat, max_lon, max_lat)

    Returns:
        True if outer contains inner, False otherwise.
    """
    o_min_lon, o_min_lat, o_max_lon, o_max_lat = outer
    i_min_lon, i_min_lat, i_max_lon, i_max_lat = inner
    return (o_min_lon <= i_min_lon and o_max_lon >= i_max_lon and
            o_min_lat <= i_min_lat and o_max_lat >= i_max_lat)


@dataclass
class RoadSegment:
    u: int
    v: int
    distance_km: float
    speed_kph: float
    road_type: str
    elevation_delta: float
    congestion_factor: float
    weather_factor: float


class RoadGraph:
    """Builds and maintains a road graph from OpenStreetMap with caching."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph | None = None
        self._cache: Dict[Bbox, Tuple[nx.MultiDiGraph, Bbox]] = {}
        self._node_cache: Dict[Tuple[float, float], int] = {}

    def _deg_span_from_km(self, km: float, lat: float) -> Tuple[float, float]:
        lat_deg = km / 111.0
        # Use cosine of latitude to scale longitude degrees; clamp to avoid poles.
        lat_rad = math.radians(max(-85.0, min(85.0, lat)))
        cos_lat = max(0.2, math.cos(lat_rad))
        lon_deg = km / (111.320 * cos_lat)
        return lat_deg, lon_deg

    def _bbox_key(self, bbox: Bbox) -> Bbox:
        """
        Quantize bbox to 2 decimal places for cache key.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)

        Returns:
            Quantized bbox tuple.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        return (
            round(min_lon, 2),
            round(min_lat, 2),
            round(max_lon, 2),
            round(max_lat, 2),
        )

    def build_graph(
        self,
        source_lat: float,
        source_lon: float,
        dest_lat: float | None = None,
        dest_lon: float | None = None,
    ) -> nx.MultiDiGraph:
        print("🚨 BUILDING OSM GRAPH 🚨")
        # Reuse the global graph built at startup
        if self.graph is not None:
            return self.graph
        # Fallback if not built (should not happen)
        raise RuntimeError("Global graph not built")

    def nearest_node(self, lat: float, lon: float) -> int:
        if self.graph is None:
            raise RuntimeError("Graph not built")
        key = (lat, lon)
        if key in self._node_cache:
            return self._node_cache[key]
        node = int(ox.distance.nearest_nodes(self.graph, lon, lat))
        self._node_cache[key] = node
        return node

    def get_segment(self, u: int, v: int, key: int) -> RoadSegment:
        if self.graph is None:
            raise RuntimeError("Graph not built")
        
        # Handle different OSMnx versions - try both int and str keys
        segment_data: Dict[str, Any] = {}
        key_as_str = str(key)
        
        # First try the original key (which might be int)
        if u in self.graph and v in self.graph[u]:
            # Check if the key exists directly
            if key in self.graph[u][v]:
                segment_data = self.graph[u][v][key]
            elif key_as_str in self.graph[u][v]:
                segment_data = self.graph[u][v][key_as_str]
            else:
                # If both fail, try to get the first available key
                available_keys = list(self.graph[u][v].keys())
                if available_keys:
                    first_key = available_keys[0]
                    segment_data = self.graph[u][v][first_key]
                else:
                    # If no keys available, return default values
                    segment_data = {}
        else:
            # If nodes don't exist, return default values
            segment_data = {}
        
        if "_cached_segment" in segment_data:
            return segment_data["_cached_segment"]

        distance_km = float(segment_data.get("length", 0.0)) / 1000.0
        speed_kph = float(segment_data.get("speed_kph", 30.0))
        road_type = str(segment_data.get("highway", "unclassified"))
        elevation_delta = float(segment_data.get("grade", 0.0))
        congestion_factor = float(segment_data.get("congestion_factor", 1.0))
        weather_factor = float(segment_data.get("weather_factor", 1.0))
        
        seg = RoadSegment(
            u=u,
            v=v,
            distance_km=distance_km,
            speed_kph=speed_kph,
            road_type=road_type,
            elevation_delta=elevation_delta,
            congestion_factor=congestion_factor,
            weather_factor=weather_factor,
        )
        segment_data["_cached_segment"] = seg
        return seg


def build_global_graph():
    """Build a global graph covering a wide area at startup."""
    # Use the correct parameters for graph_from_bbox function
    # The correct order is: north, south, east, west
    # Define bounding box as (north, south, east, west)
    bbox = (28.7041, 28.6139, 77.2090, 77.1025)
    global_graph = ox.graph_from_bbox(
        bbox,
        network_type="drive"
    )
    global_graph = ox.add_edge_speeds(global_graph)
    global_graph = ox.add_edge_lengths(global_graph)
    road_graph.graph = global_graph


road_graph = RoadGraph()