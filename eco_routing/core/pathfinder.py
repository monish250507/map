from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from eco_routing.core.cost_function import SegmentCostContext, segment_cost_calculator
from eco_routing.core.road_graph import RoadSegment, road_graph


@dataclass
class PathResult:
    nodes: List[int]
    edges: List[Tuple[int, int, int]]
    total_cost: float
    total_distance_km: float
    total_time_min: float


class Pathfinder:
    """Dijkstra search using sustainability cost on road segments."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def _heuristic_cost(self, u: int, target: int) -> float:
        # Admissible optimistic heuristic: straight-line distance converted to a small cost.
        ux, uy = self.graph.nodes[u].get("x"), self.graph.nodes[u].get("y")
        tx, ty = self.graph.nodes[target].get("x"), self.graph.nodes[target].get("y")
        if ux is None or uy is None or tx is None or ty is None:
            return 0.0
        # Rough km distance using haversine approximation scaled down to stay optimistic.
        dx = (ux - tx) * 111.320 * abs(uy) / 90 if uy else 0
        dy = (uy - ty) * 111.0
        dist_km = (dx * dx + dy * dy) ** 0.5
        return dist_km * 0.01  # small optimistic factor

    def _edge_segments(self, u: int, v: int) -> List[Tuple[int, RoadSegment]]:
        segments: List[Tuple[int, RoadSegment]] = []
        for key in self.graph[u][v]:
            seg = road_graph.get_segment(u, v, key)
            segments.append((key, seg))
        return segments

    def shortest_path(
        self,
        source: int,
        target: int,
        ctx: SegmentCostContext,
        max_paths: int = 3,
    ) -> List[PathResult]:
        """Compute up to `max_paths` distinct low-cost paths using A* with an optimistic heuristic."""

        def astar(avoid_edges: set[Tuple[int, int, int]]) -> PathResult | None:
            heuristic_cache: Dict[int, float] = {}
            g_score: Dict[int, float] = {source: 0.0}
            heuristic_cache[source] = self._heuristic_cost(source, target)
            f_score: Dict[int, float] = {source: heuristic_cache[source]}
            prev: Dict[int, Tuple[int, int]] = {}
            pq: List[Tuple[float, int]] = [(f_score[source], source)]

            # Pre-compute cost factors to avoid lookups in the loop
            factors = segment_cost_calculator.compute_context_factors(ctx)
            cong_map = segment_cost_calculator.behavior.congestion_by_road_type
            idle_map = segment_cost_calculator.behavior.idle_penalty_by_road_type

            while pq:
                curr_f, u = heapq.heappop(pq)
                
                # Lazy deletion check (if we found a better path to u valid in PQ handling, though here g_score check usually handles it)
                if curr_f > f_score.get(u, float("inf")):
                    continue

                if u == target:
                    break
                
                curr_g = g_score[u]

                # Optimized graph iteration: access adjacency dict directly
                for v, edge_dict in self.graph[u].items():
                    for key in edge_dict:
                        edge_id = (u, v, key)
                        if edge_id in avoid_edges:
                            continue
                        
                        # Use cached segment retrieval
                        seg = road_graph.get_segment(u, v, key)
                        
                        # Use fast vector-like cost calculation
                        seg_cost = segment_cost_calculator.segment_cost_fast(
                            seg, factors, cong_map, idle_map
                        )
                        
                        tentative_g = curr_g + seg_cost
                        if tentative_g < g_score.get(v, float("inf")):
                            g_score[v] = tentative_g
                            if v not in heuristic_cache:
                                heuristic_cache[v] = self._heuristic_cost(v, target)
                            f_score[v] = tentative_g + heuristic_cache[v]
                            prev[v] = (u, key)
                            heapq.heappush(pq, (f_score[v], v))

            if target not in g_score:
                return None

            nodes: List[int] = []
            edges: List[Tuple[int, int, int]] = []
            current = target
            while current != source:
                nodes.append(current)
                u, key = prev[current]
                edges.append((u, current, key))
                current = u
            nodes.append(source)
            nodes.reverse()
            edges.reverse()

            total_distance = 0.0
            total_time_min = 0.0
            for u, v, key in edges:
                seg = road_graph.get_segment(u, v, key)
                total_distance += seg.distance_km
                if seg.speed_kph > 0:
                    total_time_min += seg.distance_km / seg.speed_kph * 60.0

            return PathResult(
                nodes=nodes,
                edges=edges,
                total_cost=g_score[target],
                total_distance_km=total_distance,
                total_time_min=total_time_min,
            )

        results: List[PathResult] = []
        avoided: set[Tuple[int, int, int]] = set()
        for _ in range(max_paths):
            res = astar(avoided)
            if not res:
                break
            results.append(res)
            if res.edges:
                avoided.add(res.edges[len(res.edges) // 2])
        return results


