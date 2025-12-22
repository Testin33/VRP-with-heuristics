# utils.py
from __future__ import annotations

import math
from typing import Dict, Tuple, List
from instance_data import Instance


def euclid(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance."""
    return math.hypot(x1 - x2, y1 - y2)


def build_distance(instance: Instance) -> Dict[Tuple[int, int], float]:
    """
    Build distance matrix dict dist[(i,j)] for all i,j in V.
    Mirrors AMPL: Cdist[i,j] = sqrt((X[i]-X[j])^2 + (Y[i]-Y[j])^2).
    """
    V = instance.V
    dist: Dict[Tuple[int, int], float] = {}
    for i in V:
        for j in V:
            if i == j:
                dist[(i, j)] = 0.0
            else:
                dist[(i, j)] = euclid(instance.X[i], instance.Y[i], instance.X[j], instance.Y[j])
    return dist


def route_distance(route: List[int], dist: Dict[Tuple[int, int], float]) -> float:
    """Sum of travel distances along a route [n0, n1, ..., nm]."""
    if len(route) < 2:
        return 0.0
    total = 0.0
    for t in range(len(route) - 1):
        total += dist[(route[t], route[t + 1])]
    return total


def route_service_time(route: List[int], O: Dict[int, float]) -> float:
    """Sum of service/drop times for nodes in the route (excluding depot is optional; here include all nodes)."""
    return sum(O[node] for node in route)


def route_total_time(route: List[int],
                     dist: Dict[Tuple[int, int], float],
                     O: Dict[int, float]) -> float:
    """
    Total route time using AMPL logic:
    Travel time = distance (Ttime = Cdist) + service time at nodes.
    """
    return route_distance(route, dist) + route_service_time(route, O)


def total_distance_over_fleet(routes: Dict[str, List[int]],
                             dist: Dict[Tuple[int, int], float]) -> float:
    """Total distance summed across all vehicle routes."""
    return sum(route_distance(r, dist) for r in routes.values())
