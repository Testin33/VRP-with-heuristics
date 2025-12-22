# heuristic_two_phase.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Set

from instance_data import Instance


def customer_product(instance: Instance, j: int) -> int:
    """
    Returns the product p such that d[p,j] > 0.
    Assumes each customer requests exactly one product (as in your .dat).
    """
    for p in instance.P:
        if instance.d[(p, j)] > 0:
            return p
    raise ValueError(f"Customer {j} has no positive demand for any product.")


def phase1_greedy_assignment(
    instance: Instance,
    dist: Dict[Tuple[int, int], float],
) -> Dict[int, int]:
    """
    Phase 1A: Greedy assignment of customers to stores:
    - Feasible if store has remaining inventory for the customer's requested product.
    - Choose the closest feasible store (min dist(store, customer)).
    """
    R: List[int] = instance.R
    C: List[int] = instance.C
    P: List[int] = instance.P

    remaining = {(p, i): float(instance.I[(p, i)]) for p in P for i in R}
    assign: Dict[int, int] = {}

    for j in C:
        p = customer_product(instance, j)
        feasible = [i for i in R if remaining[(p, i)] >= instance.d[(p, j)]]
        if not feasible:
            raise ValueError(f"No feasible store for customer {j} (product {p}).")

        best_store = min(feasible, key=lambda i: dist[(i, j)])
        assign[j] = best_store
        remaining[(p, best_store)] -= instance.d[(p, j)]

    return assign


def _nearest_neighbor_order(start: int, nodes: List[int], dist: Dict[Tuple[int, int], float]) -> List[int]:
    """Simple nearest-neighbor ordering starting from 'start'."""
    if not nodes:
        return []
    remaining: Set[int] = set(nodes)
    ordered: List[int] = []
    current = start
    while remaining:
        nxt = min(remaining, key=lambda u: dist[(current, u)])
        ordered.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return ordered


def phase1_build_routes_by_store(
    instance: Instance,
    dist: Dict[Tuple[int, int], float],
    assign: Dict[int, int],
) -> List[List[int]]:
    """
    Phase 1B: Build an initial route for each store i:
      route_i = [depot, i, customers_assigned_to_i (ordered), depot]

    This returns a list of routes (NOT yet mapped to vehicles K).
    Because in your instance, #stores=3 and #vehicles=2, Phase 2 will merge routes.
    """
    depot = instance.depot
    R = instance.R
    C = instance.C

    store_to_customers = defaultdict(list)
    for j in C:
        store_to_customers[assign[j]].append(j)

    routes_list: List[List[int]] = []
    for i in R:
        custs = store_to_customers[i]
        ordered = _nearest_neighbor_order(i, custs, dist)
        route = [depot, i] + ordered + [depot]
        routes_list.append(route)

    return routes_list


def pack_routes_to_vehicles_naive(instance: Instance, routes_list: List[List[int]]) -> Dict[str, List[int]]:
    """
    Naively assign each route in routes_list to a vehicle in instance.K.
    Works only if len(routes_list) <= len(K).
    For your current instance, len(routes_list)=3 and len(K)=2 -> NOT possible.
    """
    if len(routes_list) > len(instance.K):
        raise ValueError(
            f"Cannot pack {len(routes_list)} routes into {len(instance.K)} vehicles. "
            f"Need Phase 2 merging."
        )

    routes: Dict[str, List[int]] = {}
    for k, route in zip(instance.K, routes_list):
        routes[k] = route

    # If there are extra vehicles (unlikely in your case), give them a dummy depot loop.
    for k in instance.K[len(routes_list):]:
        routes[k] = [instance.depot, instance.depot]

    return routes


def two_phase_heuristic_phase1_only(
    instance: Instance,
    dist: Dict[Tuple[int, int], float],
) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    Runs Phase 1 only:
      - assignment
      - initial routes per store
    Returns (assign, routes_list).
    """
    assign = phase1_greedy_assignment(instance, dist)
    routes_list = phase1_build_routes_by_store(instance, dist, assign)
    return assign, routes_list


# ---- Placeholder for Phase 2 (to be implemented next) ----
def two_phase_heuristic(
    instance: Instance,
    dist: Dict[Tuple[int, int], float],
) -> Tuple[Dict[int, int], Dict[str, List[int]]]:
    """
    Full two-phase heuristic:
      Phase 1: greedy assignment + initial store routes
      Phase 2: merge routes until exactly len(K) routes remain, then map to vehicles

    For now: raises NotImplementedError because Phase 2 is next step.
    """
    assign, routes_list = two_phase_heuristic_phase1_only(instance, dist)

    # Phase 2 will go here (merge/savings + feasibility checks)
    raise NotImplementedError(
        "Phase 2 (route merging/savings) not implemented yet. "
        "Use two_phase_heuristic_phase1_only() to get initial routes."
    )
