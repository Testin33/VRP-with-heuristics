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
# --- Phase 2: route merging (savings-style) ---

def _route_stores_customers(instance: Instance, route: List[int]) -> Tuple[Set[int], Set[int]]:
    stores = {n for n in route if n in set(instance.R)}
    customers = {n for n in route if n in set(instance.C)}
    return stores, customers


def _capacity_ok(instance: Instance, route: List[int]) -> bool:
    # Your AMPL load uses D[node]; in your data, only stores have D>0.
    load_sum = sum(instance.D.get(n, 0.0) for n in route if n != instance.depot)
    return load_sum <= instance.Q + 1e-9


def _precedence_ok(instance: Instance, route: List[int], assign: Dict[int, int]) -> bool:
    pos = {node: idx for idx, node in enumerate(route)}
    # Every customer in route must have its assigned store in route, before it.
    for j in route:
        if j in set(instance.C):
            i = assign[j]
            if i not in pos:
                return False
            if pos[i] > pos[j]:
                return False
    return True


def _length_ok(instance: Instance, route: List[int], dist: Dict[Tuple[int, int], float]) -> bool:
    travel = 0.0
    for t in range(len(route) - 1):
        travel += dist[(route[t], route[t+1])]
    service = sum(instance.O.get(n, 0.0) for n in route)
    return (travel + service) <= instance.L + 1e-9


def is_route_feasible(instance: Instance,
                      dist: Dict[Tuple[int, int], float],
                      assign: Dict[int, int],
                      route: List[int]) -> bool:
    # must start/end at depot
    if route[0] != instance.depot or route[-1] != instance.depot:
        return False
    # no duplicates (except depot)
    seen = set()
    for n in route:
        if n == instance.depot:
            continue
        if n in seen:
            return False
        seen.add(n)

    return _capacity_ok(instance, route) and _precedence_ok(instance, route, assign) and _length_ok(instance, route, dist)


def merge_two_routes_simple(route_a: List[int], route_b: List[int], depot: int) -> List[int]:
    """
    Simple concatenation merge:
      [0, ...a..., 0] + [0, ...b..., 0]  -> [0, ...a..., ...b..., 0]
    We remove the ending depot of A and starting depot of B.
    """
    assert route_a[0] == depot and route_a[-1] == depot
    assert route_b[0] == depot and route_b[-1] == depot
    return route_a[:-1] + route_b[1:]


def saving_between_routes(route_a: List[int], route_b: List[int],
                          dist: Dict[Tuple[int, int], float],
                          depot: int) -> float:
    """
    Clarke-Wright style saving using last non-depot of A and first non-depot of B.
    """
    a_last = route_a[-2]   # last node before depot
    b_first = route_b[1]   # first node after depot
    return dist[(a_last, depot)] + dist[(depot, b_first)] - dist[(a_last, b_first)]


def phase2_merge_routes_to_k(instance: Instance,
                             dist: Dict[Tuple[int, int], float],
                             assign: Dict[int, int],
                             routes_list: List[List[int]]) -> List[List[int]]:
    """
    Merge routes until we have exactly len(K) routes.
    Greedy: compute savings for all pairs, try best merges first.
    """
    target = len(instance.K)
    depot = instance.depot

    routes = routes_list[:]

    while len(routes) > target:
        # compute pair savings
        candidates = []
        for a in range(len(routes)):
            for b in range(len(routes)):
                if a == b:
                    continue
                s = saving_between_routes(routes[a], routes[b], dist, depot)
                candidates.append((s, a, b))
        # try merges from best saving to worst
        candidates.sort(reverse=True, key=lambda x: x[0])

        merged = False
        for _, a_idx, b_idx in candidates:
            if a_idx >= len(routes) or b_idx >= len(routes) or a_idx == b_idx:
                continue
            ra = routes[a_idx]
            rb = routes[b_idx]
            proposal = merge_two_routes_simple(ra, rb, depot)
            if is_route_feasible(instance, dist, assign, proposal):
                # accept merge: remove the two routes, add merged
                # remove higher index first to not shift indices
                i1, i2 = sorted([a_idx, b_idx], reverse=True)
                routes.pop(i1)
                routes.pop(i2)
                routes.append(proposal)
                merged = True
                break

        if not merged:
            raise RuntimeError(
                "Phase 2 merging failed: could not find any feasible merge to reduce routes. "
                "You may need a smarter merge (reordering) or adjust Phase 1."
            )

    return routes


def pack_routes_to_vehicles(instance: Instance, routes_list: List[List[int]]) -> Dict[str, List[int]]:
    if len(routes_list) != len(instance.K):
        raise ValueError(f"Need exactly {len(instance.K)} routes to pack, got {len(routes_list)}.")
    return {k: r for k, r in zip(instance.K, routes_list)}


def two_phase_heuristic(instance: Instance,
                        dist: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, int], Dict[str, List[int]]]:
    """
    Full two-phase heuristic:
      Phase 1: assignment + one route per store
      Phase 2: merge routes until #routes == #vehicles
    """
    assign, routes_list = two_phase_heuristic_phase1_only(instance, dist)
    merged_routes_list = phase2_merge_routes_to_k(instance, dist, assign, routes_list)
    routes = pack_routes_to_vehicles(instance, merged_routes_list)
    return assign, routes
