# heuristic_two_phase.py
from __future__ import annotations

from typing import Dict, Tuple, List, Set
from instance_data import Instance

# Helpers: demand / products

def customer_product(instance: Instance, j: int) -> int:
    for p in instance.P:
        if instance.d[(p, j)] > 0:
            return p
    raise ValueError(f"Customer {j} has no positive online demand in d[p,j].")


def route_distance(route: List[int], dist: Dict[Tuple[int, int], float]) -> float:
    total = 0.0
    for t in range(len(route) - 1):
        total += dist[(route[t], route[t + 1])]
    return total

# Phase 1: assignment
def phase1_greedy_assignment(instance: Instance,
                             dist: Dict[Tuple[int, int], float]) -> Dict[int, int]:
    R: List[int] = instance.R
    C: List[int] = instance.C
    P: List[int] = instance.P

    remaining = {(p, i): float(instance.I[(p, i)]) for p in P for i in R}
    assign: Dict[int, int] = {}

    cust_prod: Dict[int, int] = {j: customer_product(instance, j) for j in C}
    total_inv = {p: sum(remaining[(p, i)] for i in R) for p in P}

    # hardest-first
    C_sorted = sorted(C, key=lambda j: (total_inv[cust_prod[j]], j))

    for j in C_sorted:
        p = cust_prod[j]
        demand = float(instance.d[(p, j)])  # usually 1
        feasible = [i for i in R if remaining[(p, i)] >= demand]
        if not feasible:
            raise ValueError(
                f"No feasible store for customer {j} (product {p}). "
                f"TotalInv(p)={total_inv[p]}, demand={demand}"
            )
        best_store = min(feasible, key=lambda i: dist[(i, j)])
        assign[j] = best_store
        remaining[(p, best_store)] -= demand

    return assign


# Phase 1: build initial routes (one per store)

def build_initial_routes_per_store(instance: Instance,
                                   dist: Dict[Tuple[int, int], float],
                                   assign: Dict[int, int]) -> List[List[int]]:
    depot = instance.depot
    routes: List[List[int]] = []

    store_to_customers: Dict[int, List[int]] = {i: [] for i in instance.R}
    for j, i in assign.items():
        store_to_customers[i].append(j)

    # order customers by nearest-neighbor starting at store
    for i in instance.R:
        customers = store_to_customers[i]
        if not customers:
            routes.append([depot, i, depot])
            continue

        unvisited = set(customers)
        current = i
        ordered: List[int] = []
        while unvisited:
            nxt = min(unvisited, key=lambda j: dist[(current, j)])
            ordered.append(nxt)
            unvisited.remove(nxt)
            current = nxt

        routes.append([depot, i] + ordered + [depot])

    return routes


def two_phase_heuristic_phase1_only(instance: Instance,
                                   dist: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, int], List[List[int]]]:
    assign = phase1_greedy_assignment(instance, dist)
    routes_list = build_initial_routes_per_store(instance, dist, assign)
    return assign, routes_list


# Feasibility checks

def _capacity_ok(instance: Instance, route: List[int]) -> bool:
    # capacity uses offline store deliveries D[i]
    total = 0.0
    for n in route:
        if n != instance.depot:
            total += float(instance.D.get(n, 0.0))
    return total <= instance.Q + 1e-9


def _precedence_ok(instance: Instance, route: List[int], assign: Dict[int, int]) -> bool:
    pos = {node: idx for idx, node in enumerate(route)}
    for node in route:
        if node in instance.C:
            j = node
            i = assign[j]
            if i not in pos:
                return False
            if pos[i] > pos[j]:
                return False
    return True


def _length_ok(instance: Instance, route: List[int], dist: Dict[Tuple[int, int], float]) -> bool:
    travel = route_distance(route, dist)
    service = sum(float(instance.O.get(n, 0.0)) for n in route)
    return (travel + service) <= instance.L + 1e-9


def is_route_feasible(instance: Instance,
                      dist: Dict[Tuple[int, int], float],
                      assign: Dict[int, int],
                      route: List[int]) -> bool:
    depot = instance.depot
    if not route or route[0] != depot or route[-1] != depot:
        return False

    seen: Set[int] = set()
    for n in route:
        if n == depot:
            continue
        if n in seen:
            return False
        seen.add(n)

    return _capacity_ok(instance, route) and _precedence_ok(instance, route, assign) and _length_ok(instance, route, dist)

# Phase 2: merge by "blocks"

def extract_blocks(route: List[int], instance: Instance, assign: Dict[int, int]) -> List[Tuple[int, List[int]]]:
    """
    From a route like [0, store, c1, c2, 0] return [(store, [c1,c2])].
    If a route has multiple stores (after merges), still returns blocks in visit order:
      [(storeA, customers_of_A_in_route), (storeB, customers_of_B_in_route), ...]
    Customers are assigned to exactly one store by `assign`.
    """
    depot = instance.depot
    nodes = [n for n in route if n != depot]

    blocks: List[Tuple[int, List[int]]] = []
    current_store = None
    current_customers: List[int] = []

    for n in nodes:
        if n in instance.R:
            # close previous block
            if current_store is not None:
                blocks.append((current_store, current_customers))
            current_store = n
            current_customers = []
        else:
            # customer
            j = n
            # customer belongs to a store; it should match current_store in a feasible route,
            # but during intermediate steps we just collect it under its assigned store.
            if current_store is None:
                # invalid structure; treat as its assigned store start
                current_store = assign[j]
                current_customers = [j]
            else:
                current_customers.append(j)

    if current_store is not None:
        blocks.append((current_store, current_customers))

    # normalize: ensure each customer appears in its store block
    #if route order was weird, we still regroup
    store_to_customers: Dict[int, List[int]] = {}
    for n in nodes:
        if n in instance.C:
            store_to_customers.setdefault(assign[n], []).append(n)

    normalized: List[Tuple[int, List[int]]] = []
    seen_stores = []
    for s, _ in blocks:
        if s not in seen_stores:
            seen_stores.append(s)
    # add any missing stores that appear only via customers
    for s in store_to_customers.keys():
        if s not in seen_stores:
            seen_stores.append(s)

    for s in seen_stores:
        normalized.append((s, store_to_customers.get(s, [])))

    return normalized


def order_customers_in_block(store: int, customers: List[int],
                            dist: Dict[Tuple[int, int], float]) -> List[int]:
    """
    Simple nearest-neighbor order inside a block starting from the store.
    """
    if not customers:
        return []

    unvisited = set(customers)
    current = store
    ordered: List[int] = []
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[(current, j)])
        ordered.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return ordered


def build_route_from_blocks(instance: Instance,
                            dist: Dict[Tuple[int, int], float],
                            blocks: List[Tuple[int, List[int]]]) -> List[int]:
    depot = instance.depot
    route: List[int] = [depot]
    for store, custs in blocks:
        route.append(store)
        ordered = order_customers_in_block(store, custs, dist)
        route.extend(ordered)
    route.append(depot)
    return route


def merge_routes_by_blocks(instance: Instance,
                           dist: Dict[Tuple[int, int], float],
                           assign: Dict[int, int],
                           ra: List[int],
                           rb: List[int],
                           order: str) -> List[int]:
    """
    order = "AB" -> blocks(A) followed by blocks(B)
    order = "BA" -> blocks(B) followed by blocks(A)
    """
    ba = extract_blocks(ra, instance, assign)
    bb = extract_blocks(rb, instance, assign)

    blocks = ba + bb if order == "AB" else bb + ba
    return build_route_from_blocks(instance, dist, blocks)


def best_saving_pair(instance: Instance,
                     dist: Dict[Tuple[int, int], float],
                     routes: List[List[int]]) -> List[Tuple[float, int, int]]:
    """
    Savings list based on terminal nodes of routes.
    """
    depot = instance.depot
    cand = []
    for a in range(len(routes)):
        for b in range(len(routes)):
            if a == b:
                continue
            ra = routes[a]
            rb = routes[b]
            a_last = ra[-2]
            b_first = rb[1]
            s = dist[(a_last, depot)] + dist[(depot, b_first)] - dist[(a_last, b_first)]
            cand.append((s, a, b))
    cand.sort(reverse=True, key=lambda x: x[0])
    return cand


def phase2_merge_routes_to_k(instance: Instance,
                             dist: Dict[Tuple[int, int], float],
                             assign: Dict[int, int],
                             routes_list: List[List[int]]) -> List[List[int]]:
    target = len(instance.K)
    routes = routes_list[:]

    while len(routes) > target:
        candidates = best_saving_pair(instance, dist, routes)
        merged = False

        for _, a_idx, b_idx in candidates:
            if a_idx >= len(routes) or b_idx >= len(routes) or a_idx == b_idx:
                continue

            ra = routes[a_idx]
            rb = routes[b_idx]

            # Try both block orders
            for order in ("AB", "BA"):
                proposal = merge_routes_by_blocks(instance, dist, assign, ra, rb, order)
                if is_route_feasible(instance, dist, assign, proposal):
                    # commit
                    i1, i2 = sorted([a_idx, b_idx], reverse=True)
                    routes.pop(i1)
                    routes.pop(i2)
                    routes.append(proposal)
                    merged = True
                    break

            if merged:
                break

        if not merged:
            # Instead of crashing, stop merging and return what we have.
            # This indicates target K is too small given Q/L, or we need stronger local search.
            return routes

    return routes


def pack_routes_to_vehicles(instance: Instance, routes_list: List[List[int]]) -> Dict[str, List[int]]:
    """
    If we have <=|K| routes, assign them to first routes and give empty-ish routes to remaining vehicles.
    But your validator expects each node visited exactly once, so extra vehicles must be unused.
    Our validator should allow unused vehicles only if you changed it; otherwise keep exactly |K| routes.
    Here we keep EXACTLY |K| by requiring same length.
    """
    if len(routes_list) != len(instance.K):
        raise ValueError(
            f"Need exactly {len(instance.K)} routes to pack, got {len(routes_list)}. "
            f"(Either increase K in generator, or allow unused vehicles in the validator/model.)"
        )
    return {k: r for k, r in zip(instance.K, routes_list)}


def two_phase_heuristic(instance: Instance,
                        dist: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, int], Dict[str, List[int]]]:
    assign, routes_list = two_phase_heuristic_phase1_only(instance, dist)
    merged_routes = phase2_merge_routes_to_k(instance, dist, assign, routes_list)

    routes: Dict[str, List[int]] = {}
    for idx, r in enumerate(merged_routes):
        routes[f"k{idx+1}"] = r

    return assign, routes

