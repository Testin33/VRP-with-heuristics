# validator.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from instance_data import Instance


def check_solution(
    instance: Instance,
    dist: Dict[Tuple[int, int], float],
    assign: Dict[int, int],               # customer -> store
    routes: Dict[str, List[int]],         # vehicle -> route [0,...,0]
    check_route_length: bool = True,
) -> Tuple[bool, str]:
    """
    Checks feasibility aligned with your AMPL model logic, but allowing a variable number of vehicles/routes.

    Constraints checked:
      - Each customer assigned to exactly one store (given by assign dict)
      - Inventory limits per store/product
      - Each node in N visited exactly once across ALL provided routes
      - Each route starts/ends at depot
      - Precedence: for each customer j, its assigned store i must be in same route and appear before j
      - Capacity: per route, sum D[node] for visited nodes <= Q (in your data, only stores have D>0)
      - Max route length (optional): travel + service <= L
    """

    P = instance.P
    N = instance.N
    R = set(instance.R)
    C = set(instance.C)
    depot = instance.depot

    # 0) Must have at least one route
    if not routes:
        return False, "No routes provided."

    # 1) Assignment completeness & domain
    for j in C:
        if j not in assign:
            return False, f"Missing assignment for customer {j}."
        if assign[j] not in R:
            return False, f"Customer {j} assigned to non-store {assign[j]}."

    # 2) Inventory constraints: sum d[p,j] over customers assigned to store i <= I[p,i]
    used = {(p, i): 0.0 for p in P for i in R}
    for j in C:
        i = assign[j]
        for p in P:
            used[(p, i)] += instance.d[(p, j)]

    for p in P:
        for i in R:
            if used[(p, i)] > instance.I[(p, i)] + 1e-9:
                return False, (
                    f"Inventory violated at store {i}, product {p}: "
                    f"used={used[(p, i)]}, I={instance.I[(p, i)]}"
                )

    # 3) Routing: every provided route must start/end at depot
    for k, r in routes.items():
        if not isinstance(r, list) or len(r) < 2:
            return False, f"Route for vehicle {k} is too short or invalid: {r}"
        if r[0] != depot or r[-1] != depot:
            return False, f"Route for vehicle {k} must start/end at depot: {r}"

    # 4) Each node in N visited exactly once across all routes (excluding depot)
    visit_counts = defaultdict(int)
    for k, r in routes.items():
        for node in r:
            if node != depot:
                visit_counts[node] += 1

    for node in N:
        if visit_counts[node] != 1:
            return False, f"Node {node} visited {visit_counts[node]} times (must be 1)."

    # 5) Precedence: store must be visited before its assigned customers, in the same route
    for k, r in routes.items():
        pos = {node: idx for idx, node in enumerate(r)}
        for node in r:
            if node in C:  # customer
                j = node
                i = assign[j]  # assigned store
                if i not in pos:
                    return False, f"Customer {j} in route {k} but its store {i} is not in the same route."
                if pos[i] > pos[j]:
                    return False, f"Precedence violated in route {k}: store {i} appears after customer {j}."

    # 6) Capacity per route: sum D[node] over visited nodes <= Q
    for k, r in routes.items():
        load_sum = sum(instance.D.get(node, 0.0) for node in r if node != depot)
        if load_sum > instance.Q + 1e-9:
            return False, f"Capacity violated for {k}: load={load_sum}, Q={instance.Q}."

    # 7) Max route length (optional): travel + service <= L
    if check_route_length:
        for k, r in routes.items():
            travel = 0.0
            for t in range(len(r) - 1):
                travel += dist[(r[t], r[t + 1])]
            service = sum(instance.O.get(node, 0.0) for node in r)
            total = travel + service
            if total > instance.L + 1e-9:
                return False, f"Route length violated for {k}: total={total}, L={instance.L}."

    return True, "OK"
