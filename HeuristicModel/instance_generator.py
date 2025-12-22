# instance_generator.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import numpy as np

from instance_data import Instance


def generate_large_instance(
    n_stores: int,
    n_customers: int,
    scenario: int,
    seed: int = 0,
    n_vehicles: Optional[int] = None,
    n_products: Optional[int] = None,
    coord_low: float = 0.0,
    coord_high: float = 100.0,
    store_demand_low: int = 25,
    store_demand_high: int = 50,
    Q: float = 100.0,
    # usamos MINUTOS como unidad
    L_minutes: float = 8.0 * 60.0,
    drop_time_minutes: float = 5.0,
) -> Instance:
    """
    Large instance generator aligned with paper Section 5.1, fixed to be FEASIBLE for 1-unit demands:
      - Coordinates random in [0,100]
      - Store offline demand U[25,50]
      - Vehicle capacity Q=100
      - Each customer chooses exactly one product (qty=1)
      - Inventory scenarios based on PDp (total online demand for product p):
          1) tight:   PI_p = ceil(PD_p*(1+U[0.1,0.2]))
          2) relaxed: PI_p = ceil(PD_p*(1+U[0.5,1.0]))
          3) abundant: I[p,i] = PD_p for all stores i
    For scenarios 1&2, PI_p is distributed across stores as integer units using multinomial.
    """

    if scenario not in (1, 2, 3):
        raise ValueError("scenario must be 1, 2, or 3")

    rng = np.random.default_rng(seed)

    # --- Sets ---
    depot = 0
    R = list(range(1, n_stores + 1))
    C = list(range(n_stores + 1, n_stores + n_customers + 1))
    N = R + C

    # Products
    if n_products is None:
        n_products = int(rng.integers(10, 21))  # [10,20]
    P = list(range(1, n_products + 1))

    # --- Offline store demand D[i] ---
    store_D_values = rng.integers(store_demand_low, store_demand_high + 1, size=n_stores)

    # Vehicles: estimate if not given
    if n_vehicles is None:
        est = int(np.ceil(store_D_values.sum() / Q))
        n_vehicles = max(1, est)
    K = [f"k{i+1}" for i in range(n_vehicles)]

    # --- Coordinates ---
    V = [depot] + N
    X: Dict[int, float] = {}
    Y: Dict[int, float] = {}

    X[depot] = float(rng.uniform(coord_low, coord_high))
    Y[depot] = float(rng.uniform(coord_low, coord_high))
    for node in N:
        X[node] = float(rng.uniform(coord_low, coord_high))
        Y[node] = float(rng.uniform(coord_low, coord_high))

    # --- D[node] ---
    D: Dict[int, float] = {}
    for idx, i in enumerate(R):
        D[i] = float(store_D_values[idx])
    for j in C:
        D[j] = 0.0

    # --- Online demand d[(p,j)] : each customer chooses one product, qty=1 ---
    d: Dict[Tuple[int, int], float] = {}
    chosen_products = rng.integers(1, n_products + 1, size=n_customers)

    for j_idx, j in enumerate(C):
        pj = int(chosen_products[j_idx])
        for p in P:
            d[(p, j)] = 1.0 if p == pj else 0.0

    # --- PD[p] total online demand per product ---
    PD = {p: 0 for p in P}
    for j in C:
        # exactly one product has 1.0
        for p in P:
            if d[(p, j)] > 0:
                PD[p] += 1
                break

    # --- Inventory I[(p,i)] ---
    I: Dict[Tuple[int, int], float] = {}

    if scenario in (1, 2):
        if scenario == 1:
            a, b = 0.1, 0.2
        else:
            a, b = 0.5, 1.0

        for p in P:
            PDp = PD[p]
            if PDp == 0:
                for i in R:
                    I[(p, i)] = 0.0
                continue

            u = float(rng.uniform(a, b))
            total_PI = int(np.ceil(PDp * (1.0 + u)))  # IMPORTANT: integer units

            # Distribute integer units across stores
            # Use random weights -> probabilities -> multinomial integer allocation
            weights = rng.random(len(R))
            probs = weights / weights.sum()
            alloc = rng.multinomial(total_PI, probs)

            # alloc is integer counts per store
            for idx, i in enumerate(R):
                I[(p, i)] = float(alloc[idx])

    else:
        # abundant: each store can satisfy all online demand for each product
        for p in P:
            for i in R:
                I[(p, i)] = float(PD[p])

    # --- Service times O[node] ---
    O: Dict[int, float] = {node: float(drop_time_minutes) for node in V}
    O[depot] = 0.0

    return Instance(
        K=K, P=P, N=N, R=R, C=C, depot=depot,
        Q=float(Q), L=float(L_minutes),
        D=D, d=d, I=I, O=O, X=X, Y=Y
    )
