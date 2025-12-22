# main_large.py
import time

from instance_generator import generate_large_instance
from utils import build_distance, total_distance_over_fleet
from validator import check_solution
from heuristic_two_phase import two_phase_heuristic


def inventory_summary(inst):
    """Stats rápidas del inventario y demanda online por producto."""
    # PD[p] = total online demand of product p
    PD = {p: 0 for p in inst.P}
    for j in inst.C:
        for p in inst.P:
            if inst.d[(p, j)] > 0:
                PD[p] += inst.d[(p, j)]
                break

    # Total inventory in network for each product
    PI = {p: 0.0 for p in inst.P}
    for p in inst.P:
        PI[p] = sum(inst.I[(p, i)] for i in inst.R)

    products_with_demand = [p for p in inst.P if PD[p] > 0]
    zero_demand = [p for p in inst.P if PD[p] == 0]

    def _minmaxavg(vals):
        if not vals:
            return (0.0, 0.0, 0.0)
        return (min(vals), max(vals), sum(vals) / len(vals))

    pd_vals = [PD[p] for p in products_with_demand]
    pi_vals = [PI[p] for p in products_with_demand]

    pd_min, pd_max, pd_avg = _minmaxavg(pd_vals)
    pi_min, pi_max, pi_avg = _minmaxavg(pi_vals)

    return {
        "n_products": len(inst.P),
        "products_with_demand": len(products_with_demand),
        "products_zero_demand": len(zero_demand),
        "PD_min": pd_min, "PD_max": pd_max, "PD_avg": pd_avg,
        "PI_min": pi_min, "PI_max": pi_max, "PI_avg": pi_avg,
    }


def run_one(n_stores: int, n_customers: int, scenario: int, seed: int = 0, n_vehicles=None):
    print("\n=== LARGE INSTANCE RUN ===")
    print(f"n_stores={n_stores}, n_customers={n_customers}, scenario={scenario}, seed={seed}")

    inst = generate_large_instance(
        n_stores=n_stores,
        n_customers=n_customers,
        scenario=scenario,
        seed=seed,
        n_vehicles=n_vehicles,
    )

    dist = build_distance(inst)

    # -------- NEW PRINTS: instance stats --------
    inv = inventory_summary(inst)
    sum_offline = sum(inst.D[i] for i in inst.R)
    cap_lb = int((sum_offline + inst.Q - 1) // inst.Q)  # ceil(sumD/Q) para Q float queda ok si Q int; simple

    print(f"Vehicles (given/estimated) |K| = {len(inst.K)}  |  Capacity Q={inst.Q}  |  Route limit L={inst.L}")
    print(f"Products |P| = {len(inst.P)}  (with demand: {inv['products_with_demand']}, zero-demand: {inv['products_zero_demand']})")
    print(f"Nodes |N| = {len(inst.N)} (stores={len(inst.R)}, customers={len(inst.C)})")
    print(f"Offline total demand sum(D_i)={sum_offline:.1f}  => capacity lower bound ceil(sumD/Q)≈{cap_lb}")

    print(f"Online demand PD stats (only products with demand): min={inv['PD_min']:.0f}, max={inv['PD_max']:.0f}, avg={inv['PD_avg']:.2f}")
    print(f"Inventory PI stats (only products with demand): min={inv['PI_min']:.1f}, max={inv['PI_max']:.1f}, avg={inv['PI_avg']:.2f}")
    # -------------------------------------------

    t0 = time.time()
    assign, routes = two_phase_heuristic(inst, dist)
    t1 = time.time()

    ok, msg = check_solution(inst, dist, assign, routes)
    total_dist = total_distance_over_fleet(routes, dist)

    # -------- NEW PRINTS: heuristic stats --------
    n_routes = len(routes)
    print("\n[HEURISTIC SUMMARY]")
    print(f"Vehicles used (routes) = {n_routes}")
    # si querés, imprime solo 2 rutas como muestra:
    sample = list(routes.items())[:2]
    for k, r in sample:
        print(f"  sample {k}: len={len(r)} | {r[:10]}{'...' if len(r) > 10 else ''}")
    # --------------------------------------------

    print("\n[RESULT]")
    print("Feasible?", ok, "|", msg)
    print(f"Total distance = {total_dist:.4f}")
    print(f"Runtime (s) = {t1 - t0:.4f}")

    return ok, total_dist, (t1 - t0)


def main():
    for scenario in (1, 2, 3):
        run_one(n_stores=10, n_customers=25, scenario=scenario, seed=0)


if __name__ == "__main__":
    main()
