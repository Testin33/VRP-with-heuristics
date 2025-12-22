# main.py
import importlib.util
import pathlib

from instance_data import load_sample_instance_from_dat
from utils import build_distance, total_distance_over_fleet
from validator import check_solution

# Cargar funciones de ploteo desde utils.py en el directorio raíz (para evitar el utils local)
_root_utils_path = pathlib.Path(__file__).resolve().parent.parent / "utils.py"
_spec = importlib.util.spec_from_file_location("plot_utils_root", _root_utils_path)
plot_utils_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_utils_root)  # type: ignore
plot_from_instance = plot_utils_root.plot_from_instance


def build_manual_feasible_solution():
    """
    Solución factible manual para esta instancia (.dat).
    """
    assign = {
        4: 1,  # p3 -> store1
        5: 1,  # p2 -> store1
        6: 1,  # p2 -> store1
        7: 2,  # p1 -> store2
        8: 3,  # p1 -> store3
        9: 2,  # p2 -> store2
    }

    routes = {
        "k1": [0, 1, 4, 5, 6, 3, 8, 0],
        "k2": [0, 2, 7, 9, 0],
    }

    return assign, routes


def build_precedence_broken_solution():
    """
    Rompe precedencia intencionalmente: cliente 4 antes que tienda 1.
    """
    assign, routes = build_manual_feasible_solution()
    routes_bad = dict(routes)
    routes_bad["k1"] = [0, 4, 1, 5, 6, 3, 8, 0]
    return assign, routes_bad


def main():
    # ===== Load instance and distances =====
    inst = load_sample_instance_from_dat()
    dist = build_distance(inst)

    print("=== OMNI-CHANNEL VRP: Smoke test ===")
    print(f"Vehicles: {inst.K}")
    print(f"Stores R: {inst.R}")
    print(f"Customers C: {inst.C}")
    print(f"Capacity Q: {inst.Q}, Route limit L: {inst.L}")

    # ===== TEST 1: Manual feasible solution =====
    assign, routes = build_manual_feasible_solution()
    ok, msg = check_solution(inst, dist, assign, routes)
    print("\n[TEST 1] Manual feasible solution:")
    print("Feasible?", ok, "|", msg)

    if ok:
        total_dist = total_distance_over_fleet(routes, dist)
        print(f"Total distance = {total_dist:.4f}")
        for k, r in routes.items():
            print(f"  {k}: {r}")

    # ===== TEST 2: Precedence-broken solution =====
    assign2, routes_bad = build_precedence_broken_solution()
    ok2, msg2 = check_solution(inst, dist, assign2, routes_bad)
    print("\n[TEST 2] Precedence-broken solution (should fail):")
    print("Feasible?", ok2, "|", msg2)
    for k, r in routes_bad.items():
        print(f"  {k}: {r}")

    # ===== PHASE 1 (optional print) =====
    from heuristic_two_phase import two_phase_heuristic_phase1_only
    assign_p1, routes_list = two_phase_heuristic_phase1_only(inst, dist)
    print("\n[PHASE 1] Greedy assignment + initial routes per store:")
    print("Assignment:", assign_p1)
    for idx, r in enumerate(routes_list, 1):
        print(f"  route_{idx}: {r}")

    # ===== FULL TWO-PHASE HEURISTIC (Phase 1 + Phase 2) =====
    from heuristic_two_phase import two_phase_heuristic
    assign_h, routes_h = two_phase_heuristic(inst, dist)
    okh, msgh = check_solution(inst, dist, assign_h, routes_h)

    print("\n[HEURISTIC] Full two-phase heuristic:")
    print("Feasible?", okh, "|", msgh)
    print("Assignment:", assign_h)
    for k, r in routes_h.items():
        print(f"  {k}: {r}")
    print(f"Total distance = {total_distance_over_fleet(routes_h, dist):.4f}")
    plot_from_instance(inst, routes_h, assign_h, title="Heuristic two-phase")


if __name__ == "__main__":
    main()

