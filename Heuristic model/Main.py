# main.py
from instance_data import load_sample_instance_from_dat
from utils import build_distance, total_distance_over_fleet
from validator import check_solution


def build_manual_feasible_solution():
    """
    Construye una solución factible PARA ESTA INSTANCIA (.dat).
    Importante:
    - K = {k1,k2} => 2 rutas
    - N debe visitarse exactamente 1 vez
    - Cada cliente asignado a una tienda con inventario suficiente
    - Precedencia: tienda antes que sus clientes en la misma ruta
    - Capacidad Q=100 sobre suma D (solo tiendas tienen D>0): 
      37+42+28 = 107 => no pueden ir las 3 tiendas en un solo vehículo.
      Entonces: 2 tiendas en una ruta y 1 tienda en la otra.
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
    Solución que rompe la precedencia intencionalmente:
    pone un cliente antes que su tienda en la misma ruta.
    """
    assign, routes = build_manual_feasible_solution()
    # Rompemos: en k1 ponemos el cliente 4 antes que la tienda 1
    routes_bad = dict(routes)
    routes_bad["k1"] = [0, 4, 1, 5, 6, 3, 8, 0]
    return assign, routes_bad


def main():
    inst = load_sample_instance_from_dat()
    dist = build_distance(inst)

    print("=== OMNI-CHANNEL VRP: Smoke test ===")
    print(f"Vehicles: {inst.K}")
    print(f"Stores R: {inst.R}")
    print(f"Customers C: {inst.C}")
    print(f"Capacity Q: {inst.Q}, Route limit L: {inst.L}")

    # 1) Test con solución factible
    assign, routes = build_manual_feasible_solution()
    ok, msg = check_solution(inst, dist, assign, routes)
    print("\n[TEST 1] Manual feasible solution:")
    print("Feasible?", ok, "|", msg)

    if ok:
        total_dist = total_distance_over_fleet(routes, dist)
        print(f"Total distance = {total_dist:.4f}")
        for k, r in routes.items():
            print(f"  {k}: {r}")

    # 2) Test con solución que rompe precedencia
    assign2, routes_bad = build_precedence_broken_solution()
    ok2, msg2 = check_solution(inst, dist, assign2, routes_bad)
    print("\n[TEST 2] Precedence-broken solution (should fail):")
    print("Feasible?", ok2, "|", msg2)
    for k, r in routes_bad.items():
        print(f"  {k}: {r}")


if __name__ == "__main__":
    main()
