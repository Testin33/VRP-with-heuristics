# main.py
import pyomo.environ as pyo
from data import get_data
from model import build_model
from utils import extract_routes_from_model, extract_assign_from_model, plot_from_data


def print_active_arcs(m):
    print("\n---- Active arcs (x[k,i,j] = 1) ----")
    for k in m.K:
        for i in m.V:
            for j in m.V:
                if i != j and pyo.value(m.x[k,i,j]) > 0.5:
                    print(f"x[{k},{i},{j}] = 1")

def print_routes(m):
    print("\n---- Path per vehicle ----")
    V = list(m.V)
    for k in m.K:
        # start from depot
        nxt = None
        for j in m.N:
            if pyo.value(m.x[k,0,j]) > 0.5:
                nxt = j
                break
        if nxt is None:
            print(f"Vehicle {k}: is not used.")
            continue

        path = [0, nxt]
        cur = nxt
        while cur != 0:
            found = None
            for h in V:
                if h != cur and pyo.value(m.x[k,cur,h]) > 0.5:
                    found = h
                    break
            if found is None:
                break
            path.append(found)
            cur = found
            if len(path) > 200:
                break
        print(f"Vehicle {k}: " + " -> ".join(map(str, path)))

if __name__ == "__main__":
    data = get_data()
    m = build_model(data)

    solver = pyo.SolverFactory("highs")
    res = solver.solve(m, tee=True)

    print("\n")
    print("Total_Distance =", pyo.value(m.Total_Distance))

    print_active_arcs(m)
    print_routes(m)

    assign_opt = extract_assign_from_model(m)
    routes_opt = extract_routes_from_model(m)
    plot_from_data(data, routes_opt, assign_opt, title="Baseline Pyomo")
