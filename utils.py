"""
Utilidades de graficado para VRP (baseline Pyomo y heurística).
Requiere matplotlib.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Tuple, Any

import matplotlib.pyplot as plt
import pyomo.environ as pyo

# -------------------------------------------------------------
# Extracción de soluciones desde el modelo Pyomo
# -------------------------------------------------------------
def extract_assign_from_model(m: Any) -> Dict[int, int]:
    assign: Dict[int, int] = {}
    for j in m.C:
        for i in m.R:
            try:
                val = pyo.value(m.y[i, j])
            except Exception:
                val = 0
            if val is not None and val > 0.5:
                assign[int(j)] = int(i)
                break
    return assign


def extract_routes_from_model(m: Any, depot: int = 0) -> Dict[str, List[int]]:
    routes: Dict[str, List[int]] = {}
    V = list(m.V)
    for k in m.K:
        # construir sucesores elegidos
        succ = {}
        for i in V:
            for j in V:
                if i == j:
                    continue
                try:
                    val = pyo.value(m.x[k, i, j])
                except Exception:
                    val = 0
                if val is not None and val > 0.5:
                    succ[int(i)] = int(j)
        # recorrer ruta
        route = [depot]
        visited = set()
        cur = depot
        while True:
            nxt = succ.get(cur)
            if nxt is None:
                break
            route.append(nxt)
            if nxt == depot:
                break
            if nxt in visited:
                route.append(depot)
                break
            visited.add(nxt)
            cur = nxt
        if route[-1] != depot:
            route.append(depot)
        routes[str(k)] = route
    return routes


# -------------------------------------------------------------
# Plot genérico
# -------------------------------------------------------------
COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def _scatter_nodes(coords: Mapping[int, Tuple[float, float]], stores: List[int], customers: List[int], depot: int, assign: Dict[int, int] | None):
    for i in stores:
        x, y = coords[i]
        plt.scatter(x, y, marker="s", color="black", s=80, label="Store" if i == stores[0] else None)
    for j in customers:
        x, y = coords[j]
        if assign:
            store = assign.get(j)
            cidx = stores.index(store) % len(COLORS) if store in stores else 0
            color = COLORS[cidx]
        else:
            color = "gray"
        plt.scatter(x, y, marker="o", color=color, s=50, label="Customer" if j == customers[0] else None)
    dx, dy = coords[depot]
    plt.scatter(dx, dy, marker="*", color="gold", s=200, edgecolors="k", label="Depot")


def plot_routes(coords: Mapping[int, Tuple[float, float]],
                routes: Dict[str, List[int]],
                stores: List[int],
                customers: List[int],
                depot: int = 0,
                assign: Dict[int, int] | None = None,
                title: str = "Rutas",
                show: bool = True,
                save_path: str | None = None) -> None:
    plt.figure(figsize=(7, 6))
    _scatter_nodes(coords, stores, customers, depot, assign)

    for idx, (k, r) in enumerate(routes.items()):
        color = COLORS[idx % len(COLORS)]
        xs = [coords[n][0] for n in r]
        ys = [coords[n][1] for n in r]
        plt.plot(xs, ys, "-", color=color, label=f"{k} route")
        for n in r:
            plt.text(coords[n][0] + 0.5, coords[n][1] + 0.5, str(n), fontsize=8, color=color)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# -------------------------------------------------------------
# Wrappers para data dict o Instance (heurística)
# -------------------------------------------------------------
def plot_from_data(data: Mapping[str, Any],
                   routes: Dict[str, List[int]],
                   assign: Dict[int, int] | None = None,
                   title: str = "Rutas (data)",
                   show: bool = True,
                   save_path: str | None = None) -> None:
    coords = {v: (float(data["X"][v]), float(data["Y"][v])) for v in [0] + list(data["N"])}
    stores = list(data["R"])
    customers = list(data["C"])
    plot_routes(coords, routes, stores, customers, depot=0, assign=assign, title=title, show=show, save_path=save_path)


def plot_from_instance(instance: Any,
                       routes: Dict[str, List[int]],
                       assign: Dict[int, int] | None = None,
                       title: str = "Rutas (instance)",
                       show: bool = True,
                       save_path: str | None = None) -> None:
    depot = int(getattr(instance, "depot", 0))
    V = getattr(instance, "V", lambda: [])
    nodes = list(V()) if callable(V) else list(V)
    coords = {int(v): (float(instance.X[v]), float(instance.Y[v])) for v in nodes}
    stores = list(getattr(instance, "R", []))
    customers = list(getattr(instance, "C", []))
    plot_routes(coords, routes, stores, customers, depot=depot, assign=assign, title=title, show=show, save_path=save_path)