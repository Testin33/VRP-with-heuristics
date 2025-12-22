from __future__ import annotations

"""Plot utilities for VRP baseline Pyomo and heuristic.
To create a visual tool to analize the answer, that is the focus of this file.
"""

from typing import Any, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import pyomo.environ as pyo


def extract_assign_from_model(m: Any) -> Dict[int, int]:
    """Return customer to store assignment from a solved Pyomo model."""
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
    """Return vehicle routes from binary x[k,i,j] values."""
    routes: Dict[str, List[int]] = {}
    V = list(m.V)
    for k in m.K:
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


COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def _scatter_nodes(ax, coords: Mapping[int, Tuple[float, float]], stores: List[int], customers: List[int], depot: int, assign: Dict[int, int] | None):
    for i in stores:
        x, y = coords[i]
        ax.scatter(x, y, marker="s", color="black", s=80, label="Store" if i == stores[0] else None)
    for j in customers:
        x, y = coords[j]
        if assign:
            store = assign.get(j)
            cidx = stores.index(store) % len(COLORS) if store in stores else 0
            color = COLORS[cidx]
        else:
            color = "gray"
        ax.scatter(x, y, marker="o", color=color, s=50, label="Customer" if j == customers[0] else None)
    dx, dy = coords[depot]
    ax.scatter(dx, dy, marker="*", color="gold", s=200, edgecolors="k", label="Depot")


def plot_routes(coords: Mapping[int, Tuple[float, float]],
                routes: Dict[str, List[int]],
                stores: List[int],
                customers: List[int],
                depot: int = 0,
                assign: Dict[int, int] | None = None,
                title: str = "Routes",
                show: bool = True,
                save_path: str | None = None,
                ax=None) -> None:
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        own_fig = True

    _scatter_nodes(ax, coords, stores, customers, depot, assign)

    for idx, (k, r) in enumerate(routes.items()):
        color = COLORS[idx % len(COLORS)]
        xs = [coords[n][0] for n in r]
        ys = [coords[n][1] for n in r]
        ax.plot(xs, ys, "-", color=color, label=f"{k} route")
        for n in r:
            ax.text(coords[n][0] + 0.5, coords[n][1] + 0.5, str(n), fontsize=8, color=color)

    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9, borderaxespad=0.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close()


def plot_from_data(data: Mapping[str, Any],
                   routes: Dict[str, List[int]],
                   assign: Dict[int, int] | None = None,
                   title: str = "Routes (data)",
                   show: bool = True,
                   save_path: str | None = None) -> None:
    coords = {v: (float(data["X"][v]), float(data["Y"][v])) for v in [0] + list(data["N"])}
    stores = list(data["R"])
    customers = list(data["C"])
    plot_routes(coords, routes, stores, customers, depot=0, assign=assign, title=title, show=show, save_path=save_path)


def plot_from_instance(instance: Any,
                       routes: Dict[str, List[int]],
                       assign: Dict[int, int] | None = None,
                       title: str = "Routes (instance)",
                       show: bool = True,
                       save_path: str | None = None) -> None:
    depot = int(getattr(instance, "depot", 0))
    V = getattr(instance, "V", lambda: [])
    nodes = list(V()) if callable(V) else list(V)
    coords = {int(v): (float(instance.X[v]), float(instance.Y[v])) for v in nodes}
    stores = list(getattr(instance, "R", []))
    customers = list(getattr(instance, "C", []))
    plot_routes(coords, routes, stores, customers, depot=depot, assign=assign, title=title, show=show, save_path=save_path)


def _plot_metric_bars(ax,
                      baseline: Mapping[str, Any],
                      heuristic: Mapping[str, Any]) -> None:
    """Bar chart for distance, runtime, and vehicles used."""
    keys = [
        ("total_distance", "Total distance"),
        ("runtime", "Runtime (s)"),
        ("vehicles_used", "Vehicles used"),
    ]
    bvals = [float(baseline.get(k, 0.0)) for k, _ in keys]
    hvals = [float(heuristic.get(k, 0.0)) for k, _ in keys]
    labels = [label for _, label in keys]
    x = list(range(len(keys)))
    width = 0.36

    bars_b = ax.bar([xi - width / 2 for xi in x], bvals, width, label="Baseline")
    bars_h = ax.bar([xi + width / 2 for xi in x], hvals, width, label="Heuristic")

    def _annotate(bars, vals):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(1.0, val),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    _annotate(bars_b, bvals)
    _annotate(bars_h, hvals)

    max_val = max(bvals + hvals) if (bvals or hvals) else 1.0
    ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


def plot_metrics_only(baseline: Mapping[str, Any],
                      heuristic: Mapping[str, Any],
                      title: str = "Key metrics",
                      save_path: str | None = None,
                      show: bool = True) -> None:
    """Standalone metrics figure (distance, runtime, vehicles)."""
    fig, ax = plt.subplots(figsize=(12, 3))
    _plot_metric_bars(ax, baseline, heuristic)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison_overview(coords: Mapping[int, Tuple[float, float]],
                             stores: List[int],
                             customers: List[int],
                             baseline_routes: Dict[str, List[int]],
                             heuristic_routes: Dict[str, List[int]],
                             depot: int = 0,
                             baseline_assign: Dict[int, int] | None = None,
                             heuristic_assign: Dict[int, int] | None = None,
                             baseline_metrics: Mapping[str, Any] | None = None,
                             heuristic_metrics: Mapping[str, Any] | None = None,
                             title: str = "Baseline vs Heuristic",
                             save_path: str | None = None,
                             show: bool = True) -> None:
    """Composite figure with both route maps and key metrics."""
    baseline_metrics = baseline_metrics or {}
    heuristic_metrics = heuristic_metrics or {}

    fig = plt.figure(figsize=(12, 8))
    ax_map_base = plt.subplot2grid((2, 2), (0, 0))
    ax_map_heur = plt.subplot2grid((2, 2), (0, 1))
    ax_bars = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    plot_routes(coords, baseline_routes, stores, customers, depot=depot,
                assign=baseline_assign, title="Baseline routes", show=False, ax=ax_map_base)
    plot_routes(coords, heuristic_routes, stores, customers, depot=depot,
                assign=heuristic_assign, title="Heuristic routes", show=False, ax=ax_map_heur)

    _plot_metric_bars(ax_bars, baseline_metrics, heuristic_metrics)
    ax_bars.set_title("Key metrics")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 0.95, 0.95], w_pad=2.0, h_pad=1.2)

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
