from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Dict, Any

import pyomo.environ as pyo

import utils as base_utils
from data import get_data
from model import build_model


HERE = Path(__file__).parent
HEURISTIC_DIR = HERE / "HeuristicModel"
import sys
if str(HEURISTIC_DIR) not in sys.path:
    sys.path.insert(0, str(HEURISTIC_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load {name} from {path}")
    spec.loader.exec_module(module)
    return module


heuristic_two_phase = _load_module("heuristic_two_phase", HEURISTIC_DIR / "heuristic_two_phase.py")
heuristic_utils = _load_module("heuristic_utils", HEURISTIC_DIR / "utils.py")
heuristic_data = _load_module("heuristic_instance_data", HEURISTIC_DIR / "instance_data.py")


def instance_from_data(data: Dict[str, Any]):
    """Build heuristic Instance from the baseline data dict."""
    return heuristic_data.Instance(
        K=list(data["K"]),
        P=list(data["P"]),
        N=list(data["N"]),
        R=list(data["R"]),
        C=list(data["C"]),
        depot=0,
        Q=float(data["Q"]),
        L=float(data["L"]),
        D={int(k): float(v) for k, v in data["D"].items()},
        d={(int(p), int(j)): float(v) for (p, j), v in data["d"].items()},
        I={(int(p), int(i)): float(v) for (p, i), v in data["I"].items()},
        O={int(k): float(v) for k, v in data["O"].items()},
        X={int(k): float(v) for k, v in data["X"].items()},
        Y={int(k): float(v) for k, v in data["Y"].items()},
    )


def run_baseline(data: Dict[str, Any]) -> Dict[str, Any]:
    """Solve the Pyomo model and collect metrics."""
    model = build_model(data)
    solver = pyo.SolverFactory("highs")
    t0 = time.time()
    solver.solve(model, tee=False)
    runtime = time.time() - t0

    routes = base_utils.extract_routes_from_model(model)
    assign = base_utils.extract_assign_from_model(model)
    vehicles_used = sum(1 for r in routes.values() if len(r) > 2)

    return {
        "total_distance": float(pyo.value(model.Total_Distance)),
        "runtime": runtime,
        "vehicles_used": vehicles_used,
        "routes": routes,
        "assign": assign,
    }


def run_heuristic(inst) -> Dict[str, Any]:
    """Run the two-phase heuristic and collect metrics."""
    dist = heuristic_utils.build_distance(inst)
    t0 = time.time()
    assign, routes = heuristic_two_phase.two_phase_heuristic(inst, dist)
    runtime = time.time() - t0

    vehicles_used = sum(1 for r in routes.values() if len(r) > 2)
    total_distance = heuristic_utils.total_distance_over_fleet(routes, dist)

    return {
        "total_distance": float(total_distance),
        "runtime": runtime,
        "vehicles_used": vehicles_used,
        "routes": routes,
        "assign": assign,
    }


def build_coords(data: Dict[str, Any]) -> Dict[int, tuple[float, float]]:
    return {int(v): (float(data["X"][v]), float(data["Y"][v])) for v in [0] + list(data["N"])}


def main():
    data = get_data()

    baseline = run_baseline(data)
    inst = instance_from_data(data)
    heuristic = run_heuristic(inst)

    baseline_img = HERE / "baseline_only.png"
    heuristic_img = HERE / "heuristic_only.png"
    metrics_img = HERE / "key_metrics.png"
    show_plots = True  # set to False if you only want to save images

    base_utils.plot_from_data(
        data,
        baseline["routes"],
        baseline["assign"],
        title="Baseline Pyomo",
        save_path=str(baseline_img),
        show=show_plots,
    )
    base_utils.plot_from_instance(
        inst,
        heuristic["routes"],
        heuristic["assign"],
        title="Heuristic two-phase",
        save_path=str(heuristic_img),
        show=show_plots,
    )
    base_utils.plot_metrics_only(
        baseline,
        heuristic,
        title="Key metrics",
        save_path=str(metrics_img),
        show=show_plots,
    )

    print("Baseline image saved at", baseline_img)
    print("Heuristic image saved at", heuristic_img)
    print("Metrics image saved at", metrics_img)


if __name__ == "__main__":
    main()
