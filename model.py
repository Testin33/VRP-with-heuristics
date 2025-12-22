# model.py
from math import sqrt
import pyomo.environ as pyo


def build_model(data: dict) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # --------------------
    # SETS
    # --------------------
    m.K = pyo.Set(initialize=list(data["K"]))  # Vehicles
    m.P = pyo.Set(initialize=list(data["P"]))  # Products
    m.N = pyo.Set(initialize=list(data["N"]))  # Nodes (retail + consumers)
    m.R = pyo.Set(within=m.N, initialize=list(data["R"]))  # Retail stores
    m.C = pyo.Set(within=m.N, initialize=list(data["C"]))  # Consumers

    # V = {0} union N
    m.V = pyo.Set(initialize=[0] + list(m.N.value))

    # --------------------
    # PARAMETERS (scalars)
    # --------------------
    m.Q = pyo.Param(initialize=float(data["Q"]), within=pyo.NonNegativeReals)
    m.L = pyo.Param(initialize=float(data["L"]), within=pyo.NonNegativeReals)
    m.Mbig = pyo.Param(initialize=float(data.get("Mbig", 1e5)), within=pyo.NonNegativeReals)

    # --------------------
    # PARAMETERS (indexed)
    # --------------------
    D = data["D"]   # {n: val}
    d = data["d"]   # {(p,c): val}
    I = data["I"]   # {(p,r): val}
    O = data["O"]   # {v: val}
    X = data["X"]   # {v: val}
    Y = data["Y"]   # {v: val}

    m.D = pyo.Param(m.N, initialize=lambda _, n: float(D.get(n, 0.0)), within=pyo.NonNegativeReals)

    m.d = pyo.Param(
        m.P, m.C,
        initialize=lambda _, p, c: float(d.get((p, c), 0.0)),
        within=pyo.NonNegativeReals
    )

    m.I = pyo.Param(
        m.P, m.R,
        initialize=lambda _, p, r: float(I.get((p, r), 0.0)),
        within=pyo.NonNegativeReals
    )

    m.O = pyo.Param(m.V, initialize=lambda _, v: float(O.get(v, 0.0)), within=pyo.NonNegativeReals)
    m.X = pyo.Param(m.V, initialize=lambda _, v: float(X[v]), within=pyo.NonNegativeReals)
    m.Y = pyo.Param(m.V, initialize=lambda _, v: float(Y[v]), within=pyo.NonNegativeReals)

    # --------------------
    # DISTANCE + TRAVEL TIME
    # --------------------
    def dist_rule(_, i, j):
        if i == j:
            return 0.0
        xi, yi = float(X[i]), float(Y[i])
        xj, yj = float(X[j]), float(Y[j])
        return sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    m.Cdist = pyo.Param(m.V, m.V, initialize=dist_rule, within=pyo.NonNegativeReals, mutable=False)
    m.Ttime = pyo.Param(m.V, m.V, initialize=lambda _, i, j: pyo.value(m.Cdist[i, j]),
                        within=pyo.NonNegativeReals, mutable=False)

    # --------------------
    # VARIABLES
    # --------------------
    m.x = pyo.Var(m.K, m.V, m.V, within=pyo.Binary)  # vehicle k travels i->j
    m.y = pyo.Var(m.R, m.C, within=pyo.Binary)       # consumer j assigned to store i

    # vehicle load at each node
    m.veh_load = pyo.Var(m.K, m.V, within=pyo.NonNegativeReals,
                         bounds=lambda _, k, i: (0, pyo.value(m.Q)))
    m.s = pyo.Var(m.K, m.V, within=pyo.NonNegativeReals)  # service start time

    # --------------------
    # OBJECTIVE
    # --------------------
    def obj_rule(m):
        return sum(m.Cdist[i, j] * m.x[k, i, j] for k in m.K for i in m.V for j in m.V)
    m.Total_Distance = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --------------------
    # CONSTRAINTS
    # --------------------

    # No self loops
    def no_self_rule(m, k, i):
        return m.x[k, i, i] == 0
    m.No_Self_Loops = pyo.Constraint(m.K, m.V, rule=no_self_rule)

    # Every node visited exactly once (incoming across all vehicles)
    def visit_once_rule(m, j):
        return sum(m.x[k, i, j] for k in m.K for i in m.V if i != j) == 1
    m.Visit_Once = pyo.Constraint(m.N, rule=visit_once_rule)

    # Each vehicle starts at depot
    def start_depot_rule(m, k):
        return sum(m.x[k, 0, j] for j in m.N) == 1
    m.Start_Depot = pyo.Constraint(m.K, rule=start_depot_rule)

    # Each vehicle returns to depot
    def end_depot_rule(m, k):
        return sum(m.x[k, i, 0] for i in m.N) == 1
    m.End_Depot = pyo.Constraint(m.K, rule=end_depot_rule)

    # Flow conservation
    def flow_balance_rule(m, k, n):
        return (sum(m.x[k, n, j] for j in m.V if j != n)
                == sum(m.x[k, j, n] for j in m.V if j != n))
    m.Flow_Balance = pyo.Constraint(m.K, m.N, rule=flow_balance_rule)

    # Assignment: each consumer assigned to exactly one store
    def assign_consumer_rule(m, j):
        return sum(m.y[i, j] for i in m.R) == 1
    m.Assign_Consumer = pyo.Constraint(m.C, rule=assign_consumer_rule)

    # Inventory must cover assigned demand
    def inventory_limit_rule(m, i, p):
        return sum(m.d[p, j] * m.y[i, j] for j in m.C) <= m.I[p, i]
    m.Inventory_Limit = pyo.Constraint(m.R, m.P, rule=inventory_limit_rule)

    # Linking routing & assignment (copied from AMPL)
    def link_sc_1_rule(m, i, j, k):
        return (sum(m.x[k, i, v] for v in m.V) - sum(m.x[k, v, j] for v in m.V)
                <= m.Mbig * (1 - m.y[i, j]))
    m.Link_SC_1 = pyo.Constraint(m.R, m.C, m.K, rule=link_sc_1_rule)

    def link_sc_2_rule(m, i, j, k):
        return (sum(m.x[k, v, j] for v in m.V) - sum(m.x[k, i, v] for v in m.V)
                <= m.Mbig * (1 - m.y[i, j]))
    m.Link_SC_2 = pyo.Constraint(m.R, m.C, m.K, rule=link_sc_2_rule)

    # Load propagation (same inequality direction as your AMPL)
    def load_prop_rule(m, k, i, j):
        if i == j:
            return pyo.Constraint.Skip
        # load[k,j] + D[j] - load[k,i] <= M(1 - x[k,i,j])
        return m.veh_load[k, j] + m.D[j] - m.veh_load[k, i] <= m.Mbig * (1 - m.x[k, i, j])
    m.Load_Propagation = pyo.Constraint(m.K, m.V, m.N, rule=load_prop_rule)

    # Depot time
    def depot_time_rule(m, k):
        return m.s[k, 0] == 0
    m.Depot_Time = pyo.Constraint(m.K, rule=depot_time_rule)

    # Time consistency on arcs
    def time_arc_rule(m, k, i, j):
        if i == j:
            return pyo.Constraint.Skip
        return m.s[k, j] >= m.s[k, i] + m.Ttime[i, j] + m.O[i] - m.Mbig * (1 - m.x[k, i, j])
    m.Time_Arc = pyo.Constraint(m.K, m.V, m.N, rule=time_arc_rule)

    # Store must be served before its consumers
    def precedence_sc_rule(m, i, j, k):
        return m.s[k, j] >= m.s[k, i] + m.Ttime[i, j] + m.O[i] - m.Mbig * (1 - m.y[i, j])
    m.Precedence_SC = pyo.Constraint(m.R, m.C, m.K, rule=precedence_sc_rule)

    # Route length limit
    def max_route_len_rule(m, k, i):
        return (m.s[k, i] + m.O[i] + m.Ttime[i, 0]
                <= m.L + m.Mbig * (1 - sum(m.x[k, i, j] for j in m.V if j != i)))
    m.Max_Route_Length = pyo.Constraint(m.K, m.N, rule=max_route_len_rule)

    return m


def solve_model(m: pyo.ConcreteModel, solver_name="highs", tee=True):
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(
            f"Solver '{solver_name}' not available. "
            f"Try: pip install highspy (for HiGHS) or install another solver."
        )
    return solver.solve(m, tee=tee)
