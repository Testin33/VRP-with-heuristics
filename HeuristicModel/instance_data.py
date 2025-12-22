# instance_data.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Instance:
    K: List[str]                 # vehicles
    P: List[int]                 # products
    N: List[int]                 # all non-depot nodes
    R: List[int]                 # retail stores
    C: List[int]                 # consumers
    depot: int                   # depot id (0)

    Q: float                     # vehicle capacity
    L: float                     # max route length

    # offline delivery demand to stores (and 0 for customers)
    D: Dict[int, float]          # D[node]

    # online demand: d[(p, j)] for product p and customer j
    d: Dict[Tuple[int, int], float]

    # inventory: I[(p, i)] for product p and store i
    I: Dict[Tuple[int, int], float]

    # service/drop time
    O: Dict[int, float]          # O[node]

    # coordinates for depot and nodes
    X: Dict[int, float]
    Y: Dict[int, float]

    @property
    def V(self) -> List[int]:
        # all nodes including depot
        return [self.depot] + self.N


def load_sample_instance_from_dat() -> Instance:
    K = ["k1", "k2"]
    P = [1, 2, 3]

    N = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    R = [1, 2, 3]
    C = [4, 5, 6, 7, 8, 9]
    depot = 0

    Q = 100.0
    L = 10000.0

    # Offline demand D (stores have positive, customers 0)
    D = {
        1: 37, 2: 42, 3: 28,
        4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0
    }

    # Online demand d[(p, customer)]
    d = {
        (1, 4): 0, (2, 4): 0, (3, 4): 1,
        (1, 5): 0, (2, 5): 1, (3, 5): 0,
        (1, 6): 0, (2, 6): 1, (3, 6): 0,
        (1, 7): 1, (2, 7): 0, (3, 7): 0,
        (1, 8): 1, (2, 8): 0, (3, 8): 0,
        (1, 9): 0, (2, 9): 1, (3, 9): 0,
    }

    # Inventory I[(p, store)]
    I = {
        (1, 1): 0, (2, 1): 2, (3, 1): 1,
        (1, 2): 1, (2, 2): 2, (3, 2): 0,
        (1, 3): 2, (2, 3): 0, (3, 3): 1,
    }

    # Coordinates X, Y (including depot 0)
    X = {0: 95, 1: 54, 2: 19, 3: 75, 4: 60, 5: 8, 6: 35, 7: 60, 8: 48, 9: 41}
    Y = {0: 66, 1: 23, 2: 40, 3: 19, 4: 20, 5: 89, 6: 42, 7: 10, 8: 93, 9: 99}

    # Service time O (all zeros in your dat)
    O = {node: 0.0 for node in [0] + N}

    return Instance(
        K=K, P=P, N=N, R=R, C=C, depot=depot,
        Q=Q, L=L, D=D, d=d, I=I, O=O, X=X, Y=Y
    )

