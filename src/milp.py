"""Small helper that collects variables and linear constraints, then calls SciPy's HiGHS MILP solver."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class MILPBuilder:
    def __init__(self):
        self.cost, self.lb, self.ub, self.integer = [], [], [], []
        self._rows, self._cols, self._vals, self._lo, self._hi = [], [], [], [], []

    def add_var(self, cost=0.0, lb=0.0, ub=np.inf, integer=False):
        self.cost.append(cost)
        self.lb.append(lb)
        self.ub.append(ub)
        self.integer.append(1 if integer else 0)
        return len(self.cost) - 1

    def add_vars(self, n, **kw):
        return np.array([self.add_var(**kw) for _ in range(n)])

    def add_constr(self, coefs: dict, lb=-np.inf, ub=np.inf):
        """Add  lb <= sum(coef * var) <= ub.  `coefs` maps variable index -> coefficient."""
        i = len(self._lo)
        for j, v in coefs.items():
            self._rows.append(i)
            self._cols.append(int(j))
            self._vals.append(v)
        self._lo.append(lb)
        self._hi.append(ub)

    def solve(self, time_limit=120, mip_gap=1e-4):
        A = coo_matrix((self._vals, (self._rows, self._cols)), shape=(len(self._lo), len(self.cost))).tocsr()
        res = milp(
            c=np.array(self.cost),
            constraints=LinearConstraint(A, np.array(self._lo), np.array(self._hi)),
            integrality=np.array(self.integer),
            bounds=Bounds(np.array(self.lb), np.array(self.ub)),
            options={"time_limit": time_limit, "mip_rel_gap": mip_gap},
        )
        if res.status not in (0, 1) or res.x is None:
            raise RuntimeError(f"Solver failed: {res.message}")
        return res
