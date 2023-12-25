import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint


class DimError(Exception):
    pass


class COptimizerPortfolio(object):
    def __init__(self, m: np.ndarray, v: np.ndarray):
        self.m = m
        self.v = v
        self.p, _ = self.v.shape
        if self.p != _:
            raise DimError
        if self.p != len(self.m):
            raise DimError

    @property
    def variable_n(self) -> int:
        return self.p

    def returns(self, w: np.ndarray):
        return w @ self.m

    def covariance(self, w: np.ndarray):
        return w @ self.v @ w

    def volatility(self, w: np.ndarray):
        return self.covariance(w) ** 0.5

    def utility(self, w: np.ndarray, lbd: float):
        return -self.returns(w) + 0.5 * lbd * self.covariance(w)

    def sharpe(self, w: np.ndarray):
        return self.returns(w) / self.volatility(w)

    def utility_sharpe(self, w: np.ndarray):
        return -self.sharpe(w)

    @staticmethod
    def _parse_res(func):
        def parse_res_for_fun(self, **kwargs):
            res = func(self, **kwargs)
            if res.success:
                return res.x, res.fun
            else:
                print("ERROR! Optimizer exits with a failure")
                print("Detailed Description: {}".format(res.message))
                return None, None

        return parse_res_for_fun

    @_parse_res
    def optimize_utility(self, lbd: float, bounds: list[tuple[float, float]], max_iter: int = 50000):
        cons = NonlinearConstraint(lambda z: np.sum(np.abs(z)), lb=0, ub=1)  # control total market value
        res = minimize(
            fun=self.utility, x0=np.ones(self.p) / self.p,
            args=(lbd,),
            bounds=bounds,
            constraints=cons,
            options={"maxiter": max_iter}
        )
        return res

    @_parse_res
    def optimize_sharpe(self, bounds: list[tuple[float, float]], max_iter: int = 50000):
        cons = NonlinearConstraint(lambda z: np.sum(np.abs(z)), lb=0, ub=1)  # control total market value
        res = minimize(
            fun=self.utility_sharpe, x0=np.ones(self.p) / self.p,
            bounds=bounds,
            constraints=cons,
            options={"maxiter": max_iter}
        )
        return res


if __name__ == "__main__":
    import sys

    m0 = np.array([0.9, 0, -0.6])
    v0 = np.array([
        [1.1, 0.2, -0.1],
        [0.2, 1.2, 0.15],
        [-0.1, 0.15, 1.3],
    ])
    l0 = float(sys.argv[1])
    w0 = np.array([0.2, 0.3, 0.5])
    opt_po = COptimizerPortfolio(m=m0, v=v0)
    p = opt_po.variable_n
    bounds0 = [(1 / p / 1.5, 1.5 / p)] * p
    w_opt_sr, _ = opt_po.optimize_sharpe(bounds=bounds0)
    w_opt_l0, _ = opt_po.optimize_utility(lbd=l0, bounds=bounds0)

    print("=" * 24)
    print(pd.DataFrame({"raw": w0, "opt_sr": w_opt_sr, "opt_l0": w_opt_l0}))
    print("-" * 24)
    print(f"raw Sharpe : {opt_po.sharpe(w0):>9.6f}")
    print(f"opt Sharpe : {opt_po.sharpe(w_opt_sr):>9.6f}")
    print(f"raw Utility: {opt_po.utility(w0, l0):>9.6f}")
    print(f"opt Utility: {opt_po.utility(w_opt_l0, l0):>9.6f}")
