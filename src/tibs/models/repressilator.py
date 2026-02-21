# src/tibs/models/repressilator.py
import numpy as np
from .base import DynamicalSystem, Params

class Repressilator(DynamicalSystem):
    names = ("x", "y", "z")

    def default_params(self) -> Params:
        return dict(alpha=10.0, n=2.0, k=1.0)

    def default_initial_state(self, p: Params) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3], dtype=float)

    def rhs(self, x, t, p):
        X, Y, Z = x
        alpha, n, k = p["alpha"], p["n"], p["k"]
        dX = alpha/(1.0 + Z**n) - k*X
        dY = alpha/(1.0 + X**n) - k*Y
        dZ = alpha/(1.0 + Y**n) - k*Z
        return np.array([dX, dY, dZ], dtype=float)
