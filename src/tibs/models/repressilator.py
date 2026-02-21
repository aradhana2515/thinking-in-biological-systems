import numpy as np

from .base import ODEModel


class Repressilator(ODEModel):
    names = ("x", "y", "z")

    def initial_state(self):
        return np.array([0.1, 0.2, 0.3], dtype=float)

    def parameters(self):
        return dict(alpha=10.0, n=2.0, k=1.0)

    def rhs(self, t, x, params):
        X, Y, Z = x
        alpha, n, k = params["alpha"], params["n"], params["k"]
        dX = alpha / (1.0 + Z**n) - k * X
        dY = alpha / (1.0 + X**n) - k * Y
        dZ = alpha / (1.0 + Y**n) - k * Z
        return np.array([dX, dY, dZ], dtype=float)
