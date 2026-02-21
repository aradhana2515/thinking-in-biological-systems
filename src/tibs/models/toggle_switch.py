import numpy as np

from .base import ODEModel


class ToggleSwitch(ODEModel):
    names = ("u", "v")

    def initial_state(self):
        return np.array([0.1, 0.1], dtype=float)

    def parameters(self):
        return dict(alpha=10.0, beta=10.0, n=2.0, m=2.0, ku=1.0, kv=1.0)

    def rhs(self, t, x, params):
        u, v = x
        alpha, beta = params["alpha"], params["beta"]
        n, m = params["n"], params["m"]
        ku, kv = params["ku"], params["kv"]
        du = alpha / (1.0 + v**n) - ku * u
        dv = beta / (1.0 + u**m) - kv * v
        return np.array([du, dv], dtype=float)
