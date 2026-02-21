import numpy as np

from .base import ODEModel


class SIR(ODEModel):
    names = ("S", "I", "R")

    def initial_state(self):
        return np.array([0.99, 0.01, 0.0], dtype=float)

    def parameters(self):
        return dict(beta=0.35, gamma=0.1, N=1.0)

    def rhs(self, t, x, params):
        S, I_, R = x
        beta, gamma, N = params["beta"], params["gamma"], params["N"]
        dS = -beta * S * I_ / N
        dI = beta * S * I_ / N - gamma * I_
        dR = gamma * I_
        return np.array([dS, dI, dR], dtype=float)
