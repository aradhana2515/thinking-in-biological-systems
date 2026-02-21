# src/tibs/models/sir.py
import numpy as np
from .base import DynamicalSystem, Params

class SIR(DynamicalSystem):
    names = ("S", "I", "R")

    def default_params(self) -> Params:
        return dict(beta=0.35, gamma=0.1, N=1.0)

    def default_initial_state(self, p: Params) -> np.ndarray:
        # normalized population
        return np.array([0.99, 0.01, 0.0], dtype=float)

    def rhs(self, t, x, params):
        S, I, R = x
        beta, gamma, N = p["beta"], p["gamma"], p["N"]
        dS = -beta * S * I / N
        dI =  beta * S * I / N - gamma * I
        dR =  gamma * I
        return np.array([dS, dI, dR], dtype=float)
