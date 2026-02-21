
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

Params = Dict[str, float]


class Model(ABC):
    """Base class for mechanistic biological models."""

    @abstractmethod
    def initial_state(self):
        """Return initial state vector."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> Params:
        """Return dict of model parameters."""
        raise NotImplementedError

    @abstractmethod
    def stoichiometry(self):
        """Return stoichiometry matrix (N_species x N_reactions)."""
        raise NotImplementedError

    @abstractmethod
    def propensities(self, x, t, params: Params):
        """Return reaction propensities for Gillespie."""
        raise NotImplementedError

    @abstractmethod
    def rhs(self, t, x, params: Params):
        """ODE right-hand side."""
        raise NotImplementedError

    def observe(self, x, params: Params):
        """Map internal state to observable (e.g. luminescence)."""
        return x


@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray
    names: Tuple[str, ...]


class ODEModel(Model):
    """
    Convenience base class for ODE-only models.
    - Implements SSA-specific methods with clear errors.
    - Adds a dependency-free RK4 simulator + helpers.
    """

    names: Tuple[str, ...] = ()

    def stoichiometry(self):
        raise NotImplementedError("This model is ODE-only (no stoichiometry / SSA).")

    def propensities(self, x, t, params: Params):
        raise NotImplementedError("This model is ODE-only (no propensities / SSA).")

    def simulate(
        self,
        t_span=(0.0, 50.0),
        dt=0.01,
        x0: Optional[np.ndarray] = None,
        params: Optional[Params] = None,
    ) -> SimResult:
        p: Params = self.parameters()
        if params is not None:
            p = dict(p, **params)

        x = self.initial_state() if x0 is None else np.asarray(x0, dtype=float)
        x = np.asarray(x, dtype=float)

        t0, t1 = float(t_span[0]), float(t_span[1])
        if t1 <= t0:
            raise ValueError("t_span must satisfy t_span[1] > t_span[0].")
        if dt <= 0:
            raise ValueError("dt must be > 0.")

        n = int(np.floor((t1 - t0) / dt)) + 1
        t = np.linspace(t0, t0 + dt * (n - 1), n)

        X = np.zeros((n, x.size), dtype=float)
        X[0] = x

        for i in range(n - 1):
            ti = t[i]
            xi = X[i]
            k1 = np.asarray(self.rhs(ti, xi, p), dtype=float)
            k2 = np.asarray(self.rhs(ti + 0.5 * dt, xi + 0.5 * dt * k1, p), dtype=float)
            k3 = np.asarray(self.rhs(ti + 0.5 * dt, xi + 0.5 * dt * k2, p), dtype=float)
            k4 = np.asarray(self.rhs(ti + dt, xi + dt * k3, p), dtype=float)
            X[i + 1] = xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return SimResult(t=t, x=X, names=getattr(self, "names", tuple()))

    def sweep(self, param_name: str, values, **simulate_kwargs):
        out = {}
        for v in values:
            out[float(v)] = self.simulate(params={param_name: float(v)}, **simulate_kwargs)
        return out

    def plot(self, res: SimResult):
        import matplotlib.pyplot as plt

        plt.figure()
        names = res.names if res.names else tuple(f"x{i}" for i in range(res.x.shape[1]))
        for j, name in enumerate(names):
            plt.plot(res.t, res.x[:, j], label=name)
        plt.xlabel("time")
        plt.ylabel("state")
        plt.legend()
        plt.tight_layout()
