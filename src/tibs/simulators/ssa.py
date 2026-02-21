"""
Stochastic Simulation Algorithm (SSA) / Gillespie simulation.

Model interface expected:
- initial_state() -> np.ndarray
- parameters() -> dict
- stoichiometry() -> np.ndarray  (n_species x n_rxns)
- propensities(x, t, p) -> np.ndarray (n_rxns,)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def gillespie(model: Any, t_max: float = 100.0, seed: int | None = None):
    rng = np.random.default_rng(seed)

    x = np.array(model.initial_state(), dtype=float)
    p = model.parameters()
    S = np.array(model.stoichiometry(), dtype=float)

    t = 0.0
    times = [t]
    states = [x.copy()]

    while t < t_max:
        a = np.asarray(model.propensities(x, t, p), dtype=float)
        if np.any(a < 0):
            raise ValueError("Negative propensity encountered.")
        a0 = float(a.sum())
        if a0 <= 0.0:
            break

        tau = float(rng.exponential(1.0 / a0))
        if t + tau > t_max:
            break

        r = float(rng.random() * a0)
        j = int(np.searchsorted(np.cumsum(a), r, side="right"))

        x = x + S[:, j]
        t += tau
        times.append(t)
        states.append(x.copy())

    return np.asarray(times), np.asarray(states)
