"""
Ensemble simulation helpers.

Runs many stochastic simulations (SSA/Gillespie) to estimate distributions over trajectories.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .ssa import gillespie


def run_ensemble(
    model,
    n: int = 200,
    t_max: float = 100.0,
    seed: int = 0,
) -> List[List[Tuple[float, np.ndarray]]]:
    """
    Run n independent SSA simulations for a model up to time t_max.

    Returns:
        trajectories: list of trajectories, each trajectory is a list of (t, x) pairs.
    """
    rng = np.random.default_rng(seed)
    trajectories: List[List[Tuple[float, np.ndarray]]] = []

    for _ in range(n):
        run_seed = int(rng.integers(0, 2**31 - 1))
        traj = gillespie(model, t_max=t_max, seed=run_seed)
        trajectories.append(traj)

    return trajectories
