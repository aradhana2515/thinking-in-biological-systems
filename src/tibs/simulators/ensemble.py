from .ssa import gillespie
import numpy as np

def run_ensemble(model, n=200, t_max=100, seed=0):
    rng = np.random.default_rng(seed)
    trajectories = []

    for i in range(n):
        times, states = gillespie(model, t_max, rng)
        trajectories.append((times, states))

    return trajectories
