import numpy as np

<<<<<<< HEAD
def gillespie(model, t_max=100, rng=None):
=======

def gillespie(model, t_max=100.0, rng=None):
    """
    Gillespie Direct Method (SSA)
    """
>>>>>>> 1455e7e (Clean packaging artifacts and add simulators/features)
    if rng is None:
        rng = np.random.default_rng()

    x = model.initial_state().astype(float)
    p = model.parameters()
    S = model.stoichiometry()

    t = 0.0
    times = [t]
    states = [x.copy()]

    while t < t_max:
        a = model.propensities(x, t, p)
<<<<<<< HEAD
        a0 = np.sum(a)
        if a0 <= 0:
            break

        tau = rng.exponential(1 / a0)
        r = rng.random() * a0
        mu = np.searchsorted(np.cumsum(a), r)

        x = x + S[:, mu]
        t += tau

=======
        if np.any(a < 0):
            raise ValueError("Negative propensity encountered.")

        a0 = float(np.sum(a))
        if a0 <= 0:
            break

        tau = rng.exponential(1.0 / a0)
        r = rng.random() * a0
        mu = int(np.searchsorted(np.cumsum(a), r, side="right"))

        x = x + S[:, mu]
        x = np.maximum(x, 0.0)

        t += tau
>>>>>>> 1455e7e (Clean packaging artifacts and add simulators/features)
        times.append(t)
        states.append(x.copy())

    return np.array(times), np.array(states)
