import numpy as np

def gillespie(model, t_max=100, rng=None):
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
        a0 = np.sum(a)
        if a0 <= 0:
            break

        tau = rng.exponential(1 / a0)
        r = rng.random() * a0
        mu = np.searchsorted(np.cumsum(a), r)

        x = x + S[:, mu]
        t += tau

        times.append(t)
        states.append(x.copy())

    return np.array(times), np.array(states)
