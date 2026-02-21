import numpy as np


def auc(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.trapz(y, t))


def peak(y):
    y = np.asarray(y, dtype=float)
    return float(np.max(y))


def time_to_halfmax(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float("nan")
    hm = 0.5 * np.max(y)
    idx = np.where(y >= hm)[0]
    return float(t[idx[0]]) if idx.size else float("nan")
