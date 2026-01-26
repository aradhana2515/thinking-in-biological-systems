import numpy as np

def auc(t, y):
    return np.trapz(y, t)

def peak(y):
    return np.max(y)

def time_to_halfmax(t, y):
    hm = 0.5 * np.max(y)
    idx = np.where(y >= hm)[0]
    return t[idx[0]] if len(idx) > 0 else np.nan
