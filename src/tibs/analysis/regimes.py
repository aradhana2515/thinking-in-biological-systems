from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RegimeResult:
    label: str                 # "steady" | "oscillatory" | "blowup" | "other"
    score: float               # confidence-ish (bigger = more of that label)
    details: dict              # debug info (var, turns, etc.)

def _dominant_dim(y: np.ndarray) -> int:
    # choose dimension with largest variance after transient
    v = np.var(y, axis=0)
    return int(np.argmax(v)) if y.shape[1] > 1 else 0

def classify_timeseries(
    x: np.ndarray,
    *,
    transient_frac: float = 0.3,
    blowup_thresh: float = 1e6,
    steady_var_thresh: float = 1e-4,
    min_turns: int = 12,
    min_rel_amp: float = 0.05,
) -> RegimeResult:
    """
    Classify behavior from a timeseries x(t) of shape (T, D).

    Heuristics:
      - blowup: NaN/inf or very large magnitude
      - steady: low variance after transient
      - oscillatory: lots of turning points + nontrivial amplitude after transient
      - other: anything else
    """
    x = np.asarray(x, dtype=float)

    # --- blowup checks ---
    if x.size == 0:
        return RegimeResult("other", 0.0, {"reason": "empty"})
    if not np.isfinite(x).all():
        return RegimeResult("blowup", 1.0, {"reason": "nonfinite"})
    max_abs = float(np.max(np.abs(x)))
    if max_abs > blowup_thresh:
        return RegimeResult("blowup", max_abs / blowup_thresh, {"reason": "threshold", "max_abs": max_abs})

    # --- drop transient ---
    T = x.shape[0]
    start = int(np.clip(np.floor(T * transient_frac), 0, max(T - 2, 0)))
    y = x[start:]  # (T', D)

    # --- steady check (variance-based) ---
    var_per_dim = np.var(y, axis=0)
    mean_var = float(np.mean(var_per_dim))
    if mean_var < steady_var_thresh:
        # smaller variance -> more confident steady
        score = float(steady_var_thresh / max(mean_var, 1e-12))
        return RegimeResult("steady", score, {"mean_var": mean_var, "var_per_dim": var_per_dim.tolist(), "start": start})

    # --- oscillation check (turning points + amplitude) ---
    j = _dominant_dim(y)
    s = y[:, j]
    ds = np.diff(s)

    # Turning points: sign changes in derivative
    sign = np.sign(ds)
    turns = int(np.sum((sign[1:] * sign[:-1]) < 0))

    amp = float(np.max(s) - np.min(s))
    scale = float(np.mean(np.abs(s)) + 1e-12)
    rel_amp = amp / scale

    # Require both: enough turns + enough amplitude
    if turns >= min_turns and rel_amp >= min_rel_amp:
        score = float(turns) * rel_amp
        return RegimeResult(
            "oscillatory",
            score,
            {"turns": turns, "rel_amp": rel_amp, "amp": amp, "dim": j, "mean_var": mean_var, "start": start},
        )

    return RegimeResult(
        "other",
        float(turns) * rel_amp,
        {"turns": turns, "rel_amp": rel_amp, "amp": amp, "dim": j, "mean_var": mean_var, "start": start},
    )
