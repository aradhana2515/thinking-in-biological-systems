"""
pLac promoter models — thermodynamic, ODE, and stochastic.

Three modeling approaches for the IPTG-inducible lac promoter:

1. ThermodynamicPromoter  — statistical mechanics (partition function)
2. PLac                   — ODE model subclassing ODEModel
3. GillespiePLac          — exact stochastic simulation (SSA)

Typical usage
-------------
    from tibs.plac import ThermodynamicPromoter, PLac, GillespiePLac

    # --- stat-mech dose–response ---
    tp = ThermodynamicPromoter()
    iptg, fc = tp.dose_response()

    # --- ODE dynamics ---
    m = PLac()
    res = m.simulate(t_span=(0, 120), dt=0.05)
    m.plot(res)

    # --- single-cell stochastic traces ---
    g = GillespiePLac()
    t, X = g.run(t_end=200.0)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Attempt to import the repo's base class; fall back to a minimal shim so
# the module also works stand-alone.
# ---------------------------------------------------------------------------
try:
    from tibs.models import ODEModel
except ImportError:

    class ODEModel:
        """Minimal shim when tibs is not installed."""

        def initial_state(self) -> np.ndarray:
            raise NotImplementedError

        def parameters(self) -> dict:
            raise NotImplementedError

        def rhs(self, t: float, x: np.ndarray, params: dict) -> np.ndarray:
            raise NotImplementedError

        def simulate(self, t_span=(0, 100), dt=0.05):
            t0, tf = t_span
            t = np.arange(t0, tf, dt)
            x = np.zeros((len(t), len(self.initial_state())))
            x[0] = self.initial_state()
            p = self.parameters()
            for i in range(len(t) - 1):
                h = dt
                k1 = np.asarray(self.rhs(t[i], x[i], p), dtype=float)
                k2 = np.asarray(self.rhs(t[i] + h / 2, x[i] + h / 2 * k1, p), dtype=float)
                k3 = np.asarray(self.rhs(t[i] + h / 2, x[i] + h / 2 * k2, p), dtype=float)
                k4 = np.asarray(self.rhs(t[i] + h, x[i] + h * k3, p), dtype=float)
                x[i + 1] = x[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return _SimResult(t=t, x=x)

        def plot(self, res, **kw):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            labels = getattr(self, "state_labels", None) or [
                f"x{i}" for i in range(res.x.shape[1])
            ]
            for i, lbl in enumerate(labels):
                ax.plot(res.t, res.x[:, i], label=lbl, linewidth=1.5)
            ax.set(xlabel="time (min)", ylabel="molecules / cell")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            return fig


@dataclass
class _SimResult:
    t: np.ndarray
    x: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# 1. THERMODYNAMIC (STATISTICAL MECHANICS) MODEL
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ThermodynamicPromoter:
    """
    Equilibrium statistical-mechanics model of the pLac promoter.

    Uses the "simple repression" partition-function formulation from
    Garcia & Phillips (2011) and Razo-Mejia et al. (2018).

    The promoter has three relevant states:

    1. **Empty** — neither RNAP nor repressor bound (weight = 1, reference)
    2. **RNAP bound** — transcriptionally active
       weight = (P / N_NS) · exp(−βΔε_P)
    3. **Repressor bound** — transcription blocked
       weight = (R_A / N_NS) · exp(−βΔε_R)

    The fold-change in gene expression (relative to constitutive) is:

        fc = 1 / (1 + (R_A / N_NS) · exp(−Δε_R))

    where R_A is the number of active (non-IPTG-bound) repressors:

        R_A = R_total · (1 / (1 + [IPTG] / K_d))^n_iptg

    Parameters match measured values from Garcia & Phillips (2011) and
    Razo-Mejia et al. (2018).
    """

    # --- copy numbers ---
    R_total: int = 10          # LacI tetramers per cell
    P_rnap: int = 1000         # RNAP molecules per cell
    N_NS: float = 4.6e6        # non-specific binding sites on genome

    # --- binding energies (units of kBT, negative = favourable) ---
    dE_rnap: float = -10.0     # RNAP → promoter
    dE_R: float = -15.3        # LacI → operator (O1, strongest natural operator)

    # --- IPTG–LacI interaction ---
    Kd_IPTG: float = 0.53e-6   # dissociation constant (M)
    n_iptg: int = 2            # inducer binding sites per functional dimer

    # -----------------------------------------------------------------
    def active_repressors(self, IPTG: float | np.ndarray) -> float | np.ndarray:
        """Number of active (DNA-competent) repressors at given [IPTG]."""
        return self.R_total * (1.0 / (1.0 + np.asarray(IPTG) / self.Kd_IPTG)) ** self.n_iptg

    def fold_change(self, IPTG: float | np.ndarray) -> float | np.ndarray:
        """
        Expression fold-change relative to constitutive (R=0) promoter.

        fc = 1 / (1 + (R_A / N_NS) · exp(−Δε_R))
        """
        R_A = self.active_repressors(IPTG)
        return 1.0 / (1.0 + (R_A / self.N_NS) * np.exp(-self.dE_R))

    def p_bound(self, IPTG: float | np.ndarray) -> float | np.ndarray:
        """
        Absolute probability that RNAP occupies the promoter.

        p_bound = p_constitutive · fold_change
        """
        return self._p_constitutive() * self.fold_change(IPTG)

    def _p_constitutive(self) -> float:
        """RNAP occupancy with no repressor present."""
        w_rnap = (self.P_rnap / self.N_NS) * np.exp(-self.dE_rnap)
        return w_rnap / (1.0 + w_rnap)

    def dose_response(
        self,
        iptg_range: tuple[float, float] = (1e-9, 1e-2),
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (iptg_concentrations, fold_change) arrays."""
        iptg = np.logspace(np.log10(iptg_range[0]), np.log10(iptg_range[1]), n_points)
        return iptg, self.fold_change(iptg)

    def state_probabilities(self, IPTG: float) -> dict[str, float]:
        """Return a dict of microstate occupancy probabilities."""
        R_A = self.active_repressors(IPTG)
        weights = {
            "empty": 1.0,
            "RNAP_bound": (self.P_rnap / self.N_NS) * np.exp(-self.dE_rnap),
            "LacI_bound": (R_A / self.N_NS) * np.exp(-self.dE_R),
        }
        Z = sum(weights.values())
        return {k: v / Z for k, v in weights.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 2. ODE MODEL  (subclasses ODEModel from the repo)
# ═══════════════════════════════════════════════════════════════════════════


class PLac(ODEModel):
    """
    Deterministic ODE model of pLac-driven gene expression.

    State vector: [mRNA, Protein]

    Transcription rate is set by the thermodynamic fold-change:

        dm/dt = alpha_m · fc(IPTG) − gamma_m · m
        dp/dt = beta_p · m         − gamma_p · p

    Parameters
    ----------
    IPTG : float
        Extracellular IPTG concentration (M).  Default 1 mM (fully induced).
    """

    state_labels = ["mRNA", "protein"]

    def __init__(self, IPTG: float = 1e-3, **overrides):
        self.IPTG = IPTG
        self._overrides = overrides

    def initial_state(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def parameters(self) -> dict:
        tp = ThermodynamicPromoter()
        fc = float(tp.fold_change(self.IPTG))

        defaults = dict(
            alpha_m=0.5,       # max transcription rate (mRNA / min)
            fc=fc,             # fold-change from stat-mech model [0, 1]
            gamma_m=0.1,       # mRNA degradation rate (1/min), half-life ~7 min
            beta_p=0.04,       # translation rate (protein / mRNA / min)
            gamma_p=0.005,     # protein dilution+degradation (1/min)
        )
        defaults.update(self._overrides)
        return defaults

    def rhs(self, t: float, x: np.ndarray, params: dict) -> np.ndarray:
        m, p = x
        dm = params["alpha_m"] * params["fc"] - params["gamma_m"] * m
        dp = params["beta_p"] * m - params["gamma_p"] * p
        return np.array([dm, dp])


# ═══════════════════════════════════════════════════════════════════════════
# 3. GILLESPIE STOCHASTIC SIMULATION (SSA)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GillespiePLac:
    """
    Exact stochastic simulation of pLac gene expression via the
    Gillespie algorithm (direct method).

    Species
    -------
    0: mRNA
    1: protein

    Reactions
    ---------
    0: ∅ → mRNA          (transcription, rate = k_tx · fc)
    1: mRNA → ∅          (mRNA decay,    rate = gamma_m · mRNA)
    2: mRNA → mRNA + P   (translation,   rate = k_tl · mRNA)
    3: P → ∅             (protein decay,  rate = gamma_p · P)

    The transcription rate is scaled by the thermodynamic fold-change,
    so the deterministic steady state matches the ODE model.
    """

    # IPTG concentration (M)
    IPTG: float = 1e-3

    # kinetic rates
    k_tx: float = 0.5          # max transcription rate (events / min)
    gamma_m: float = 0.1       # mRNA decay (1 / min)
    k_tl: float = 0.04         # translation per mRNA (events / min)
    gamma_p: float = 0.005     # protein decay + dilution (1 / min)

    # repressor parameters (forwarded to ThermodynamicPromoter)
    R_total: int = 10
    Kd_IPTG: float = 0.53e-6

    # random seed (None = non-deterministic)
    seed: int | None = None

    def _fold_change(self) -> float:
        tp = ThermodynamicPromoter(R_total=self.R_total, Kd_IPTG=self.Kd_IPTG)
        return float(tp.fold_change(self.IPTG))

    def propensities(self, state: np.ndarray, fc: float) -> np.ndarray:
        mRNA, protein = state
        return np.array([
            self.k_tx * fc,             # transcription
            self.gamma_m * mRNA,        # mRNA decay
            self.k_tl * mRNA,           # translation
            self.gamma_p * protein,     # protein decay
        ])

    # stoichiometry matrix: rows = species, cols = reactions
    S = np.array([
        [+1, -1,  0,  0],   # mRNA
        [ 0,  0, +1, -1],   # protein
    ], dtype=int)

    def run(
        self,
        t_end: float = 200.0,
        x0: np.ndarray | None = None,
        max_steps: int = 5_000_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one SSA trajectory.

        Returns
        -------
        t : 1-D array of event times
        X : 2-D array (n_events, 2) of [mRNA, protein] counts
        """
        rng = np.random.default_rng(self.seed)
        fc = self._fold_change()

        state = np.array([0, 0]) if x0 is None else np.asarray(x0, dtype=int)
        t_curr = 0.0

        t_history = [t_curr]
        x_history = [state.copy()]

        for _ in range(max_steps):
            a = self.propensities(state, fc)
            a0 = a.sum()
            if a0 == 0:
                break

            # time to next reaction (exponential)
            tau = rng.exponential(1.0 / a0)
            t_curr += tau
            if t_curr > t_end:
                break

            # which reaction fires?
            r = rng.random() * a0
            j = 0
            cumsum = a[0]
            while cumsum < r:
                j += 1
                cumsum += a[j]

            # update state
            state = state + self.S[:, j]
            state = np.maximum(state, 0)  # safety clamp

            t_history.append(t_curr)
            x_history.append(state.copy())

        return np.array(t_history), np.array(x_history)

    def run_ensemble(
        self,
        n_cells: int = 200,
        t_end: float = 200.0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run *n_cells* independent SSA trajectories."""
        results = []
        for i in range(n_cells):
            g = GillespiePLac(
                IPTG=self.IPTG,
                k_tx=self.k_tx,
                gamma_m=self.gamma_m,
                k_tl=self.k_tl,
                gamma_p=self.gamma_p,
                R_total=self.R_total,
                Kd_IPTG=self.Kd_IPTG,
                seed=(self.seed + i) if self.seed is not None else None,
            )
            results.append(g.run(t_end=t_end))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONVENIENCE PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_dose_response(
    tp: ThermodynamicPromoter | None = None,
    repressor_counts: list[int] | None = None,
    ax=None,
):
    """
    Plot IPTG dose–response curves for one or more repressor copy numbers.
    """
    import matplotlib.pyplot as plt

    if tp is None:
        tp = ThermodynamicPromoter()
    if repressor_counts is None:
        repressor_counts = [1, 5, 10, 30, 100]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, len(repressor_counts)))

    for R, c in zip(repressor_counts, colors):
        tp_r = ThermodynamicPromoter(R_total=R)
        iptg, fc = tp_r.dose_response()
        ax.semilogx(iptg * 1e6, fc, color=c, lw=2, label=f"R = {R}")

    ax.set(
        xlabel="[IPTG] (µM)",
        ylabel="fold-change",
        title="pLac thermodynamic model — dose–response",
    )
    ax.legend(title="LacI / cell", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_stochastic_traces(
    traces: list[tuple[np.ndarray, np.ndarray]],
    species: int = 1,
    n_show: int = 20,
    ax=None,
):
    """
    Overlay single-cell stochastic trajectories for a given species.

    Parameters
    ----------
    traces : list of (t, X) from GillespiePLac.run_ensemble()
    species : 0 = mRNA, 1 = protein
    n_show : max traces to draw
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    label = ["mRNA", "protein"][species]
    for i, (t, X) in enumerate(traces[:n_show]):
        ax.step(t, X[:, species], where="post", lw=0.6, alpha=0.5)

    ax.set(
        xlabel="time (min)",
        ylabel=f"{label} molecules / cell",
        title=f"Gillespie SSA — single-cell {label} traces",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_noise_vs_induction(
    iptg_values: np.ndarray | None = None,
    n_cells: int = 300,
    t_end: float = 300.0,
    ax=None,
):
    """
    Show how expression noise (CV = σ/μ) changes with IPTG concentration.
    Runs an ensemble at each IPTG value and samples final protein count.
    """
    import matplotlib.pyplot as plt

    if iptg_values is None:
        iptg_values = np.logspace(-7, -2, 8)

    means, cvs = [], []
    for iptg in iptg_values:
        g = GillespiePLac(IPTG=iptg, seed=42)
        ensemble = g.run_ensemble(n_cells=n_cells, t_end=t_end)
        finals = np.array([X[-1, 1] for (_, X) in ensemble], dtype=float)
        mu = finals.mean()
        sigma = finals.std()
        means.append(mu)
        cvs.append(sigma / mu if mu > 0 else np.nan)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig = ax.figure
        axes = [ax, ax.twinx()]

    axes[0].semilogx(iptg_values * 1e6, means, "o-", color="#2563eb", lw=2)
    axes[0].set(xlabel="[IPTG] (µM)", ylabel="mean protein / cell",
                title="mean expression vs IPTG")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(iptg_values * 1e6, cvs, "s-", color="#dc2626", lw=2)
    axes[1].set(xlabel="[IPTG] (µM)", ylabel="CV (σ / μ)",
                title="expression noise vs IPTG")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
