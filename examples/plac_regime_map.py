#!/usr/bin/env python
"""
plac_regime_map.py — Explore pLac dynamical regimes across (IPTG, R_total).

Generates:
  1. Thermodynamic dose–response heatmap (fold-change vs IPTG & repressor count)
  2. Stochastic noise (CV) heatmap across the same parameter space
  3. Example single-cell traces at three induction levels

Run:
    python examples/plac_regime_map.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# If tibs is installed, import directly; otherwise load from local path.
try:
    from tibs.plac import (
        GillespiePLac,
        ThermodynamicPromoter,
        plot_dose_response,
    )
except ImportError:
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from tibs.plac import (
        GillespiePLac,
        ThermodynamicPromoter,
        plot_dose_response,
    )


def main():
    # ── 1. Fold-change heatmap ────────────────────────────────────────────
    iptg_vals = np.logspace(-8, -2, 60)
    R_vals = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])

    fc_map = np.zeros((len(R_vals), len(iptg_vals)))
    for i, R in enumerate(R_vals):
        tp = ThermodynamicPromoter(R_total=R)
        fc_map[i, :] = tp.fold_change(iptg_vals)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.pcolormesh(
        iptg_vals * 1e6, np.arange(len(R_vals)),
        fc_map, cmap="viridis", shading="auto",
    )
    ax1.set_xscale("log")
    ax1.set_yticks(np.arange(len(R_vals)))
    ax1.set_yticklabels(R_vals)
    ax1.set_xlabel("[IPTG] (µM)")
    ax1.set_ylabel("LacI tetramers / cell")
    ax1.set_title("fold-change (stat-mech model)")
    fig.colorbar(im, ax=ax1, label="fold-change")

    # ── 2. Dose–response curves ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    plot_dose_response(repressor_counts=[1, 10, 50, 200], ax=ax2)

    # ── 3. Stochastic traces at three IPTG levels ────────────────────────
    iptg_levels = {"low (0.1 µM)": 1e-7, "mid (10 µM)": 1e-5, "high (1 mM)": 1e-3}

    for idx, (label, iptg) in enumerate(iptg_levels.items()):
        ax = fig.add_subplot(gs[1, 0]) if idx == 0 else fig.add_subplot(gs[1, 1]) if idx == 2 else None
        if idx == 1:
            # middle panel spans center — just use gs[1,0] and gs[1,1]
            continue

        g = GillespiePLac(IPTG=iptg, seed=42)
        traces = g.run_ensemble(n_cells=30, t_end=200.0)
        for t, X in traces[:15]:
            ax.step(t, X[:, 1], where="post", lw=0.5, alpha=0.5)
        ax.set(xlabel="time (min)", ylabel="protein / cell", title=f"SSA — {label}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("pLac Promoter — Regime Exploration", fontsize=15, y=1.01)
    fig.savefig("plac_regime_map.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plac_regime_map.png")


if __name__ == "__main__":
    main()
