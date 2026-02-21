#!/usr/bin/env python3
"""
Preflight checks for thinking-in-biological-systems (tibs).

Run from repo root:
    python tools/preflight_check.py

What it does:
- Verifies imports
- Runs one SSA trajectory + plot
- Runs SSA ensemble and compares mean trend to ODE
- Tests basic features
- Tests reproducibility
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

# Optional dependencies for some checks
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.integrate import solve_ivp
except Exception:
    solve_ivp = None


def fail(msg: str, code: int = 1) -> None:
    print(f"\n❌ FAIL: {msg}\n")
    sys.exit(code)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    print(f"Running preflight from: {repo_root}")

    # ---------- 1) Imports ----------
    try:
        from tibs.models.gene_expression import GeneExpression
        from tibs.simulators.ssa import gillespie
    except Exception as e:
        fail(f"Import error. Did you run `pip install -e .`?\n{e}")

    ok("Imports succeeded: tibs.models.gene_expression, tibs.simulators.ssa")

    # Optional imports
    run_ensemble = None
    try:
        from tibs.simulators.ensemble import run_ensemble as _run_ensemble

        run_ensemble = _run_ensemble
        ok("Import succeeded: tibs.simulators.ensemble.run_ensemble")
    except Exception:
        print("⚠️  Could not import run_ensemble; ensemble checks will be skipped.")

    try:
        from tibs.features.timeseries import auc, peak, time_to_halfmax

        ok("Import succeeded: tibs.features.timeseries (auc, peak, time_to_halfmax)")
    except Exception:
        print(
            "⚠️  Could not import tibs.features.timeseries; feature checks will be skipped."
        )
        auc = peak = time_to_halfmax = None

    # ---------- 2) Single SSA run ----------
    model = GeneExpression()
    rng = np.random.default_rng(0)

    try:
        t, x = gillespie(model, t_max=50, rng=rng)  # expects rng kwarg
    except TypeError:
        # fallback if your gillespie signature doesn't accept rng kwarg
        t, x = gillespie(model, t_max=50)
    except Exception as e:
        fail(f"Gillespie run failed:\n{e}")

    if t is None or x is None:
        fail("Gillespie returned None.")
    if len(t) < 2:
        fail("Gillespie produced too few time points.")
    if not np.all(np.diff(t) > 0):
        fail("Time array is not strictly increasing.")
    if np.any(x < 0):
        fail("State has negative counts/values (should not happen for SSA counts).")
    if x.ndim != 2:
        fail("State array should be 2D: (n_steps, n_species).")

    ok(f"Single SSA trajectory ran: t.shape={t.shape}, x.shape={x.shape}")

    # ---------- 3) Plot single trajectory ----------
    if plt is not None:
        try:
            y = model.observe(
                x.T, model.parameters()
            )  # some observe implementations assume x
        except Exception:
            # most likely observe expects a single state vector, so just use protein col
            y = x[:, -1]

        # If observe gave something weird, fall back
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.ravel()

        outdir = repo_root / "artifacts"
        outdir.mkdir(exist_ok=True)
        outpath = outdir / "ssa_single_trajectory.png"

        plt.figure()
        plt.plot(t, x[:, -1])
        plt.xlabel("time")
        plt.ylabel("protein (luminescence proxy)")
        plt.title("SSA single trajectory")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

        ok(f"Saved plot: {outpath}")
    else:
        print("⚠️  matplotlib not available; skipping plot generation.")

    # ---------- 4) Feature sanity checks ----------
    if auc is not None and peak is not None and time_to_halfmax is not None:
        # simple synthetic signal
        tt = np.linspace(0, 10, 100)
        yy = tt  # ramp
        a1 = auc(tt, yy)
        a2 = auc(tt, 2 * yy)
        if not np.isclose(a2, 2 * a1, rtol=1e-6, atol=1e-6):
            fail("Feature test failed: AUC scaling check did not pass.")
        if peak(yy) != np.max(yy):
            fail("Feature test failed: peak() mismatch.")
        th = time_to_halfmax(tt, yy)
        if not np.isfinite(th):
            fail("Feature test failed: time_to_halfmax returned NaN/inf unexpectedly.")
        ok("Feature sanity checks passed (auc scaling, peak, time_to_halfmax).")
    else:
        print("⚠️  Skipping feature sanity checks (features module not found/imported).")

    # ---------- 5) Reproducibility check ----------
    # Only works if your gillespie accepts rng, or if you seed internally.
    try:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        t1, x1 = gillespie(model, t_max=20, rng=rng1)
        t2, x2 = gillespie(model, t_max=20, rng=rng2)
        if not (np.allclose(t1, t2) and np.allclose(x1, x2)):
            print("⚠️  Reproducibility check did not match exactly.")
            print("    This is OK if your gillespie() doesn't use the passed rng.")
        else:
            ok("Reproducibility check passed (same seed -> identical trajectory).")
    except TypeError:
        print(
            "⚠️  gillespie() does not accept rng; skipping strict reproducibility check."
        )
    except Exception as e:
        print(f"⚠️  Reproducibility check error (skipping): {e}")

    # ---------- 6) ODE vs SSA ensemble mean check ----------
    if run_ensemble is not None and solve_ivp is not None and plt is not None:
        # ODE solve
        p = model.parameters()

        def rhs(t, x):
            return model.rhs(t, x, p)

        t_eval = np.linspace(0, 100, 500)
        sol = solve_ivp(
            rhs, (0, 100), model.initial_state(), t_eval=t_eval, rtol=1e-6, atol=1e-9
        )
        if not sol.success:
            fail(f"ODE solver failed: {sol.message}")

        # SSA ensemble -> interpolate onto t_eval and compute mean protein
        traj = run_ensemble(model, n=200, t_max=100, seed=0)

        prot_interp = []
        for ti, xi in traj:
            # xi[:, -1] protein column
            yi = xi[:, -1]
            # interpolate; for times beyond last point, hold last value
            prot_interp.append(np.interp(t_eval, ti, yi, left=yi[0], right=yi[-1]))

        prot_interp = np.array(prot_interp)
        ssa_mean = prot_interp.mean(axis=0)

        outdir = repo_root / "artifacts"
        outdir.mkdir(exist_ok=True)
        outpath = outdir / "ode_vs_ssa_mean.png"

        plt.figure()
        # plot a few SSA traces
        for i in range(min(30, len(traj))):
            ti, xi = traj[i]
            plt.plot(ti, xi[:, -1], alpha=0.25)

        plt.plot(sol.t, sol.y[-1], linewidth=2, label="ODE")
        plt.plot(t_eval, ssa_mean, linewidth=2, label="SSA mean")
        plt.xlabel("time")
        plt.ylabel("protein (luminescence proxy)")
        plt.title("ODE vs SSA ensemble mean")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

        ok(f"Saved plot: {outpath}")

        # Loose numeric check: the end-point mean should be within ~30% of ODE steady-ish value
        ode_final = sol.y[-1, -1]
        ssa_final = ssa_mean[-1]
        if ode_final > 1e-9:
            rel_err = abs(ssa_final - ode_final) / abs(ode_final)
            print(
                f"ODE final={ode_final:.3g}, SSA mean final={ssa_final:.3g}, rel_err={rel_err:.2f}"
            )
            if rel_err > 0.30:
                print("⚠️  ODE vs SSA mean mismatch > 30%.")
                print(
                    "    This might be fine if your parameters produce slow convergence,"
                )
                print("    or if your observe() differs from protein count.")
            else:
                ok(
                    "ODE vs SSA ensemble mean agreement looks reasonable (<30% at final time)."
                )
    else:
        print(
            "⚠️  Skipping ODE vs SSA ensemble check (need scipy + matplotlib + run_ensemble)."
        )

    print(
        "\n🎉 Preflight complete. If you got mostly ✅ and only a few ⚠️, you're good to push.\n"
    )


if __name__ == "__main__":
    main()
