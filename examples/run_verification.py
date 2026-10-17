#!/usr/bin/env python
"""Generate the verification artefacts for Variance Compass.

Run from the repository root (or anywhere; paths are resolved relative to
this file):

    python examples/run_verification.py

Artefacts (written to docs/results/):
  - ishigami_surface.png   still cover image of the Ishigami response surface
  - convergence.png        Saltelli S_i / T_i estimates vs N with analytic
                           lines and +/- 2 SE bootstrap uncertainty bands
  - convergence_sweep.csv  the full numeric sweep behind the plots
  - convergence.gif        animated convergence progression (one real frame
                           per sample size; 44 frames, recorded data only)
  - run_summary.json       runtime, final estimates vs analytic, errors

Everything is deterministic: seed 20261018 threads one numpy Generator
through the entire sweep, and the anchor run uses seed 424242.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, __version__ as pillow_version

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as matplotlib_version_module  # noqa: E402

from variance_compass import (  # noqa: E402
    ISHIGAMI_A,
    ISHIGAMI_B,
    __version__ as vc_version,
    additive_analytic,
    additive_bounds,
    additive_model,
    analyze_sensitivity,
    convergence_sweep,
    index_errors,
    ishigami,
    ishigami_analytic,
    ishigami_bounds,
)

RESULTS_DIR = REPO_ROOT / "docs" / "results"
SEED = 20261018
N_BOOTSTRAP = 100
# 44 sample sizes from 2**7 to 2**18 -> 44 animation frames (within 30..80).
N_GRID = np.unique(
    np.rint(np.logspace(np.log10(2**7), np.log10(2**18), 44)).astype(np.int64)
)
ANCHOR_COEFFS = (2.0, 1.5, 0.5, -1.0)
ANCHOR_SEED = 424242
ANCHOR_N = 2**18

# Okabe-Ito colourblind-safe palette.
COLORS = ("#0072B2", "#D55E00", "#009E73")
XLABEL = "base sample size $N$ (Saltelli design, total cost $(k+2)N$)"
YLABEL = "index estimate"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.edgecolor": "0.35",
        "axes.labelcolor": "0.15",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def make_cover(path: Path) -> None:
    """Still cover: Ishigami response surface over (x1, x2) at x3 = 0."""
    m = 240
    x1 = np.linspace(-np.pi, np.pi, m)
    x2 = np.linspace(-np.pi, np.pi, m)
    X1, X2 = np.meshgrid(x1, x2)
    pts = np.column_stack(
        [X1.ravel(), X2.ravel(), np.zeros(X1.size)]
    )
    F = ishigami(pts).reshape(m, m)
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        X1, X2, F, cmap="viridis", linewidth=0, antialiased=True,
        rstride=2, cstride=2,
    )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(1.0)
    ax.grid(False)
    ax.view_init(elev=28, azim=-58)
    ax.set_xlabel("$x_1$", labelpad=8)
    ax.set_ylabel("$x_2$", labelpad=8)
    ax.set_zlabel("$f$", labelpad=8)
    ax.set_title(
        f"Ishigami response surface at $x_3 = 0$   ($a$ = {ISHIGAMI_A:g}, $b$ = {ISHIGAMI_B:g})"
    )
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.09, label="$f(x_1, x_2, 0)$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_axes(ax, n_values, S, T, S_se, T_se, S_true, T_true, upto, title):
    """Shared drawing routine for the static plot and every GIF frame."""
    for i in range(3):
        c = COLORS[i]
        ax.fill_between(
            n_values[:upto],
            S[:upto, i] - 2 * S_se[:upto, i],
            S[:upto, i] + 2 * S_se[:upto, i],
            color=c,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            n_values[:upto], S[:upto, i], "o-", color=c, ms=3.6, lw=1.3,
            label=f"$S_{{{i + 1}}}$ estimate",
        )
        ax.plot(
            n_values[:upto], T[:upto, i], "^--", color=c, ms=4.0, lw=1.1,
            mfc="none", label=f"$T_{{{i + 1}}}$ estimate",
        )
        ax.axhline(S_true[i], color=c, ls=":", lw=1.0)
        ax.axhline(T_true[i], color=c, ls=":", lw=1.0, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_xlim(n_values[0] * 0.85, n_values[-1] * 1.18)
    ax.set_ylim(-0.09, 0.88)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.grid(False)
    ax.legend(ncol=3, fontsize=8, loc="upper right", frameon=False)
    ax.annotate(
        "dotted: analytic value",
        xy=(0.02, 0.03),
        xycoords="axes fraction",
        fontsize=8,
        color="0.35",
    )


def make_convergence_plot(path, sweep, S_true, T_true) -> None:
    n_values = sweep.n_values
    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    _draw_axes(
        ax,
        n_values,
        sweep.S_matrix,
        sweep.T_matrix,
        sweep.S_se_matrix,
        sweep.T_se_matrix,
        S_true,
        T_true,
        upto=len(n_values),
        title="Ishigami: Saltelli index convergence vs analytic ground truth",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_convergence_gif(path, sweep, S_true, T_true) -> int:
    """One real frame per sample size; frames replay the recorded sweep."""
    n_values = sweep.n_values
    frames = len(n_values)
    S, T = sweep.S_matrix, sweep.T_matrix
    S_se, T_se = sweep.S_se_matrix, sweep.T_se_matrix
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=126)

    def draw(t):
        ax.clear()
        _draw_axes(
            ax, n_values, S, T, S_se, T_se, S_true, T_true, upto=t + 1,
            title=(
                "Ishigami: Saltelli estimator convergence   "
                f"$N$ = {n_values[t]:,} ($2^{{{np.log2(n_values[t]):.1f}}}$)"
                f"   frame {t + 1}/{frames}"
            ),
        )
        fig.tight_layout()
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=150)
    anim.save(str(path), writer=PillowWriter(fps=7))
    plt.close(fig)
    return frames


def write_csv(path, sweep) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["n", "total_model_evals"]
            + [f"S{i + 1}" for i in range(3)]
            + [f"T{i + 1}" for i in range(3)]
            + [f"S{i + 1}_se" for i in range(3)]
            + [f"T{i + 1}_se" for i in range(3)]
            + ["mean", "variance", "runtime_s"]
        )
        for r in sweep.results:
            writer.writerow(
                [int(r.n), int(r.n * (r.dim + 2))]
                + [f"{v:.6f}" for v in r.S]
                + [f"{v:.6f}" for v in r.T]
                + [f"{v:.6f}" for v in r.S_se]
                + [f"{v:.6f}" for v in r.T_se]
                + [f"{r.mean:.6f}", f"{r.variance:.6f}", f"{r.runtime_s:.3f}"]
            )


def main() -> None:
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    analytic = ishigami_analytic()
    S_true, T_true = np.array(analytic.S), np.array(analytic.T)
    print("Variance Compass verification run")
    print(f"  package version : {vc_version}")
    print(f"  numpy           : {np.__version__}")
    print(f"  analytic V      : {analytic.variance:.6f}")
    print(f"  analytic S      : {np.round(S_true, 6).tolist()}")
    print(f"  analytic T      : {np.round(T_true, 6).tolist()}")

    print(f"[1/5] cover still ...", end=" ", flush=True)
    make_cover(RESULTS_DIR / "ishigami_surface.png")
    print("done (ishigami_surface.png)")

    print(
        f"[2/5] Saltelli sweep: {len(N_GRID)} sample sizes, "
        f"N = {N_GRID[0]}..{N_GRID[-1]}, seed {SEED}, "
        f"bootstrap B = {N_BOOTSTRAP} ...",
        flush=True,
    )
    t0 = time.perf_counter()
    sweep = convergence_sweep(
        ishigami, ishigami_bounds(), N_GRID, seed=SEED, n_bootstrap=N_BOOTSTRAP
    )
    sweep_seconds = time.perf_counter() - t0
    print(f"      done in {sweep_seconds:.1f} s "
          f"({int(np.sum(N_GRID) * 5)} model evaluations)")

    print("[3/5] writing CSV + static convergence plot ...", end=" ", flush=True)
    write_csv(RESULTS_DIR / "convergence_sweep.csv", sweep)
    make_convergence_plot(
        RESULTS_DIR / "convergence.png", sweep, S_true, T_true
    )
    print("done (convergence_sweep.csv, convergence.png)")

    print("[4/5] rendering animated GIF ...", end=" ", flush=True)
    t0 = time.perf_counter()
    gif_frames = make_convergence_gif(
        RESULTS_DIR / "convergence.gif", sweep, S_true, T_true
    )
    gif_seconds = time.perf_counter() - t0
    print(f"done ({gif_frames} frames in {gif_seconds:.1f} s)")

    print("[5/5] additive exact-regression anchor ...", end=" ", flush=True)
    anchor_model = additive_model(ANCHOR_COEFFS)
    anchor_analytic = additive_analytic(ANCHOR_COEFFS)
    anchor = analyze_sensitivity(
        anchor_model,
        ANCHOR_N,
        additive_bounds(len(ANCHOR_COEFFS)),
        seed=ANCHOR_SEED,
        n_bootstrap=N_BOOTSTRAP,
    )
    anchor_errors = index_errors(
        anchor.S, anchor.T, anchor_analytic.S, anchor_analytic.T
    )
    print("done")

    final = sweep.results[-1]
    errors = index_errors(final.S, final.T, S_true, T_true)
    runtime = time.perf_counter() - t_start

    with Image.open(RESULTS_DIR / "convergence.gif") as im:
        gif_frames_verified = int(im.n_frames)
        gif_size = im.size

    below_002 = [
        int(r.n)
        for r in sweep.results
        if np.all(np.abs(r.S - S_true) < 0.02) and np.all(np.abs(r.T - T_true) < 0.02)
    ]

    summary = {
        "generator": "examples/run_verification.py",
        "variance_compass_version": vc_version,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib_version_module.__version__,
        "pillow_version": pillow_version,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_grid": [int(v) for v in N_GRID],
        "runtime_seconds": round(runtime, 3),
        "sweep_seconds": round(sweep_seconds, 3),
        "gif_seconds": round(gif_seconds, 3),
        "ishigami": {
            "a": ISHIGAMI_A,
            "b": ISHIGAMI_B,
            "analytic": analytic.to_dict(),
            "final_n": int(final.n),
            "final_estimates": {"S": final.S.tolist(), "T": final.T.tolist()},
            "final_standard_errors": {
                "S": final.S_se.tolist(),
                "T": final.T_se.tolist(),
            },
            "final_absolute_errors": {
                "S": errors["S_abs"].tolist(),
                "T": errors["T_abs"].tolist(),
            },
            "max_abs_error": {
                "S": errors["max_S_abs"],
                "T": errors["max_T_abs"],
            },
            "variance_estimate": final.variance,
            "variance_relative_error": final.variance / analytic.variance - 1.0,
            "first_n_all_indices_within_0p02": (
                min(below_002) if below_002 else None
            ),
        },
        "additive_anchor": {
            "coefficients": list(ANCHOR_COEFFS),
            "seed": ANCHOR_SEED,
            "n": ANCHOR_N,
            "analytic": {
                "variance": anchor_analytic.variance,
                "S": list(anchor_analytic.S),
                "T": list(anchor_analytic.T),
            },
            "estimates": {"S": anchor.S.tolist(), "T": anchor.T.tolist()},
            "standard_errors": {
                "S": anchor.S_se.tolist(),
                "T": anchor.T_se.tolist(),
            },
            "absolute_errors": {
                "S": anchor_errors["S_abs"].tolist(),
                "T": anchor_errors["T_abs"].tolist(),
            },
            "max_abs_error": {
                "S": anchor_errors["max_S_abs"],
                "T": anchor_errors["max_T_abs"],
            },
            "variance_estimate": anchor.variance,
            "variance_relative_error": (
                anchor.variance / anchor_analytic.variance - 1.0
            ),
        },
        "artifacts": {
            "cover_png": "docs/results/ishigami_surface.png",
            "convergence_png": "docs/results/convergence.png",
            "csv": "docs/results/convergence_sweep.csv",
            "gif": {
                "path": "docs/results/convergence.gif",
                "frames": gif_frames_verified,
                "pixel_size": list(gif_size),
                "bytes": (RESULTS_DIR / "convergence.gif").stat().st_size,
            },
        },
    }
    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print()
    print("VERIFICATION SUMMARY")
    print(f"  final N                       : {final.n:,} (2^{int(np.log2(final.n))})")
    print(f"  S estimates                   : {np.round(final.S, 4).tolist()}")
    print(f"  S analytic                    : {np.round(S_true, 4).tolist()}")
    print(f"  S |estimate - analytic| max   : {errors['max_S_abs']:.5f}")
    print(f"  T estimates                   : {np.round(final.T, 4).tolist()}")
    print(f"  T analytic                    : {np.round(T_true, 4).tolist()}")
    print(f"  T |estimate - analytic| max   : {errors['max_T_abs']:.5f}")
    print(f"  max bootstrap SE (S, T)       : {final.S_se.max():.5f}, {final.T_se.max():.5f}")
    print(f"  variance estimate             : {final.variance:.4f} "
          f"(analytic {analytic.variance:.4f}, rel err "
          f"{final.variance / analytic.variance - 1.0:+.4f})")
    print(f"  first N with all |err| < 0.02 : {min(below_002) if below_002 else 'not reached'}")
    print(f"  anchor max |S - analytic|     : {anchor_errors['max_S_abs']:.5f} "
          f"(analytic S = {np.round(anchor_analytic.S, 4).tolist()})")
    print(f"  anchor max |T - S|            : {np.abs(anchor.T - anchor.S).max():.5f} "
          f"(analytically 0: strictly additive)")
    print(f"  GIF frames (PIL-verified)     : {gif_frames_verified} @ {gif_size[0]}x{gif_size[1]} px")
    print(f"  total runtime                 : {runtime:.1f} s")


if __name__ == "__main__":
    main()
