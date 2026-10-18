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
        X1, X2, F, cmap="viridis", linewidth=0, edgecolor="none", antialiased=False,
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


def _draw_pair(axes, sweep, S_true, T_true, upto):
    low = min(np.min(sweep.S_matrix-2*sweep.S_se_matrix), np.min(sweep.T_matrix-2*sweep.T_se_matrix), 0)
    high = max(np.max(sweep.S_matrix+2*sweep.S_se_matrix), np.max(sweep.T_matrix+2*sweep.T_se_matrix), 1)
    for ax, values, se, reference, symbol, title in (
        (axes[0],sweep.S_matrix,sweep.S_se_matrix,S_true,"S","First-order effects"),
        (axes[1],sweep.T_matrix,sweep.T_se_matrix,T_true,"T","Total effects, including interactions")):
        ax.clear()
        for i,color in enumerate(COLORS):
            ax.errorbar(sweep.n_values[:upto],values[:upto,i],yerr=2*se[:upto,i],
                        fmt="o-",color=color,ms=3,lw=1,elinewidth=.6,capsize=1.5,label=f"{symbol}{i+1}: input x{i+1}")
            ax.axhline(reference[i],color=color,ls=":",lw=1)
        ax.set(xscale="log",xlim=(sweep.n_values[0]*.8,sweep.n_values[-1]*1.25),
               ylim=(low-.05,high+.05),xlabel="Base sample size N (log scale)",ylabel="Index estimate",title=title)
        ax.legend(loc="upper right",fontsize=8,frameon=False)
        ax.grid(False)


def make_convergence_plot(path,sweep,S_true,T_true):
    fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    fig.subplots_adjust(left=.07,right=.97,bottom=.18,top=.81,wspace=.23)
    _draw_pair(axes,sweep,S_true,T_true,len(sweep.results))
    fig.suptitle("Ishigami sensitivity convergence · centered covariance estimates",fontsize=13)
    fig.text(.5,.035,"Dotted: analytic reference · whiskers: ±2 bootstrap SE (not exact 95% coverage)",ha="center",fontsize=9)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def make_convergence_gif(path,sweep,S_true,T_true):
    fig,axes=plt.subplots(1,2,figsize=(11,6),dpi=100)
    fig.subplots_adjust(left=.07,right=.97,bottom=.17,top=.80,wspace=.23)
    frames=len(sweep.results)
    title=fig.suptitle("",fontsize=12)
    fig.text(.5,.035,"Recorded independent designs · dotted: analytic · whiskers: ±2 bootstrap SE",ha="center",fontsize=9)
    writer=PillowWriter(fps=7)
    with writer.saving(fig,path,dpi=100):
        for i,result in enumerate(sweep.results):
            _draw_pair(axes,sweep,S_true,T_true,i+1)
            title.set_text(f"Ishigami sensitivity · N = {result.n:,} · {5*result.n:,} model evaluations\nFrame {i+1}/{frames} · 100 row-bootstrap replicates")
            writer.grab_frame()
    plt.close(fig)
    return frames


def affine_benchmark():
    from variance_compass.estimators import saltelli_indices
    from variance_compass.sampling import saltelli_matrices
    a,b,crossed=saltelli_matrices(8192,2,np.random.default_rng(10))
    model=additive_model((1.,1.))
    fa,fb=model(2*a-1),model(2*b-1)
    fab=np.vstack([model(2*x-1) for x in crossed])
    base_s,base_t,_,_=saltelli_indices(fa,fb,fab)
    rows=[]
    for offset in (0.,1.,100.,1000.,1e6,1e8):
        aa,bb,cc=fa+offset,fb+offset,fab+offset
        stable_s,stable_t,_,variance=saltelli_indices(aa,bb,cc)
        f0=.5*(aa.mean()+bb.mean())
        old_v=.5*(np.mean(aa*aa)+np.mean(bb*bb))-f0*f0
        if old_v>0:
            old_s=(cc@bb/aa.size-f0*f0)/old_v
            old_t=1-(cc@aa/aa.size-f0*f0)/old_v
        else:
            old_s=old_t=np.full(2,np.nan)
        rows.append({"offset":offset,"centered_S1":stable_s[0],"centered_S2":stable_s[1],
                     "centered_T1":stable_t[0],"centered_T2":stable_t[1],"centered_variance":variance,
                     "legacy_S1":old_s[0],"legacy_S2":old_s[1],"legacy_T1":old_t[0],"legacy_T2":old_t[1],
                     "legacy_variance":old_v,"max_centered_index_drift":float(max(np.max(abs(stable_s-base_s)),np.max(abs(stable_t-base_t))))})
    with (RESULTS_DIR/"affine_invariance.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    offsets=np.array([r["offset"] for r in rows])
    axes[0].plot(offsets,[r["legacy_S1"] for r in rows],"o--",color="#B06024",label="Prior raw moments")
    axes[0].plot(offsets,[r["centered_S1"] for r in rows],"o-",color=COLORS[0],label="Common-centered")
    axes[0].axhline(.5,color="black",ls=":",lw=1,label="Analytic S1 = 0.5")
    axes[0].set(xscale="symlog",yscale="symlog",xlabel="Added output constant",ylabel="Estimated first-order index S1",title="Same 8,192-point design, different output zero")
    axes[0].legend(fontsize=8,frameon=False,loc="lower left")
    prior_base=np.array([rows[0]["legacy_S1"],rows[0]["legacy_S2"],rows[0]["legacy_T1"],rows[0]["legacy_T2"]])
    old_drift=[max(abs(np.array([r["legacy_S1"],r["legacy_S2"],r["legacy_T1"],r["legacy_T2"]])-prior_base)) for r in rows]
    axes[1].plot(offsets,np.maximum(old_drift,1e-16),"o--",color="#B06024",label="Prior raw moments")
    axes[1].plot(offsets,np.maximum([r["max_centered_index_drift"] for r in rows],1e-16),"o-",color=COLORS[0],label="Common-centered")
    axes[1].set(xscale="symlog",yscale="log",xlabel="Added output constant",ylabel="Maximum index change from offset zero",title="Translation invariance to data-rounding precision")
    axes[1].legend(fontsize=8,frameon=False,loc="upper left")
    for ax in axes:
        ax.set_xlim(-0.1,1.5e8)
        ax.set_xticks([0,100,1e4,1e6,1e8],["0","10²","10⁴","10⁶","10⁸"])
    axes[0].set_ylim(-1e6,1)
    fig.tight_layout(rect=(0,.04,1,.92))
    fig.suptitle("An output offset must not change sensitivity · additive model, seed 10",fontsize=12)
    fig.text(.5,.02,"Raw values saved in CSV; right-panel display floor 1e−16. No index clipping.",ha="center",fontsize=9)
    fig.savefig(RESULTS_DIR/"affine_invariance.png",dpi=150)
    plt.close(fig)
    return {"model":"x1+x2, independent U(-1,1)","n":8192,"seed":10,
            "max_centered_index_drift":max(r["max_centered_index_drift"] for r in rows),
            "offset_1000":{k:float(v) for k,v in rows[3].items()},
            "offset_1e8":{k:float(v) for k,v in rows[-1].items()}}


def write_csv(path, sweep) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["n", "total_model_evals"]
            + [f"S{i + 1}" for i in range(3)]
            + [f"T{i + 1}" for i in range(3)]
            + [f"S{i + 1}_se" for i in range(3)]
            + [f"T{i + 1}_se" for i in range(3)]
            + [f"{family}{i+1}_ci_{end}" for family in ("S","T") for i in range(3) for end in ("low","high")]
            + ["mean", "variance", "runtime_s"]
        )
        for r in sweep.results:
            writer.writerow(
                [int(r.n), int(r.n * (r.dim + 2))]
                + [f"{v:.12g}" for v in r.S]
                + [f"{v:.12g}" for v in r.T]
                + [f"{v:.12g}" for v in r.S_se]
                + [f"{v:.12g}" for v in r.T_se]
                + [f"{x:.12g}" for ci in (r.S_ci,r.T_ci) for pair in ci for x in pair]
                + [f"{r.mean:.12g}", f"{r.variance:.12g}", f"{r.runtime_s:.3f}"]
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

    affine = affine_benchmark()
    api_result = analyze_sensitivity(ishigami,2**16,ishigami_bounds(),seed=SEED,n_bootstrap=100)
    test_result = analyze_sensitivity(ishigami,2**17,ishigami_bounds(),seed=SEED,n_bootstrap=200)
    test_anchor = analyze_sensitivity(additive_model(ANCHOR_COEFFS),2**17,additive_bounds(4),seed=ANCHOR_SEED,n_bootstrap=100)
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
        "affine_invariance":affine,
        "api_example":api_result.to_dict(),
        "test_fixture_ishigami":test_result.to_dict(),
        "test_fixture_additive":test_anchor.to_dict(),
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
            "first_n_all_later_indices_within_0p02": next((int(sweep.n_values[i]) for i in range(len(sweep.results))
                if all(np.max(abs(r.S-S_true))<.02 and np.max(abs(r.T-T_true))<.02 for r in sweep.results[i:])),None),
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
