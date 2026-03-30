"""Plot sample-size results: 2-panel figure (Fig. 3).

(a) eta = 0   (b) eta = 0.5

Usage:  python -m plotting.sample_size [results_dir]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parent.parent

# ── Style ─────────────────────────────────────────────────────────────────
COLOR  = {"SUB": "#627313", "VI": "#0F62A9"}
MARKER = {"SUB": "o",       "VI": "s"}
PRETTY = {"SUB": "SUB (ours)", "VI": "VI"}


def _latest(base):
    dirs = sorted(d for d in base.iterdir()
                  if d.is_dir() and (d/"raw_errors.csv").exists())
    assert dirs, f"No results in {base}"
    return dirs[-1]


def main():
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest(_ROOT / "results" / "sample_size")
    print(f"Results: {rdir}")

    raw = pd.read_csv(rdir / "raw_errors.csv")
    etas = sorted(raw["eta"].unique())
    print(f"eta levels: {etas}, train sizes: {sorted(raw['n_train'].unique())}")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    ncols = len(etas)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5*ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    for idx, eta in enumerate(etas):
        ax = axes[idx]
        sub = raw[raw["eta"] == eta]
        agg = sub.groupby(["n_train", "method"])["error"].agg(
            median="median", lo="min", hi="max").reset_index()

        for m in sorted(agg["method"].unique()):
            s = agg[agg["method"] == m].sort_values("n_train")
            x = s["n_train"].values.astype(float)
            ax.fill_between(x, s["lo"].values, s["hi"].values,
                            alpha=.18, color=COLOR[m])
            ax.plot(x, s["median"].values, marker=MARKER[m], ms=5,
                    lw=2.2, color=COLOR[m], label=PRETTY[m])

        lbl = chr(ord("a") + idx)
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel(r"$\|\hat{x} - x^\star\|_2$")
        title = "No noise" if eta == 0 else "Noisy"
        ax.set_title(f"({lbl})  {title}: $\\eta = {eta:g}$", fontsize=10)
        ax.grid(True, alpha=.3)

    handles = [Line2D([0],[0], color=COLOR[m], marker=MARKER[m], lw=2.2,
                      ms=5, label=PRETTY[m]) for m in ["SUB","VI"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02),
               ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .93])
    out = rdir / "sample_size.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
