"""Plot noise-robustness results: 3-panel figure (Fig. 2).

(a) Equilibrium prediction error  (b) Parameter recovery error  (c) Calibration constants

Usage:  python -m plotting.noise [results_dir]
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game import BipartiteGraph
from sub_estimator import SubEstimator

# ── Style ─────────────────────────────────────────────────────────────────
C_SUB, C_VI = "#627313", "#0F62A9"
COLOR  = {"SUB": C_SUB, "VI": C_VI}
MARKER = {"SUB": "o",   "VI": "s"}
PRETTY = {"SUB": "SUB (ours)", "VI": "VI"}

DATA_DIR = _ROOT / "data"


def _latest(base):
    dirs = sorted(d for d in base.iterdir()
                  if d.is_dir() and (d/"raw_errors.csv").exists())
    assert dirs, f"No results in {base}"
    return dirs[-1]


def _fmt(v):
    if v == 0: return "0"
    return f"{v:g}"


def _panel(ax, agg, y, ylo, yhi, ylabel, title, etas):
    xm = {n: i for i, n in enumerate(etas)}
    for m in sorted(agg["method"].unique()):
        s = agg[agg["method"] == m].sort_values("eta")
        x = s["eta"].map(xm).values.astype(float)
        ax.fill_between(x, s[ylo].values, s[yhi].values, alpha=.18, color=COLOR[m])
        ax.plot(x, s[y].values, marker=MARKER[m], ms=4.5, lw=2,
                color=COLOR[m], label=PRETTY[m])
    ax.set_xticks(range(len(etas)))
    ax.set_xticklabels([_fmt(e) for e in etas], rotation=45, ha="right")
    ax.set_xlabel(r"Noise level $\eta$"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10); ax.grid(True, alpha=.3)


def _calib_panel(ax, df_c, etas):
    traces = [("SUB", "lambda_min_M", C_SUB, r"$\lambda_{\min}(\hat{M})$ SUB (ours)", "o"),
              ("VI",  "mu",           C_VI,  r"$\hat\mu$ VI",                          "s")]
    xm = {n: i for i, n in enumerate(etas)}
    for method, col, color, label, mk in traces:
        s = df_c[df_c["method"] == method]
        if s.empty: continue
        a = s.groupby("eta")[col].agg(median="median", lo="min", hi="max").reset_index()
        x = a["eta"].map(xm).values.astype(float)
        ax.fill_between(x, a["lo"].values, a["hi"].values, alpha=.18, color=color)
        ax.plot(x, a["median"].values, marker=mk, ms=4.5, lw=2, color=color, label=label)
    ax.set_xticks(range(len(etas)))
    ax.set_xticklabels([_fmt(e) for e in etas], rotation=45, ha="right")
    ax.set_xlabel(r"Noise level $\eta$"); ax.set_ylabel("Calibration constant")
    ax.set_title("(c)  Estimated calibration constants", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=.3)


def compute_constants(theta_dir):
    games = sorted(DATA_DIR.glob("game_*.npz"))
    meta = {}
    for gid, gf in enumerate(games):
        d = np.load(gf)
        g = BipartiteGraph(d["adjacency"])
        meta[gid] = {"solver": SubEstimator(g, d["qmax"], mass=float(d["parameter_mass"])),
                      "th_true": d["theta_true"]}
    pat = re.compile(r"^(sub|vi)_g(\d+)_eta([\d.]+)\.npz$")
    rows = []
    for f in sorted(theta_dir.glob("*.npz")):
        m = pat.match(f.name)
        if not m: continue
        method, gid, eta = m.group(1).upper(), int(m.group(2)), float(m.group(3))
        if gid not in meta: continue
        th = np.load(f)["theta"]
        s = meta[gid]["solver"]
        true_n = np.linalg.norm(meta[gid]["th_true"])
        hat_n = np.linalg.norm(th)
        th_s = th * (true_n / hat_n) if hat_n > 1e-12 else th
        c = s.to_game(th_s).jacobian_constants()
        rows.append({"game": gid, "eta": eta, "method": method, **c})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest(_ROOT / "results" / "noise")
    print(f"Results: {rdir}")

    raw = pd.read_csv(rdir / "raw_errors.csv")
    summ = pd.read_csv(rdir / "summary.csv")
    etas = sorted(raw["eta"].unique())

    agg_err = raw.groupby(["eta","method"])["error"].agg(
        median="median", lo="min", hi="max").reset_index()
    agg_th = summ.groupby(["eta","method"])["theta_err"].agg(
        median="median", lo="min", hi="max").reset_index()

    theta_dir = rdir / "theta"
    has_c = theta_dir.is_dir()
    df_c = compute_constants(theta_dir) if has_c else None

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    ncols = 3 if (has_c and df_c is not None and len(df_c)) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2*ncols, 4.5))

    _panel(axes[0], agg_err, "median", "lo", "hi",
           r"$\|\hat{x} - x^\star\|_2$",
           "(a)  Equilibrium prediction error", etas)
    _panel(axes[1], agg_th, "median", "lo", "hi",
           r"$\|\hat{\theta} - \theta^\star\|_2 / \|\theta^\star\|_2$",
           "(b)  Parameter recovery error", etas)
    if ncols == 3:
        _calib_panel(axes[2], df_c, etas)

    handles = [Line2D([0],[0], color=COLOR[m], marker=MARKER[m], lw=2, ms=5,
                      label=PRETTY[m]) for m in ["SUB","VI"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02),
               ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .93])
    out = rdir / "noise_robustness.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
