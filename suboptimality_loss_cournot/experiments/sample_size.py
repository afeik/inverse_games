"""Sample-size experiment: sweep n_train at two noise levels.

Usage:  python -m experiments.sample_size
"""
import sys, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game import BipartiteGraph
from sub_estimator import SubEstimator
from vi_estimator import fit_vi

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR    = _ROOT / "data"
RESULTS     = _ROOT / "results" / "sample_size" / datetime.now().strftime("%Y%m%d_%H%M%S")
ETA_LEVELS  = [0.0, 0.50]
TRAIN_SIZES = [2, 5, 10, 20, 30]

SUB_ITERS   = 15000
SUB_LR      = 0.003
SUB_BATCH   = 5
VI_REG      = 0.001
VI_SOLVER   = "MOSEK"
MAX_ITER    = 5000
SEED        = 42


# ── Helpers ──────────────────────────────────────────────────────────────
def load_game(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}

def add_noise_and_split(clean, contexts, qmax, eta, n_train, rng):
    N = clean.shape[0]
    if eta > 0:
        d = clean[0].size
        norms = np.linalg.norm(clean.reshape(N, -1), axis=1)
        std = eta * norms / np.sqrt(d)
        noise = rng.normal(size=clean.shape)
        noise *= std.reshape(N, *([1]*(clean.ndim - 1)))
        noisy = np.clip(clean + noise, 0.0, qmax)
    else:
        noisy = clean.copy()
    perm = rng.permutation(N)
    return (contexts[perm[:n_train]], noisy[perm[:n_train]],
            contexts[perm[n_train:]], clean[perm[n_train:]])

def predict(solver, th, contexts):
    return np.array([solver.solve_forward(th, s, max_iter=MAX_ITER)[0] for s in contexts])

def errors(pred, true):
    return np.linalg.norm(pred - true, axis=(1, 2))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    games = sorted(DATA_DIR.glob("game_*.npz"))
    assert games, f"No data in {DATA_DIR}"

    g0 = load_game(games[0])
    graph = BipartiteGraph(g0["adjacency"])
    print(f"Graph {graph.num_firms}×{graph.num_markets} | "
          f"{len(games)} games | eta={ETA_LEVELS} | n_train={TRAIN_SIZES}")

    rows, raw_rows = [], []

    for gid, gf in enumerate(games):
        gd = load_game(gf)
        graph = BipartiteGraph(gd["adjacency"])
        qmax, th_true, mass = gd["qmax"], gd["theta_true"], float(gd["parameter_mass"])
        fwd = SubEstimator(graph, qmax, mass=mass)

        for eta in ETA_LEVELS:
            for nt in TRAIN_SIZES:
                rng = np.random.default_rng(SEED + gid*100000 + int(eta*1e6) + nt*7)
                ctx_tr, a_tr, ctx_te, a_te = add_noise_and_split(
                    gd["clean_actions"], gd["contexts"], qmax, eta, nt, rng)

                # SUB
                res = fwd.fit(a_tr, ctx_tr, num_iters=SUB_ITERS, lr=SUB_LR,
                              batch_size=SUB_BATCH, verbose=True)
                th_sub = res["theta"]
                e_sub = errors(predict(fwd, th_sub, ctx_te), a_te)
                rows.append({"game": gid, "eta": eta, "n_train": nt,
                             "method": "SUB", "test_median": float(np.median(e_sub))})
                for k, e in enumerate(e_sub):
                    raw_rows.append({"game": gid, "eta": eta, "n_train": nt,
                                     "method": "SUB", "sample": k, "error": float(e)})

                # VI
                th_vi = fit_vi(a_tr, ctx_tr, graph, qmax, mass=mass,
                               reg=VI_REG, solver=VI_SOLVER)
                e_vi = errors(predict(fwd, th_vi, ctx_te), a_te)
                rows.append({"game": gid, "eta": eta, "n_train": nt,
                             "method": "VI", "test_median": float(np.median(e_vi))})
                for k, e in enumerate(e_vi):
                    raw_rows.append({"game": gid, "eta": eta, "n_train": nt,
                                     "method": "VI", "sample": k, "error": float(e)})

                print(f"  game {gid+1:02d}/{len(games)}  eta={eta:.2f}  n={nt}")

        pd.DataFrame(rows).to_csv(RESULTS/"summary.csv", index=False)
        pd.DataFrame(raw_rows).to_csv(RESULTS/"raw_errors.csv", index=False)

    print(f"\nResults saved to {RESULTS}")


if __name__ == "__main__":
    main()
