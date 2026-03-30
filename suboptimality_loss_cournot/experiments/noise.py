"""Noise-robustness experiment: sweep eta in [0, 1] on pre-generated games.

Usage:  python -m experiments.noise
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
DATA_DIR   = _ROOT / "data"
RESULTS    = _ROOT / "results" / "noise" / datetime.now().strftime("%Y%m%d_%H%M%S")
ETA_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0]
NUM_TRAIN  = 20

SUB_ITERS  = 5000
SUB_LR     = 0.003
SUB_REG    = 0.0
VI_REG     = 0.001
VI_SOLVER  = "MOSEK"
MAX_ITER   = 5000
SEED       = 42


# ── Helpers ───────────────────────────────────────────────────────────────
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
    tr, te = perm[:n_train], perm[n_train:]
    return (contexts[tr], noisy[tr], contexts[te], clean[te])

def predict(solver, th, contexts):
    return np.array([solver.solve_forward(th, s, max_iter=MAX_ITER)[0] for s in contexts])

def errors(pred, true):
    return np.linalg.norm(pred - true, axis=(1, 2))

def theta_error(th_hat, th_true):
    a = th_hat / max(np.linalg.norm(th_hat), 1e-12)
    b = th_true / max(np.linalg.norm(th_true), 1e-12)
    return float(np.linalg.norm(a - b))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "theta").mkdir(exist_ok=True)
    games = sorted(DATA_DIR.glob("game_*.npz"))
    assert games, f"No data in {DATA_DIR}. Run generate_data.py first."

    g0 = load_game(games[0])
    graph = BipartiteGraph(g0["adjacency"])
    N_total = g0["clean_actions"].shape[0]
    print(f"Graph {graph.num_firms}×{graph.num_markets} | "
          f"{len(games)} games | {N_total} samples/game | train={NUM_TRAIN}")

    rows, raw_rows = [], []

    for gid, gf in enumerate(games):
        gd = load_game(gf)
        graph = BipartiteGraph(gd["adjacency"])
        qmax, th_true, mass = gd["qmax"], gd["theta_true"], float(gd["parameter_mass"])

        fwd = SubEstimator(graph, qmax, mass=mass)
        jac = fwd.to_game(th_true).jacobian_constants()

        for eta in ETA_LEVELS:
            rng = np.random.default_rng(SEED + gid*10000 + int(eta*1e6))
            ctx_tr, a_tr, ctx_te, a_te = add_noise_and_split(
                gd["clean_actions"], gd["contexts"], qmax, eta, NUM_TRAIN, rng)

            # SUB
            t0 = time.perf_counter()
            res = fwd.fit(a_tr, ctx_tr, num_iters=SUB_ITERS, lr=SUB_LR,
                          reg=SUB_REG, verbose=True)
            t_sub = time.perf_counter() - t0
            th_sub = res["theta"]
            p_te = predict(fwd, th_sub, ctx_te)
            e_te = errors(p_te, a_te)
            rows.append({"game": gid, "eta": eta, "method": "SUB",
                         "test_median": float(np.median(e_te)),
                         "theta_err": theta_error(th_sub, th_true),
                         "time": t_sub, **jac})
            for k, e in enumerate(e_te):
                raw_rows.append({"game": gid, "eta": eta, "method": "SUB",
                                 "sample": k, "error": float(e)})
            np.savez(RESULTS/"theta"/f"sub_g{gid:03d}_eta{eta:.4f}.npz", theta=th_sub)

            # VI
            t0 = time.perf_counter()
            th_vi = fit_vi(a_tr, ctx_tr, graph, qmax, mass=mass,
                           reg=VI_REG, solver=VI_SOLVER)
            t_vi = time.perf_counter() - t0
            p_te_vi = predict(fwd, th_vi, ctx_te)
            e_te_vi = errors(p_te_vi, a_te)
            rows.append({"game": gid, "eta": eta, "method": "VI",
                         "test_median": float(np.median(e_te_vi)),
                         "theta_err": theta_error(th_vi, th_true),
                         "time": t_vi, **jac})
            for k, e in enumerate(e_te_vi):
                raw_rows.append({"game": gid, "eta": eta, "method": "VI",
                                 "sample": k, "error": float(e)})
            np.savez(RESULTS/"theta"/f"vi_g{gid:03d}_eta{eta:.4f}.npz", theta=th_vi)

            print(f"  game {gid+1:02d}/{len(games)}  eta={eta:.2f}  done")

        pd.DataFrame(rows).to_csv(RESULTS/"summary.csv", index=False)
        pd.DataFrame(raw_rows).to_csv(RESULTS/"raw_errors.csv", index=False)

    print(f"\nResults saved to {RESULTS}")


if __name__ == "__main__":
    main()
