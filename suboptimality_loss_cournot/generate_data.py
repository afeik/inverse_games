"""Generate clean equilibrium datasets for the inverse estimation experiments.

Usage:  python generate_data.py [--num-games 10] [--num-samples 70] [--seed 1]
"""
import argparse, json
import numpy as np
from pathlib import Path
from game import BipartiteGraph, CournotParams, CournotGame
from sub_estimator import SubEstimator

CONFIG_FILE = "config.json"
OUTPUT_DIR  = "data"


def load_config(path=CONFIG_FILE):
    with open(path) as f:
        return json.load(f)

def build_graph(cfg):
    adj = np.array(cfg["adjacency"], int)
    graph = BipartiteGraph(adj)
    qmax = cfg.get("default_qmax", 2.5) * graph.adjacency.astype(float)
    for idx_s, caps in cfg.get("qmax_overrides", {}).items():
        qmax[int(idx_s)] = np.array(caps, float)
    return graph, qmax


def sample_params(graph, rng, cfg):
    F, M = graph.num_firms, graph.num_markets
    active = graph.adjacency.astype(bool)
    r = cfg.get("param_ranges", {})
    alpha0 = rng.uniform(*r.get("alpha0", [0.5, 1.5]), size=M)
    delta  = rng.uniform(*r.get("delta",  [0.1, 0.5]), size=M)
    gamma  = rng.uniform(*r.get("gamma",  [0.3, 0.8]), size=F)
    beta = np.zeros((F, M)); beta[active] = rng.uniform(*r.get("beta", [0.3, 0.8]), size=int(active.sum()))
    c    = np.zeros((F, M)); c[active]    = rng.uniform(*r.get("c",    [0.1, 0.6]), size=int(active.sum()))
    return CournotParams(alpha0, delta, beta, c, gamma)


def pack_theta(graph, qmax, params):
    tmp = SubEstimator(graph, qmax, mass=100.0)
    th = tmp.pack(params.alpha0, params.delta, params.beta, params.c, params.gamma)
    return th, float(th.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-games", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=70)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg = load_config()
    graph, qmax_default = build_graph(cfg)
    F, M = graph.num_firms, graph.num_markets
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)

    qmax_range = cfg.get("qmax_edge_range", [0.3, 1.0])

    print(f"Graph: {F}×{M}, {int(graph.adjacency.sum())} edges")
    print(f"Games: {args.num_games}, Samples/game: {args.num_samples}\n")

    for gid in range(args.num_games):
        game_rng = np.random.default_rng(int(master_rng.integers(10**9)))
        # Per-edge capacities
        qmax_g = np.zeros_like(graph.adjacency, dtype=float)
        active = graph.adjacency.astype(bool)
        qmax_g[active] = game_rng.uniform(*qmax_range, size=int(active.sum()))

        params = sample_params(graph, game_rng, cfg)
        game = CournotGame(graph, params, qmax_g)
        theta, mass = pack_theta(graph, qmax_g, params)

        contexts = game_rng.uniform(-3, 3, size=(args.num_samples, M))
        clean = np.zeros((args.num_samples, F, M))
        conv = np.zeros(args.num_samples, dtype=bool)
        x_warm = None
        for j, s in enumerate(contexts):
            x, info = game.solve(s, x0=x_warm)
            clean[j] = x; conv[j] = info["converged"]; x_warm = x

        np.savez(out / f"game_{gid:03d}.npz",
                 contexts=contexts, clean_actions=clean, converged=conv,
                 qmax=qmax_g, theta_true=theta,
                 parameter_mass=np.array(mass), adjacency=graph.adjacency)
        print(f"  game {gid:03d}  converged={conv.mean():.0%}")

    print(f"\nSaved {args.num_games} games to {out}/")


if __name__ == "__main__":
    main()
