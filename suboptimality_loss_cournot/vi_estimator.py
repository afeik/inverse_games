"""Variational-inequality (VI) estimator for inverse equilibrium learning.
(Bertsimas et al. 2015, "Data-driven estimation in equilibrium
using inverse optimization")

Solves a convex program:  min_{theta>=0, 1'theta=mass}  (1/N) sum slack_j + 0.5*reg*||theta||^2
subject to KKT feasibility constraints for each observation.
Requires CVXPY with a conic solver (MOSEK recommended).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import cvxpy as cp

from game import BipartiteGraph
from sub_estimator import param_index

Array = np.ndarray


def _build_phi(x_edge, s, graph, edge_list, pidx):
    """Feature matrix Phi in R^{E x dim(theta)} mapping theta to the pseudo-gradient."""
    F, M = graph.num_firms, graph.num_markets
    E = len(edge_list)
    Phi = np.zeros((E, pidx.dim))
    x_mat = np.zeros((F, M))
    for e, (i, k) in enumerate(edge_list):
        x_mat[i, k] = x_edge[e]
    Q = x_mat.sum(axis=0)
    for e, (i, k) in enumerate(edge_list):
        s_val = float(s[k]) if s.size == M else float(s[0])
        Phi[e, pidx.alpha0.start + k] = 1.0
        Phi[e, pidx.delta.start + k] = s_val
        Phi[e, pidx.beta[i].start + k] = -(Q[k] + x_mat[i, k])
        Phi[e, pidx.c[i].start + k] = -1.0
        Phi[e, pidx.gamma.start + i] = -x_mat[i, k]
    return Phi


def fit_vi(actions: Array, contexts: Array, graph: BipartiteGraph,
           qmax: Array, mass: float = 100.0, reg: float = 0.0,
           solver: str = "MOSEK", zero_inactive: bool = True,
           verbose: bool = False) -> Array:
    """Solve the VI inverse estimator. Returns theta_hat of length dim(theta)."""
    actions = np.asarray(actions, float)
    qmax = np.asarray(qmax, float)
    N, F, M = actions.shape
    pidx = param_index(F, M)
    m = pidx.dim

    # Build edge representation
    edge_list = [(i, k) for i in range(F) for k in range(M) if graph.adjacency[i, k] == 1]
    E = len(edge_list)
    X_edge = np.zeros((N, E))
    for e, (i, k) in enumerate(edge_list):
        X_edge[:, e] = actions[:, i, k]
    qmax_edge = np.array([qmax[i, k] for i, k in edge_list])

    contexts = np.asarray(contexts, float)
    if contexts.ndim == 1:
        contexts = contexts.reshape(-1, 1)

    # Inactive parameter indices
    fixed_idx = []
    if zero_inactive:
        for i in range(F):
            for k in range(M):
                if graph.adjacency[i, k] == 0:
                    fixed_idx.append(pidx.beta[i].start + k)
                    fixed_idx.append(pidx.c[i].start + k)

    # Box constraints
    G = np.vstack([np.eye(E), -np.eye(E)])
    h = np.hstack([qmax_edge, np.zeros(E)])

    # CVXPY model
    theta = cp.Variable(m, nonneg=True)
    slack = cp.Variable(N, nonneg=True)
    cons = [cp.sum(theta) == mass]
    if fixed_idx:
        cons.append(theta[fixed_idx] == 0.0)

    for j in range(N):
        _, Phi = None, _build_phi(X_edge[j], contexts[j], graph, edge_list, pidx)
        lam = cp.Variable(G.shape[0], nonneg=True)
        cons += [G.T @ lam == Phi @ theta,
                 lam @ (h - G @ X_edge[j]) <= slack[j]]

    obj = (1.0 / N) * cp.sum(slack)
    if reg > 0:
        obj += 0.5 * reg * cp.sum_squares(theta)

    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=getattr(cp, solver), verbose=verbose)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=verbose)

    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"VI solve failed: {prob.status}")
    return np.asarray(theta.value).ravel()
