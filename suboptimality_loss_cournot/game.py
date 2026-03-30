"""Networked Cournot competition model.

Firm i in market k faces inverse demand  p_{ik} = alpha0_k + delta_k * s_k - beta_{ik} * Q_k
and has cost  C_i(x_i) = c_i * x_i + 0.5 * gamma_i * ||x_i||^2.
Equilibria are computed via projected gradient descent on the pseudo-gradient.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

Array = np.ndarray


@dataclass
class BipartiteGraph:
    """Firm–market adjacency: adjacency[i,k] = 1 iff firm i is in market k."""
    adjacency: Array

    def __post_init__(self):
        A = np.asarray(self.adjacency, dtype=int)
        assert A.ndim == 2 and np.isin(A, [0, 1]).all()
        assert np.all(A.sum(axis=1) > 0) and np.all(A.sum(axis=0) > 0)
        self.adjacency = A

    @property
    def num_firms(self) -> int:
        return self.adjacency.shape[0]

    @property
    def num_markets(self) -> int:
        return self.adjacency.shape[1]


@dataclass
class CournotParams:
    """Parameters: alpha0(M,) delta(M,) beta(F,M) c(F,M) gamma(F,)."""
    alpha0: Array
    delta: Array
    beta: Array
    c: Array
    gamma: Array

    def __post_init__(self):
        self.alpha0 = np.asarray(self.alpha0, float)
        self.delta = np.asarray(self.delta, float)
        self.beta = np.asarray(self.beta, float)
        self.c = np.asarray(self.c, float)
        self.gamma = np.asarray(self.gamma, float)


class CournotGame:
    """Forward game: solve Nash equilibrium via projected gradient descent."""

    def __init__(self, graph: BipartiteGraph, params: CournotParams,
                 qmax: Array) -> None:
        self.graph = graph
        self.params = params
        self.qmax = np.asarray(qmax, float) * graph.adjacency
        self.F = graph.num_firms
        self.M = graph.num_markets

    def solve(self, context: Array, x0: Optional[Array] = None,
              max_iter: int = 20_000, tol: float = 1e-10,
              step_size: float = 0.05) -> Tuple[Array, dict]:
        """Projected gradient descent on the pseudo-gradient."""
        alpha_s = self.params.alpha0 + self.params.delta * np.asarray(context, float)
        beta = self.params.beta
        x = np.zeros((self.F, self.M)) if x0 is None else np.array(x0, float)

        for it in range(max_iter):
            x_old = x.copy()
            Q = x.sum(axis=0)
            grad = (alpha_s - self.params.c
                    - beta * (Q + x)
                    - self.params.gamma[:, None] * x)
            x = np.clip(x + step_size * grad, 0.0, self.qmax)
            if np.max(np.abs(x - x_old)) <= tol:
                return x, {"converged": True, "iterations": it + 1}
        return x, {"converged": False, "iterations": max_iter}

    def jacobian_constants(self) -> Dict[str, float]:
        """Strong-monotonicity mu, smoothness L, and lambda_min(M)."""
        beta, gamma, adj = self.params.beta, self.params.gamma, self.graph.adjacency
        all_eigs, all_M_eigs = [], []
        for m in range(self.M):
            firms = np.where(adj[:, m] == 1)[0]
            if len(firms) == 0:
                continue
            b, g = beta[firms, m], gamma[firms]
            H = np.outer(b, np.ones(len(firms)))
            np.fill_diagonal(H, 2.0 * b + g)
            all_eigs.extend(np.linalg.eigvalsh(0.5 * (H + H.T)).tolist())
            Mc = H.copy()
            np.fill_diagonal(Mc, b + 0.5 * g)
            all_M_eigs.extend(np.linalg.eigvalsh(0.5 * (Mc + Mc.T)).tolist())

        mu = float(min(all_eigs)) if all_eigs else 0.0
        L = max(float(np.max(2.0 * beta[i, adj[i] == 1] + gamma[i]))
                for i in range(self.F) if adj[i].any())
        return {"mu": mu, "L": L, "lambda_min_M": float(min(all_M_eigs)) if all_M_eigs else 0.0}
