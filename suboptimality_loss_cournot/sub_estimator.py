"""Suboptimality-loss estimator (SUB) for inverse Nash equilibrium learning.

Projects onto a scaled simplex {theta >= lower : 1'theta = mass} and minimises
the suboptimality loss via projected subgradient descent.

Best responses can be solved in closed form (default) or via OSQP.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import osqp
    from scipy import sparse
except ImportError:
    osqp = sparse = None

from game import BipartiteGraph, CournotParams, CournotGame

Array = np.ndarray


# ---------------------------------------------------------------------------
# Parameter index
# ---------------------------------------------------------------------------
@dataclass
class ParamIndex:
    alpha0: slice; delta: slice
    beta: List[slice]; c: List[slice]
    gamma: slice; dim: int


def param_index(F: int, M: int) -> ParamIndex:
    off = 0
    alpha0 = slice(off, off + M); off += M
    delta  = slice(off, off + M); off += M
    beta = [slice(off + i*M, off + (i+1)*M) for i in range(F)]; off += F*M
    c    = [slice(off + i*M, off + (i+1)*M) for i in range(F)]; off += F*M
    gamma  = slice(off, off + F); off += F
    return ParamIndex(alpha0=alpha0, delta=delta, beta=beta, c=c, gamma=gamma, dim=off)


# ---------------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------------
def project_simplex(v: Array, mass: float) -> Array:
    if mass <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cs = np.cumsum(u) - mass
    rho = int(np.where(u - cs / (np.arange(len(v)) + 1) > 0)[0][-1])
    return np.maximum(v - cs[rho] / (rho + 1), 0.0)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
class SubEstimator:
    """Projected subgradient solver for the suboptimality loss."""

    def __init__(self, graph: BipartiteGraph, qmax: Array,
                 mass: float = 25.0, beta_min: float = 0.0,
                 gamma_min: float = 0.0, method: str = "closed_form",
                 zero_inactive: bool = True) -> None:
        self.graph, self.F, self.M = graph, graph.num_firms, graph.num_markets
        self.qmax = np.asarray(qmax, float) * graph.adjacency
        self.pidx = param_index(self.F, self.M)
        self.mass = float(mass)
        self.beta_min = float(beta_min)
        self.gamma_min = float(gamma_min)
        self.method = method
        assert method in ("closed_form", "osqp")

        # Lower bounds
        self.lb = np.zeros(self.pidx.dim)
        for i in range(self.F):
            self.lb[self.pidx.beta[i]] = beta_min
        self.lb[self.pidx.gamma] = gamma_min

        # Fixed-zero indices for inactive edges
        self.fixed = (self._inactive_indices() if zero_inactive
                      else np.array([], dtype=int))

    def _inactive_indices(self) -> Array:
        idx = []
        for i in range(self.F):
            for k in range(self.M):
                if self.graph.adjacency[i, k] == 0:
                    idx.append(self.pidx.beta[i].start + k)
                    idx.append(self.pidx.c[i].start + k)
        return np.array(idx, dtype=int)

    # -- pack / unpack -------------------------------------------------
    def pack(self, alpha0, delta, beta, c, gamma) -> Array:
        th = np.zeros(self.pidx.dim)
        th[self.pidx.alpha0] = alpha0
        th[self.pidx.delta] = delta
        for i in range(self.F):
            th[self.pidx.beta[i]] = np.asarray(beta)[i]
            th[self.pidx.c[i]] = np.asarray(c)[i]
        th[self.pidx.gamma] = gamma
        return th

    def unpack(self, th: Array) -> Dict[str, Array]:
        beta = np.array([th[self.pidx.beta[i]] for i in range(self.F)])
        c = np.array([th[self.pidx.c[i]] for i in range(self.F)])
        return {"alpha0": th[self.pidx.alpha0].copy(),
                "delta": th[self.pidx.delta].copy(),
                "beta": beta, "c": c,
                "gamma": th[self.pidx.gamma].copy()}

    # -- projection ----------------------------------------------------
    def project(self, th: Array) -> Array:
        th = np.asarray(th, float).copy()
        if self.fixed.size == 0:
            slack = self.mass - self.lb.sum()
            return self.lb + project_simplex(th - self.lb, max(slack, 0.0))
        free = np.ones(self.pidx.dim, bool)
        free[self.fixed] = False
        lb_f = self.lb[free]
        out = np.zeros(self.pidx.dim)
        out[free] = lb_f + project_simplex(th[free] - lb_f,
                                           max(self.mass - lb_f.sum(), 0.0))
        return out

    def initial_theta(self) -> Array:
        th0 = self.lb.copy()
        th0 += (self.mass - th0.sum()) / len(th0)
        return self.project(th0)

    def to_game(self, th: Array) -> CournotGame:
        p = self.unpack(th)
        return CournotGame(self.graph,
                           CournotParams(p["alpha0"], p["delta"],
                                        p["beta"], p["c"], p["gamma"]),
                           self.qmax)

    # -- loss and gradient ---------------------------------------------
    def loss_and_gradient(self, th: Array, actions: Array,
                          contexts: Array) -> Tuple[float, Array]:
        """Average suboptimality loss and subgradient over a batch."""
        p = self.unpack(th)
        beta = np.maximum(p["beta"], self.beta_min)
        gamma = np.maximum(p["gamma"], self.gamma_min)
        ctx = np.asarray(contexts, float)
        if ctx.ndim == 1:
            ctx = np.repeat(ctx.reshape(-1, 1), self.M, axis=1)

        alpha_s = p["alpha0"][None, :] + ctx * p["delta"][None, :]  # (N, M)
        Q_obs = actions.sum(axis=1)                                  # (N, M)
        q_oth = Q_obs[:, None, :] - actions                          # (N, F, M)

        # --- best responses (N, F, M) ---
        if self.method == "closed_form":
            num = (alpha_s[:, None, :] - p["c"][None, :, :]
                   - beta[None, :, :] * q_oth)
            den = 2 * beta[None, :, :] + gamma[None, :, None]
            br = np.clip(num / den, 0.0, self.qmax[None, :, :])
        else:
            br = self._br_osqp_batch(beta, gamma, p["c"],
                                     alpha_s, actions, Q_obs)

        # --- loss and gradient (always vectorized) ---
        Q_br = q_oth + br
        u_br = ((alpha_s[:, None, :] - p["c"][None, :, :]
                 - beta[None, :, :] * Q_br) * br
                - 0.5 * gamma[None, :, None] * br**2).sum(axis=2)
        u_obs = ((alpha_s[:, None, :] - p["c"][None, :, :]
                  - beta[None, :, :] * Q_obs[:, None, :]) * actions
                 - 0.5 * gamma[None, :, None] * actions**2).sum(axis=2)

        dx = br - actions
        loss = float((u_br - u_obs).sum(axis=1).mean())
        g_a = dx.sum(axis=1).mean(axis=0)
        g_d = (ctx[:, None, :] * dx).sum(axis=1).mean(axis=0)
        g_b = (-(br * Q_br) + actions * Q_obs[:, None, :]).mean(axis=0)
        g_c = (-dx).mean(axis=0)
        g_g = (-0.5 * (br**2).sum(2) + 0.5 * (actions**2).sum(2)).mean(0)
        return loss, self.pack(g_a, g_d, g_b, g_c, g_g)

    def _br_osqp_batch(self, beta, gamma, c, alpha_s, actions, Q_obs):
        """Batched OSQP: one setup per firm, warm-started across samples."""
        if osqp is None or sparse is None:
            raise ImportError("Install osqp and scipy")
        N = actions.shape[0]
        br = np.zeros_like(actions)                          # (N, F, M)
        for i in range(self.F):
            active = np.where(self.qmax[i] > 0)[0]
            if active.size == 0:
                continue
            Ma = active.size
            P = sparse.diags(2.0 * beta[i, active] + gamma[i], format="csc")
            A = sparse.eye(Ma, format="csc")
            prob = osqp.OSQP()
            prob.setup(P=P, q=np.zeros(Ma), A=A,
                       l=np.zeros(Ma), u=self.qmax[i, active].copy(),
                       verbose=False, polish=False, warm_start=True,
                       eps_abs=1e-7, eps_rel=1e-7, max_iter=4000)
            for n in range(N):
                q_other = Q_obs[n] - actions[n, i]
                q_new = -(alpha_s[n, active] - c[i, active]
                          - beta[i, active] * q_other[active])
                prob.update(q=q_new)
                res = prob.solve()
                br[n, i, active] = np.clip(res.x, 0.0, self.qmax[i, active])
        return br

    # -- fit -----------------------------------------------------------
    def fit(self, actions: Array, contexts: Array, *,
            num_iters: int = 2000, lr: float = 0.1, reg: float = 1e-4,
            theta0: Optional[Array] = None, batch_size: Optional[int] = None,
            verbose: bool = True, seed: int = 0) -> dict:
        th = self.initial_theta() if theta0 is None else self.project(theta0)
        N = actions.shape[0]
        rng = np.random.RandomState(seed)
        hist = []

        rng_iter = range(num_iters)
        if verbose:
            try:
                from tqdm import tqdm
                rng_iter = tqdm(rng_iter, desc="SUB fit", leave=True)
            except ImportError:
                pass

        for it in rng_iter:
            if batch_size and batch_size < N:
                idx = rng.choice(N, batch_size, replace=False)
                a_b, c_b = actions[idx], contexts[idx]
            else:
                a_b, c_b = actions, contexts
            loss, grad = self.loss_and_gradient(th, a_b, c_b)
            loss_r = loss + 0.5 * reg * float(th @ th)
            th = self.project(th - lr * (grad + reg * th))
            hist.append(loss_r)
            if verbose and hasattr(rng_iter, "set_postfix"):
                rng_iter.set_postfix(loss=f"{loss_r:.6f}")
            if len(hist) > 50:
                recent = hist[-50:]
                if abs(recent[-1] - recent[0]) < 1e-8 * (1 + abs(recent[0])):
                    if verbose:
                        if hasattr(rng_iter, "write"):
                            rng_iter.write(f"  converged at iter {it+1}  loss={loss_r:.6f}")
                        else:
                            print(f"  converged at iter {it+1}  loss={loss_r:.6f}")
                    break
        return {"theta": th, "params": self.unpack(th), "loss_history": np.array(hist)}

    def solve_forward(self, th, context, **kw):
        return self.to_game(th).solve(context=context, **kw)
