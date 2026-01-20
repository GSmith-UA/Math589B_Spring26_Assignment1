from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

ValueGrad = Callable[[np.ndarray], Tuple[float, np.ndarray]]

@dataclass
class BFGSResult:
    x: np.ndarray
    f: float
    g: np.ndarray
    n_iter: int
    n_feval: int
    converged: bool
    history: Dict[str, Any]

def backtracking_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_steps: int = 30,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Simple Armijo backtracking line search.
    Returns (alpha, f_new, g_new, n_feval_increment).
    """
    # Armijo: f(x + a p) <= f(x) + c1 a g^T p
    alpha = float(alpha0)
    nfev = 0
    gp = float(np.dot(g, p))
    # If direction is not descent, return no step (caller may handle)
    if gp >= 0:
        return 0.0, f, g, 0

    for _ in range(max_steps):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        nfev += 1
        if f_new <= f + c1 * alpha * gp:
            return alpha, f_new, g_new, nfev
        alpha *= tau

    # return last evaluated values if no Armijo satisfied
    return alpha, f_new, g_new, nfev

def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Minimize f(x) with BFGS.

    You should:
    - maintain an approximation H_k to the inverse Hessian
    - compute p_k = -H_k g_k
    - perform a line search to get step alpha_k
    - update x, f, g
    - update H via the BFGS formula (with curvature checks)

    Return BFGSResult with a small iteration history useful for plotting.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    n_feval = 1

    n = x.size
    H = np.eye(n)  # inverse Hessian approximation

    hist = {"f": [f], "gnorm": [np.linalg.norm(g)], "alpha": []}

    for k in range(max_iter):
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                              converged=True, history=hist)

        # Search direction
        p = -H @ g
        # If not a descent direction, fall back to negative gradient
        if np.dot(g, p) >= 0:
            p = -g

        # Line search
        alpha, f_new, g_new, inc = backtracking_line_search(f_and_g, x, f, g, p, alpha0=alpha0)
        n_feval += inc

        # Update step
        x_new = x + alpha * p
        s = x_new - x
        y = g_new - g

        # BFGS update for H with curvature check y^T s > 0.
        ys = float(np.dot(y, s))
        if ys > 0.0:
            rho = 1.0 / ys
            I = np.eye(n)
            # (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            Hy = np.outer(y, s)  # helper for shapes
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        # else: skip update (keep H)

        # advance
        x = x_new
        f = f_new
        g = g_new

        hist["f"].append(f)
        hist["gnorm"].append(np.linalg.norm(g))
        hist["alpha"].append(alpha)

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval,
                      converged=False, history=hist)