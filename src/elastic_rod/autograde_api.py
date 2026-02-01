from __future__ import annotations
"""
Autograder entry points.

Gradescope (and local mimic script) will import from this module.

Stable boundary:
  - RodEnergy.value_and_grad(x) must work
  - bfgs(...) must return a BFGSResult with fields documented in bfgs.py
"""
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from .bfgs import bfgs
from .model import RodEnergy, RodParams
from .utils import random_loop, pack

def run_instance(*, N: int, seed: int, steps: int, tol: float, params: Optional[RodParams] = None) -> Dict[str, Any]:
    params = params or RodParams()
    model = RodEnergy(params)

    x0 = pack(random_loop(N, radius=7.0, noise=0.5, seed=seed))

    def f_and_g(x: np.ndarray):
        return model.value_and_grad(x)

    f0, g0 = f_and_g(x0)
    t0 = time.perf_counter()
    res = bfgs(f_and_g, x0, tol=tol, max_iter=steps, alpha0=1.0)
    t1 = time.perf_counter()

    x = np.asarray(res.x, dtype=np.float64)
    checksum = float(np.sin(x[:200].sum()) + np.cos(x[:200].dot(np.arange(min(200, x.size)))))

    return {
        "N": int(N),
        "seed": int(seed),
        "steps": int(steps),
        "tol": float(tol),
        "params": asdict(params),
        "x0": [float(v) for v in x0],
        "f0": float(f0),
        "g0": [float(v) for v in g0],
        "f_final": float(res.f),
        "gnorm_final": float(np.linalg.norm(res.g)),
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "n_feval": int(res.n_feval),
        "runtime_sec": float(t1 - t0),
        "energy_history": [float(v) for v in res.history.get("f", [])],
        "checksum": checksum,
    }

def run_suite(mode: str = "accuracy") -> Dict[str, Any]:
    if mode == "accuracy":
        cases = [
            dict(N=60, seed=0, steps=120, tol=1e-6, params=RodParams(kb=1.0, ks=60.0, l0=0.5, kc=0.01, eps=1.0, sigma=0.35)),
            dict(N=80, seed=1, steps=160, tol=1e-6, params=RodParams(kb=1.0, ks=80.0, l0=0.5, kc=0.015, eps=1.0, sigma=0.35)),
        ]
    elif mode == "speed":
        cases = [
            dict(N=160, seed=2, steps=160, tol=2e-6, params=RodParams(kb=1.0, ks=90.0, l0=0.5, kc=0.02, eps=1.0, sigma=0.35)),
            dict(N=220, seed=3, steps=160, tol=2e-6, params=RodParams(kb=0.8, ks=90.0, l0=0.5, kc=0.02, eps=1.0, sigma=0.35)),
        ]
    else:
        raise ValueError("mode must be 'accuracy' or 'speed'")
    return {"mode": mode, "results": [run_instance(**c) for c in cases]}
