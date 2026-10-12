"""Decomposition-coordination solver for  -div(|grad u|^{p(x)-2} grad u) = f,  u = g on dOmega.

Iteration (r = augmented-Lagrangian parameter, r = 1 reproduces the paper):
    1. r K u_n = F + int (r nu_{n-1} - eta_n) . grad(phi)      [linear, same matrix every step]
    2. |nu_n|^{p-2} nu_n + r nu_n = eta_n + r grad u_n        [pointwise scalar equation]
    3. eta_{n+1} = eta_n + r (grad u_n - nu_n)

(In the MatLab code the variable called ``lambda`` is eta and ``eta`` is nu.)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import Mesh
from .fem import P1Space
from .nonlinear import update_nu


@dataclass
class SolveResult:
    u: np.ndarray                 # nodal values
    nu: tuple                     # (nu_x, nu_y) per element
    eta: tuple                    # (eta_x, eta_y) per element
    iterations: int
    converged: bool
    history: dict = field(default_factory=dict)   # per-iteration diagnostics
    time_setup: float = 0.0
    time_iter: float = 0.0


class PxLaplaceSolver:
    """Set-up once (mesh, p, f, g, factorisation), then call :meth:`solve`.

    Parameters
    ----------
    mesh : Mesh  (any triangulation; boundary nodes in ``mesh.dirichlet``)
    p, f, g : constant, array, or callable ``fun(x, y)`` (vectorised if possible).
        p : exponent, sampled at element barycentres (piecewise constant, as in the paper)
        f : source term,  F_i = int f phi_i
        g : Dirichlet data, evaluated at ``mesh.dirichlet`` nodes (or an array of nodal values
            of length Nd or Nn)
    r : augmented-Lagrangian parameter (paper: r = 1)
    f_degree : quadrature degree for the load vector (1 = barycentre rule like the MatLab code)
    """

    def __init__(self, mesh: Mesh, p, f=0.0, g=0.0, r: float = 1.0, f_degree: int = 1):
        t0 = time.perf_counter()
        self.mesh, self.r = mesh, float(r)
        self.fem = P1Space(mesh)
        self.p = self.fem.element_values(p)
        if np.any(self.p <= 1):
            raise ValueError("p(x) must be > 1")
        self.F = self.fem.load_vector(f, degree=f_degree)
        # Dirichlet data
        D = mesh.dirichlet
        if np.ndim(g) == 1 and not callable(g) and len(g) == mesh.n_nodes:
            self.gD = np.asarray(g, float)[D]
        else:
            self.gD = self.fem.node_values(g, D)
        mask = np.ones(mesh.n_nodes, bool)
        mask[D] = False
        self.free = np.flatnonzero(mask)
        K = self.r * self.fem.K
        self.Kff = K[self.free][:, self.free].tocsc()
        self.Kfd = K[self.free][:, D].tocsr()
        self._lu = spla.splu(self.Kff)            # factorised ONCE (matrix is constant)
        self.time_setup = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _solve_linear(self, rhs):
        u = np.empty(self.mesh.n_nodes)
        u[self.mesh.dirichlet] = self.gD
        u[self.free] = self._lu.solve(rhs[self.free] - self.Kfd @ self.gD)
        return u

    def solve(self, tol: float = 1e-5, max_iter: int = 2000, stop: str = "rel_u",
              init: str = "zero", seed: int = 0, callback=None, verbose: bool = False) -> SolveResult:
        """Run the iteration.

        stop : 'rel_u'   -> ||u_n - u_{n-1}||_2 / ||u_{n-1}||_2 < tol   (criterion in the MatLab code)
               'p_error' -> (J_p(u_n - u_{n-1}))^(1/max p) < tol         (criterion in the paper)
        init : 'zero' (nu = eta = 0, recommended) or 'random' (0.15*rand, as in the MatLab code)
        """
        if stop not in ("rel_u", "p_error"):
            raise ValueError("stop must be 'rel_u' or 'p_error'")
        t0 = time.perf_counter()
        fem, r, p, Nt = self.fem, self.r, self.p, self.fem.Nt
        if init == "random":
            rng = np.random.default_rng(seed)
            nux, nuy = 0.15 * rng.random(Nt), 0.15 * rng.random(Nt)
            etax, etay = 0.15 * rng.random(Nt), 0.15 * rng.random(Nt)
        else:
            nux = nuy = etax = etay = np.zeros(Nt)
        pmax = float(p.max())
        hist = {k: [] for k in ("rel_u", "p_error", "residual", "energy")}
        u_old = np.zeros(self.mesh.n_nodes)
        u_old[self.mesh.dirichlet] = self.gD
        converged, it = False, 0
        for it in range(1, max_iter + 1):
            # 1. linear PDE (constant matrix, pre-factorised)
            rhs = self.F + fem.weak_div(r * nux - etax, r * nuy - etay)
            u = self._solve_linear(rhs)
            gx, gy = fem.gradient(u)
            # 2. pointwise nonlinear algebraic equation
            nux, nuy = update_nu(p, etax / r + gx, etay / r + gy, r)
            # 3. multiplier update
            etax = etax + r * (gx - nux)
            etay = etay + r * (gy - nuy)
            # diagnostics
            du = u - u_old
            nrm = np.linalg.norm(u_old)
            rel = np.linalg.norm(du) / nrm if nrm > 0 else np.inf
            perr = fem.modular(du, p) ** (1.0 / pmax)
            hist["rel_u"].append(rel)
            hist["p_error"].append(perr)
            hist["residual"].append(float(np.sqrt(np.sum(fem.area * ((gx - nux) ** 2 + (gy - nuy) ** 2)))))
            hist["energy"].append(float(np.sum(fem.area * np.hypot(gx, gy) ** p / p) - self.F @ u))
            if verbose and (it % 50 == 0 or it == 1):
                print(f"it {it:5d}  rel_u={rel:.3e}  p_err={perr:.3e}  E={hist['energy'][-1]:.8f}")
            if callback is not None:
                callback(it, u, hist)
            u_old = u
            crit = rel if stop == "rel_u" else perr
            if crit < tol:
                converged = True
                break
        return SolveResult(u=u, nu=(nux, nuy), eta=(etax, etay), iterations=it, converged=converged,
                           history={k: np.array(v) for k, v in hist.items()},
                           time_setup=self.time_setup, time_iter=time.perf_counter() - t0)


def solve(mesh: Mesh, p, f=0.0, g=0.0, r=1.0, **kw) -> SolveResult:
    """One-shot convenience wrapper:  solve(mesh, p, f, g, tol=..., stop=...)."""
    init_keys = {"f_degree"}
    ik = {k: kw.pop(k) for k in list(kw) if k in init_keys}
    return PxLaplaceSolver(mesh, p, f, g, r=r, **ik).solve(**kw)
