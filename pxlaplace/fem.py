"""P1 finite elements on a triangular mesh (fully vectorised)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from .mesh import Mesh


class P1Space:
    """Piecewise-linear space on ``mesh``.

    Precomputes element areas, the (constant) gradient operators
    Gx, Gy : (Nt x Nn) sparse so that  grad u|_T = (Gx u, Gy u)[T],
    and the stiffness matrix K_ij = int grad(phi_i).grad(phi_j).
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        V, T = mesh.vertices, mesh.triangles
        self.Nn, self.Nt = mesh.n_nodes, mesh.n_elems
        x, y = V[T, 0], V[T, 1]                      # (Nt,3)
        # b_i = y_j - y_k, c_i = x_k - x_j  (cyclic);  grad phi_i = (b_i, c_i) / (2|T|)
        b = np.column_stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]])
        c = np.column_stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]])
        det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        if np.any(det <= 0):
            raise ValueError("degenerate or clockwise triangles in mesh")
        self.area = 0.5 * det
        self.gx = b / det[:, None]                   # (Nt,3) d phi_i / dx on T
        self.gy = c / det[:, None]
        self.bary = V[T].mean(axis=1)                # (Nt,2) barycentres
        rows = np.repeat(np.arange(self.Nt), 3)
        cols = T.ravel()
        self.Gx = sp.csr_matrix((self.gx.ravel(), (rows, cols)), shape=(self.Nt, self.Nn))
        self.Gy = sp.csr_matrix((self.gy.ravel(), (rows, cols)), shape=(self.Nt, self.Nn))
        # K = Gx^T diag(area) Gx + Gy^T diag(area) Gy
        W = sp.diags(self.area)
        self.K = (self.Gx.T @ W @ self.Gx + self.Gy.T @ W @ self.Gy).tocsr()

    # ---- linear forms --------------------------------------------------
    def load_vector(self, f, degree: int = 1) -> np.ndarray:
        """b_i = int f phi_i.  degree=1: barycentre rule (as in the MatLab code,
        exact for f in P1); degree=2: 3-point edge-midpoint rule (exact for P2)."""
        V, T = self.mesh.vertices, self.mesh.triangles
        b = np.zeros(self.Nn)
        if degree == 1:
            val = _eval(f, self.bary)
            contrib = np.repeat((self.area * val / 3.0)[:, None], 3, axis=1)
        elif degree == 2:
            P = V[T]
            mids = [0.5 * (P[:, 1] + P[:, 2]), 0.5 * (P[:, 2] + P[:, 0]), 0.5 * (P[:, 0] + P[:, 1])]
            fm = np.column_stack([_eval(f, m) for m in mids])      # f at midpoint opposite vertex i
            # phi_i at midpoint k = 1/2 if k != i else 0 ; weights area/3
            S = fm.sum(axis=1, keepdims=True) - fm                  # sum over k != i
            contrib = self.area[:, None] / 3.0 * 0.5 * S
        else:
            raise ValueError("degree must be 1 or 2")
        np.add.at(b, T.ravel(), contrib.ravel())
        return b

    def weak_div(self, qx, qy) -> np.ndarray:
        """b_i = int q . grad(phi_i) for an element-wise constant vector field q."""
        return self.Gx.T @ (self.area * qx) + self.Gy.T @ (self.area * qy)

    # ---- evaluation ----------------------------------------------------
    def gradient(self, u):
        """Gradient of the P1 interpolant (constant per element): (qx, qy)."""
        return self.Gx @ u, self.Gy @ u

    def element_values(self, fun_or_const) -> np.ndarray:
        """Evaluate a function at the barycentre of each element (used for p(x))."""
        return _eval(fun_or_const, self.bary, nt=self.Nt)

    def node_values(self, fun_or_const, idx=None) -> np.ndarray:
        P = self.mesh.vertices if idx is None else self.mesh.vertices[idx]
        return _eval(fun_or_const, P)

    # ---- integrals -----------------------------------------------------
    def modular(self, w, p) -> float:
        """J_p(w) = int |w|^p dx for a P1 function w (nodal values) and exponent p
        constant per element (array) or scalar; 3-point edge-midpoint rule."""
        P = w[self.mesh.triangles]                                  # (Nt,3)
        mid = 0.5 * (P + P[:, [1, 2, 0]])                           # values at 3 edge midpoints
        p = np.broadcast_to(np.asarray(p, float), (self.Nt,))
        return float(np.sum(self.area * np.mean(np.abs(mid) ** p[:, None], axis=1)))

    def norm_L2(self, w) -> float:
        return self.modular(w, 2.0) ** 0.5

    def norm_Lp(self, w, p) -> float:
        p = np.asarray(p, float)
        return self.modular(w, p) ** (1.0 / float(np.max(p)))

    @staticmethod
    def norm_inf(w) -> float:
        return float(np.max(np.abs(w)))


def _eval(fun, pts, nt=None):
    """Evaluate scalar function / constant / array at points (N,2) -> (N,)."""
    n = len(pts)
    if callable(fun):
        try:
            v = np.asarray(fun(pts[:, 0], pts[:, 1]), dtype=float)
            if v.shape == ():
                v = np.full(n, float(v))
        except Exception:
            v = np.array([fun(a, b) for a, b in pts], dtype=float)
        if v.shape != (n,):
            v = np.array([fun(a, b) for a, b in pts], dtype=float)
        return v
    v = np.asarray(fun, dtype=float)
    if v.shape == ():
        return np.full(n, float(v))
    if v.shape != (n,):
        raise ValueError(f"array of shape {v.shape} does not match {n} points")
    return v
