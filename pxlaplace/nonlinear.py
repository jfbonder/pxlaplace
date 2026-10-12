"""Pointwise (algebraic) nonlinear step of the decomposition-coordination method."""
import numpy as np


def solve_scalar(p, rhs, r=1.0, iters=70):
    """Solve  t^(p-1) + r t = rhs,  t >= 0   (vectorised, unique root).

    With r=1 and rhs=|eta+grad u| this is the equation in the paper
    (t = |nu|).  Robust bisection on [0, hi] with hi = min(rhs/r, rhs^(1/(p-1)))
    (both bounds hold for the root); the interval shrinks geometrically, so the
    *relative* accuracy is ~2^-iters even when t is tiny (singular case p<2).
    """
    p = np.asarray(p, float)
    rhs = np.asarray(rhs, float)
    p, rhs = np.broadcast_arrays(p, rhs)
    t = np.zeros_like(rhs)
    pos = rhs > 0
    if not pos.any():
        return t
    pp, rr = p[pos], rhs[pos]
    lo = np.zeros_like(rr)
    hi = np.minimum(rr / r, rr ** (1.0 / (pp - 1.0)))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = mid ** (pp - 1.0) + r * mid - rr
        neg = f < 0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    t[pos] = 0.5 * (lo + hi)
    return t


def update_nu(p, wx, wy, r=1.0):
    """Solve |nu|^(p-2) nu + r nu = r w   (w = eta/r + grad u) for nu, pointwise.

    Uses |nu| = t with t^(p-1) + r t = r |w| and nu = w * t/|w|  (no 0^(p-2)
    singularity when p < 2)."""
    mag = np.hypot(wx, wy)
    t = solve_scalar(p, r * mag, r)
    scale = np.divide(t, mag, out=np.zeros_like(mag), where=mag > 0)
    return wx * scale, wy * scale
