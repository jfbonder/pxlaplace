"""Diagnostic: how the exponent p is sampled on each triangle (discontinuous radial exponent).

Radial torsion problem with p = 1.5 (r < 0.5), 3 (r >= 0.5) (exact solution, see variable_p_convergence.py).
Compares  bary   : p at the barycentre (default of the package)
          subavg : arithmetic mean of p over an 8x8 barycentric sub-grid of the triangle
          harm   : harmonic mean of p (same sub-grid)
and prints the true relative L2 error and the nodal sup error.

Run:  python examples/p_sampling_diagnostic.py [--h 0.1 0.05 0.025 0.0125]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from variable_p_convergence import radial_problem
from pxlaplace import disk_mesh, PxLaplaceSolver, P1Space


def sub_average(mesh, p, k=8):
    V = mesh.vertices[mesh.triangles]
    out, cnt = np.zeros(len(V)), 0
    for i in range(k + 1):
        for j in range(k + 1 - i):
            l = np.array([i, j, k - i - j]) / k
            P = np.einsum("i,tij->tj", l, V)
            out += p(P[:, 0], P[:, 1]); cnt += 1
    return out / cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, nargs="+", default=[0.1, 0.05, 0.025, 0.0125])
    a = ap.parse_args()
    p, u, f = radial_problem("jump")
    for h in a.h:
        m = disk_mesh(h); fem = P1Space(m); ue = u(*m.vertices.T); row = []
        for name, pp in (("bary", p), ("subavg", sub_average(m, p)), ("harm", 1 / sub_average(m, lambda x, y: 1 / p(x, y)))):
            r = PxLaplaceSolver(m, pp, 1.0, u).solve(tol=1e-10, max_iter=4000)
            row.append((name, fem.l2_error(r.u, u), fem.norm_inf(r.u - ue) / fem.norm_inf(ue)))
        print(f"h={h:7.4f} " + " | ".join(f"{n}: L2 {e:.2e} sup(nodes) {s:.2e}" for n, e, s in row), flush=True)
