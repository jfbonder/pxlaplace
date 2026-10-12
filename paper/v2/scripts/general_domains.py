"""Qualitative examples on non-rectangular domains (figure general_domains.png).

(a) L-shaped domain (polygon_mesh / Triangle), torsion -Delta_p u = 1, u = 0, p = 3.
(b) Annulus 0.5 < |x| < 2 (polygon with a hole), -Delta_{p(x)} u = 1, u = 0, p(x) = 1.5 + 0.5|x|.
(c) Union of two overlapping discs (signed distance function / sdf_mesh), torsion with
    discontinuous exponent p = 1.3 (x < 0), p = 4 (x >= 0).
Run from paper_v2/:  python new/scripts/general_domains.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "repo"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pxlaplace import polygon_mesh, sdf_mesh, PxLaplaceSolver

cases = []
# (a) L-shape
L = np.array([[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]], float)
mA = polygon_mesh(L, max_area=4e-4)
cases.append(("(a) L-shaped domain, $p=3$", mA, 3.0))
# (b) annulus
th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
mB = polygon_mesh(np.c_[2 * np.cos(th), 2 * np.sin(th)], holes=[np.c_[.5 * np.cos(th), .5 * np.sin(th)]],
                  max_area=1.5e-3)
cases.append(("(b) annulus, $p=1.5+0.5|x|$", mB, lambda x, y: 1.5 + 0.5 * np.hypot(x, y)))
# (c) union of two discs
sdf = lambda P: np.minimum(np.hypot(P[:, 0] + 0.6, P[:, 1]) - 1.0, np.hypot(P[:, 0] - 0.6, P[:, 1]) - 1.0)
mC = sdf_mesh(sdf, (-1.6, -1.0, 1.6, 1.0), 0.03)
cases.append(("(c) union of two discs, $p=1.3$ / $p=4$", mC, lambda x, y: np.where(x < 0, 1.3, 4.0)))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, (title, m, p) in zip(axs, cases):
    res = PxLaplaceSolver(m, p, 1.0, 0.0).solve(tol=1e-8, max_iter=20000)
    print(f"{title}: nodes={m.n_nodes}, elements={m.n_elems}, iterations={res.iterations}, "
          f"converged={res.converged}, max u={res.u.max():.4e}, min u={res.u.min():.2e}")
    tri = mtri.Triangulation(m.vertices[:, 0], m.vertices[:, 1], m.triangles)
    c = ax.tripcolor(tri, res.u, shading="gouraud")
    ax.tricontour(tri, res.u, levels=10, colors="k", linewidths=0.4)
    fig.colorbar(c, ax=ax, shrink=0.85)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
plt.tight_layout()
out = os.path.join(HERE, "..", "figures", "general_domains.png")
plt.savefig(out, dpi=130)
print("saved", out)
