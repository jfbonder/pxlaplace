"""p-torsion problem on the unit disk: numerical vs. explicit solution.

    -Delta_p u = 1 in B_1,   u = 0 on dB_1,
    u(x,y) = c_p (1 - r^{p/(p-1)}),   c_p = (p-1)/p * 2^{-1/(p-1)},   r = sqrt(x^2+y^2).

Prints a mesh-refinement table (errors and observed orders) for several exponents
and saves a figure (computed / exact / error / radial profile / convergence).

Run:  python examples/torsion_disk.py [--p 4] [--plot] [--h 0.2 0.1 0.05 0.025]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pxlaplace import disk_mesh, PxLaplaceSolver, P1Space


def exact(p):
    c = (p - 1) / p * 2 ** (-1.0 / (p - 1))
    return lambda x, y: c * (1 - (x**2 + y**2) ** (p / (2 * (p - 1))))


def run(p, h0, tol=1e-9):
    mesh = disk_mesh(h0)
    ue = exact(p)
    solver = PxLaplaceSolver(mesh, p, f=1.0, g=ue)         # Dirichlet data = exact solution (= 0)
    res = solver.solve(tol=tol, stop="p_error", max_iter=5000)
    fem = P1Space(mesh)
    e = res.u - ue(*mesh.vertices.T)
    return dict(mesh=mesh, res=res, ue=ue, h=h0, nodes=mesh.n_nodes, it=res.iterations,
                inf=fem.norm_inf(e), L2=fem.l2_error(res.u, ue, relative=False), umax=ue(0.0, 0.0), emax=e)


def order(a, b, ha, hb):
    return np.log(a / b) / np.log(ha / hb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=4.0, help="exponent shown in the figure")
    ap.add_argument("--ps", type=float, nargs="+", default=[1.5, 2.0, 4.0, 10.0])
    ap.add_argument("--h", type=float, nargs="+", default=[0.2, 0.1, 0.05, 0.025])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    results = {}
    for p in sorted(set(a.ps) | {a.p}):
        print(f"\np = {p}   (u(0) = {exact(p)(0., 0.):.5f})")
        print(f"{'h':>7} {'nodes':>7} {'iter':>5} {'nodal sup err':>13} {'order':>6} {'true L2 err':>12} {'order':>6}")
        rows = []
        for h in a.h:
            r = run(p, h)
            if rows:
                oi, o2 = order(rows[-1]["inf"], r["inf"], rows[-1]["h"], h), order(rows[-1]["L2"], r["L2"], rows[-1]["h"], h)
                s = f"{oi:6.2f} {r['L2']:12.3e} {o2:6.2f}"
            else:
                s = f"{'-':>6} {r['L2']:12.3e} {'-':>6}"
            print(f"{h:7.3f} {r['nodes']:7d} {r['it']:5d} {r['inf']:13.3e} " + s)
            rows.append(r)
        results[p] = rows

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        p = a.p; r = results[p][-1]; m = r["mesh"]; V = m.vertices
        tri = mtri.Triangulation(V[:, 0], V[:, 1], m.triangles)
        ue = r["ue"](V[:, 0], V[:, 1])
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        for k, (z, t) in enumerate(((r["res"].u, f"computed, p={p:g}"), (ue, "exact"), (r["emax"], "error u_h - u"))):
            c = ax.flat[k].tripcolor(tri, z, shading="gouraud", cmap="viridis" if k < 2 else "RdBu_r")
            fig.colorbar(c, ax=ax.flat[k]); ax.flat[k].set_aspect("equal"); ax.flat[k].set_title(t)
        rr = np.linspace(0, 1, 400)
        ax[1, 0].plot(rr, r["ue"](rr, 0 * rr), "k-", label="exact")
        ang = np.hypot(V[:, 0], V[:, 1]); ax[1, 0].plot(ang, r["res"].u, ".", ms=2, alpha=.5, label="computed (all nodes)")
        ax[1, 0].set_xlabel("r"); ax[1, 0].set_title("radial profile"); ax[1, 0].legend()
        for pp, rows in results.items():
            hs = [q["h"] for q in rows]
            ax[1, 1].loglog(hs, [q["inf"] / q["umax"] for q in rows], "o-", label=f"p={pp:g}")
        hs = np.array(a.h); ax[1, 1].loglog(hs, 0.5 * hs**2, "k--", label=r"$h^2$")
        ax[1, 1].set_xlabel("h"); ax[1, 1].set_title(r"relative $L^\infty$ error"); ax[1, 1].legend()
        hist = r["res"].history
        ax[1, 2].semilogy(hist["p_error"], label="p-error"); ax[1, 2].semilogy(hist["residual"], label=r"$\|\nabla u-\nu\|_{L^2}$")
        ax[1, 2].set_xlabel("iteration"); ax[1, 2].set_title("convergence of the iteration"); ax[1, 2].legend()
        plt.tight_layout(); plt.savefig("torsion_disk.png", dpi=110)
        print("\nsaved torsion_disk.png")
