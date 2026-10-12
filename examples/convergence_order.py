"""Order of convergence in h of the p-torsion problem on the unit disk, as a function of p.

    -Delta_p u = 1 in B_1,  u = 0 on dB_1,   u = c_p (1 - r^{p/(p-1)}),  c_p = (p-1)/p * 2^{-1/(p-1)}.

For every p and every mesh size h the problem is solved to convergence and the relative errors
    ||u-u_h||_{L2},  ||u-u_h||_{Lp},  ||grad(u-u_h)||_{Lp}
are computed.  The observed order t in  err ~ C h^t  is the least-squares slope of log(err) vs log(h)
over the finest `--fit` meshes (consecutive-pair orders are printed too).

Run:  python examples/convergence_order.py --plot            (default: several p, h = 0.2 ... 0.025)
      python examples/convergence_order.py --ps 1.3 2 4 --h 0.1 0.05 0.025 0.0125 --jobs 2
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pxlaplace import disk_mesh, PxLaplaceSolver, P1Space


def exact(p):
    c = (p - 1) / p * 2 ** (-1.0 / (p - 1))
    u = lambda x, y: c * (1 - (x**2 + y**2) ** (p / (2 * (p - 1))))
    def grad(x, y):                                   # grad u = -(r/2)^{1/(p-1)} (x,y)/r
        r = np.maximum(np.hypot(x, y), 1e-300)
        m = -((r / 2) ** (1 / (p - 1)))
        return m * x / r, m * y / r
    return u, grad


def one(args):
    p, h, tol, max_iter, seed = args
    mesh = disk_mesh(h, seed=seed)
    u, grad = exact(p)
    solver = PxLaplaceSolver(mesh, p, f=1.0, g=u)
    res = solver.solve(tol=tol, max_iter=max_iter, stop="rel_u")
    fem = P1Space(mesh)
    ue = u(*mesh.vertices.T)
    e = res.u - ue
    gx, gy = fem.gradient(res.u)
    ex, ey = grad(*fem.bary.T)
    egrad = np.hypot(gx - ex, gy - ey)
    rel = lambda a, b: a / b
    return dict(p=p, h=h, nodes=mesh.n_nodes, it=res.iterations, converged=bool(res.converged),
                resid=float(res.history["residual"][-1]),
                L2=rel(fem.norm_L2(e), fem.norm_L2(ue)),
                Lp=rel(fem.modular(e, p) ** (1 / p), fem.modular(ue, p) ** (1 / p)),
                gradLp=float((np.sum(fem.area * egrad**p) / np.sum(fem.area * np.hypot(ex, ey) ** p)) ** (1 / p)),
                hmax=float(np.max(np.linalg.norm(mesh.vertices[mesh.triangles][:, [0, 1, 2]] - mesh.vertices[mesh.triangles][:, [1, 2, 0]], axis=2))))


def fit_order(h, err, k):
    h, err = np.asarray(h)[-k:], np.asarray(err)[-k:]
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=float, nargs="+", default=[1.2, 1.3, 1.5, 1.75, 2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--h", type=float, nargs="+", default=[0.2, 0.1, 0.05, 0.025])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--fit", type=int, default=3, help="number of finest meshes used for the slope")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="convergence_order.json")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    tasks = [(p, h, a.tol, a.max_iter, 0) for p in a.ps for h in a.h]
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as pool:
            out = pool.map(one, tasks, chunksize=1)
    else:
        out = [one(t) for t in tasks]
    json.dump(out, open(a.out, "w"), indent=1)

    summary = {}
    for p in a.ps:
        R = [r for r in out if r["p"] == p]
        print(f"\np = {p:g}")
        print(f"{'h':>7} {'nodes':>7} {'iter':>5} {'conv':>5} {'L2 rel':>10} {'ord':>5} {'Lp rel':>10} {'ord':>5} {'gradLp rel':>11} {'ord':>5}")
        for i, r in enumerate(R):
            def o(key):
                return f"{np.log(R[i-1][key] / r[key]) / np.log(R[i-1]['h'] / r['h']):5.2f}" if i else f"{'-':>5}"
            print(f"{r['h']:7.4f} {r['nodes']:7d} {r['it']:5d} {str(r['converged']):>5} {r['L2']:10.2e} {o('L2')} {r['Lp']:10.2e} {o('Lp')} {r['gradLp']:11.2e} {o('gradLp')}")
        hs = [r["h"] for r in R]
        summary[p] = {k: fit_order(hs, [r[k] for r in R], a.fit) for k in ("L2", "Lp", "gradLp")}
        print("   fitted orders (finest %d meshes): L2 %.2f   Lp %.2f   grad-Lp %.2f" % (a.fit, *summary[p].values()))

    print("\nSummary  t(p):")
    print(f"{'p':>6} {'t_L2':>6} {'t_Lp':>6} {'t_gradLp':>9}")
    for p, s in summary.items():
        print(f"{p:6g} {s['L2']:6.2f} {s['Lp']:6.2f} {s['gradLp']:9.2f}")

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ps = list(summary)
        for k, lab in (("L2", r"$L^2$"), ("Lp", r"$L^p$"), ("gradLp", r"$\|\nabla(u-u_h)\|_{L^p}$")):
            ax[0].plot(ps, [summary[p][k] for p in ps], "o-", label=lab)
        ax[0].axhline(2, color="gray", ls=":"); ax[0].axhline(1, color="gray", ls=":")
        ax[0].set_xlabel("p"); ax[0].set_ylabel("observed order t"); ax[0].set_title(r"$\|u-u_h\|\sim h^t$ (unit disk, torsion)"); ax[0].legend()
        for p in ps:
            R = [r for r in out if r["p"] == p]
            ax[1].loglog([r["h"] for r in R], [r["L2"] for r in R], "o-", label=f"p={p:g}")
        hh = np.array(a.h); ax[1].loglog(hh, 0.5 * hh**2, "k--", label=r"$h^2$")
        ax[1].set_xlabel("h"); ax[1].set_title(r"relative $L^2$ error"); ax[1].legend(fontsize=7)
        plt.tight_layout(); plt.savefig("convergence_order.png", dpi=110)
