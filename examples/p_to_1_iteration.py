"""Behaviour of the iteration as p -> 1 (and p large) on the unit disk.

Problem: -Delta_p u = 2 in B_1, u = exact on dB_1, u = (p-1)/p (1 - r^{p/(p-1)}).
For each p we run a fixed number of iterations and record
  * rel_u   = ||u_n - u_{n-1}||_2 / ||u_{n-1}||_2        (stopping criterion of the MatLab code)
  * res     = ||grad u_n - nu_n||_{L2} / ||grad u_n||_{L2}  (primal residual)
  * rho     = linear contraction factor of `res` (fit of log res on the window 1e-6 < res < 1e-3)
  * relative sup-error w.r.t. the exact solution

Run:  python examples/p_to_1_iteration.py --plot [--h 0.05] [--max-iter 6000]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pxlaplace import disk_mesh, PxLaplaceSolver


def exact(p, lam=2.0):
    k = (lam / 2) ** (1 / (p - 1))
    return lambda x, y: k * (p - 1) / p * (1 - (x * x + y * y) ** (p / (2 * (p - 1))))


def rho_fit(h, lo=1e-6, hi=1e-3):
    h = np.asarray(h); i = np.where((h < hi) & (h > lo))[0]
    return float(np.exp(np.polyfit(i, np.log(h[i]), 1)[0])) if len(i) >= 8 else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=float, nargs="+", default=[6, 4, 3, 2.5, 2, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05])
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=6000)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    m = disk_mesh(a.h)
    rows, hist = [], {}
    print(f"disk, h={a.h}, {m.n_nodes} nodes")
    print(f"{'p':>5} {'it(rel_u<1e-8)':>15} {'res@stop':>9} {'res@end':>9} {'rho(res)':>9} {'rel sup-err':>11}")
    for p in a.ps:
        ue = exact(p); s = PxLaplaceSolver(m, p, 2.0, ue)
        res = s.solve(tol=0, max_iter=a.max_iter)
        h = res.history; gx, gy = s.fem.gradient(res.u)
        gn = np.sqrt(np.sum(s.fem.area * (gx**2 + gy**2))); r = h["residual"] / gn
        k = np.where(h["rel_u"] < 1e-8)[0]; ks = int(k[0]) if len(k) else None
        err = np.abs(res.u - ue(*m.vertices.T)).max() / ue(0.0, 0.0)
        rows.append((p, ks + 1 if ks is not None else None, r[ks] if ks is not None else np.nan, r[-1], rho_fit(r), err))
        hist[p] = (h["rel_u"], r)
        print(f"{p:5g} {str(rows[-1][1]):>15} {rows[-1][2]:9.1e} {r[-1]:9.1e} {rows[-1][4]:9.4f} {err:11.2e}")
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
        for p, (ru, r) in hist.items():
            if p in (3, 1.5, 1.3, 1.1, 1.05):
                ax[0].semilogy(ru, label=f"p={p:g}"); ax[1].semilogy(r, label=f"p={p:g}")
        ax[0].set_title("rel_u"); ax[1].set_title("primal residual")
        for x in ax[:2]: x.legend(); x.set_xlabel("iteration")
        ps = np.array([q[0] for q in rows]); rho = np.array([q[4] for q in rows]); o = np.argsort(ps)
        ax[2].plot(ps[o], rho[o], "o-"); ax[2].set_xscale("log")
        ax[2].set_xlabel("p"); ax[2].set_title(r"contraction factor $\rho(p)$ of the residual (p=2: 1/2)")
        plt.tight_layout(); plt.savefig("p_to_1_iteration.png", dpi=110)
