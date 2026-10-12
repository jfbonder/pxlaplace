"""Examples 1-5 of the paper (rectangular domains), plus error tables.

Run:  python examples/paper_examples.py [--plot] [--n 100]
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pxlaplace import rectangle_mesh, PxLaplaceSolver, P1Space


def report(name, mesh, res, ue=None, p=None):
    fem = P1Space(mesh)
    s = f"{name}: {res.iterations} it, converged={res.converged}, {res.time_iter:.2f}s (+{res.time_setup:.2f}s setup)"
    if ue is not None:
        e = res.u - ue
        s += f"\n   ||u-uh||_inf = {fem.norm_inf(e):.2e},  ||u-uh||_p = {fem.norm_Lp(e, p):.2e}"
    print(s)


def example1(n=100, tol=1e-6, **kw):
    p = 20.0
    ex = lambda x, y: (x**2 + y**2) ** ((p - 2) / (2 * p - 2))
    m = rectangle_mesh(n, n, 0, 1, 0, 1)
    res = PxLaplaceSolver(m, p, 0.0, ex).solve(tol=tol, **kw)
    return m, res, ex(*m.vertices.T), p


def example2(n=100, tol=1e-8, **kw):
    p = 1.1
    cp = (p - 1) / p * 2 ** (-1 / (p - 1))
    ex = lambda x, y: cp * (1 - (x**2 + y**2) ** (p / (2 * p - 2)))
    a = 1 / np.sqrt(2)
    m = rectangle_mesh(n, n, -a, a, -a, a)
    res = PxLaplaceSolver(m, p, 1.0, ex).solve(tol=tol, **{'stop': 'p_error', **kw})
    return m, res, ex(*m.vertices.T), p


def example3(n=100, tol=1e-7, **kw):
    pf = lambda x, y: 1 + 1 / ((x + y) / 2 + 2)
    ex = lambda x, y: np.sqrt(2) * np.e**2 * (np.exp(0.5 * (x + y)) - 1)
    m = rectangle_mesh(n, n, -1, 1, -1, 1)
    res = PxLaplaceSolver(m, pf, 0.0, ex).solve(tol=tol, **kw)
    return m, res, ex(*m.vertices.T), PxLaplaceSolver(m, pf).p


def example4(n=100, ps=(1.15, 4.0, 50.0), tol=1e-6, **kw):
    m = rectangle_mesh(n, n, -1, 1, -1, 1)
    return m, {p: PxLaplaceSolver(m, p, 1.0, 0.0).solve(tol=tol, **kw) for p in ps}


def example5(n=100, tol=1e-6, **kw):
    pf = lambda x, y: np.where(x <= 0, 1.2, 4.0)
    m = rectangle_mesh(2 * n, n, -2, 2, -1, 1)
    return m, PxLaplaceSolver(m, pf, 1.0, 0.0).solve(tol=tol, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    out = {}
    for k, ex in (("Example 1 (p=20)", example1), ("Example 2 (p=1.1 torsion)", example2),
                  ("Example 3 (variable p)", example3)):
        m, res, ue, p = ex(a.n)
        report(k, m, res, ue, p); out[k] = (m, res)
    m, rs = example4(a.n)
    for p, res in rs.items():
        report(f"Example 4 (torsion, p={p})", m, res); out[f"Example 4 p={p}"] = (m, res)
    m, res = example5(a.n)
    report("Example 5 (discontinuous p)", m, res); out["Example 5"] = (m, res)
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 4, figsize=(18, 8))
        for ax, (k, (m, res)) in zip(axs.ravel(), out.items()):
            m.plot(res.u, ax=ax); ax.set_title(k, fontsize=9)
        for ax in axs.ravel()[len(out):]: ax.axis("off")
        plt.tight_layout(); plt.savefig("paper_examples.png", dpi=110)
