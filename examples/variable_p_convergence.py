"""Convergence in h for variable exponent p(x) with explicit solutions.

Cases
  smooth   : manufactured solution on the unit disk, p(x,y) = 2 + 0.8 x (crosses p = 2),
             u = exp(x/2)(1 + y/2); f = -div(|grad u|^{p-2} grad u) is computed symbolically (needs sympy).
  radial   : torsion problem -Delta_{p(r)} u = 1 on the unit disk with a radial exponent. The flux argument gives
             u'(r) = -(r/2)^{1/(p(r)-1)}, hence u(r) = int_r^1 (s/2)^{1/(p(s)-1)} ds, exact even if p jumps.
             --prad jump : p = 1.5 (r<0.5), 3 (r>=0.5)       --prad lin : p = 1.5 + 1.5 r
  example3 : Example 3 of the paper on (-1,1)^2 (p = 1+1/((x+y)/2+2), exact solution known) refined on squares.

Prints relative L2 / sup errors and the observed order t (least squares over the finest --fit meshes).

Run:  python examples/variable_p_convergence.py smooth|radial|example3 [--prad jump|lin] [--plot]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pxlaplace import disk_mesh, rectangle_mesh, PxLaplaceSolver, P1Space


# ---------------------------------------------------------------- problems
def smooth_problem():
    import sympy as sp
    x, y = sp.symbols("x y", real=True)
    p = 2 + sp.Rational(4, 5) * x
    u = sp.exp(x / 2) * (1 + y / 2)
    gx, gy = sp.diff(u, x), sp.diff(u, y)
    G = sp.sqrt(gx**2 + gy**2)
    f = -(sp.diff(G ** (p - 2) * gx, x) + sp.diff(G ** (p - 2) * gy, y))
    L = lambda e: sp.lambdify((x, y), e, "numpy")
    pf, uf, ff = L(p), L(u), L(f)
    return (lambda X, Y: pf(X, Y) + 0 * X), (lambda X, Y: uf(X, Y) + 0 * X), (lambda X, Y: ff(X, Y) + 0 * X)


def radial_problem(kind):
    if kind == "jump":
        pr = lambda r: np.where(r < 0.5, 1.5, 3.0)
    else:
        pr = lambda r: 1.5 + 1.5 * r
    s = np.linspace(0, 1, 2_000_001)
    # integrand (s/2)^{1/(p-1)}; the jump is located at s = 0.5 which is a grid point, so use left/right limits
    integ = (s / 2) ** (1 / (pr(s) - 1))
    cum = np.concatenate([[0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(s))])
    total = cum[-1]
    uref = lambda r: np.interp(np.clip(r, 0, 1), s, total - cum)         # u(r) = int_r^1
    u = lambda X, Y: uref(np.hypot(X, Y))
    p = lambda X, Y: pr(np.hypot(X, Y))
    return p, u, (lambda X, Y: np.ones_like(X))


def example3_problem():
    p = lambda x, y: 1 + 1 / ((x + y) / 2 + 2)
    u = lambda x, y: np.sqrt(2) * np.e**2 * (np.exp(0.5 * (x + y)) - 1)
    return p, u, (lambda x, y: np.zeros_like(x))


# ---------------------------------------------------------------- run
def run(mesh, p, u, f, tol, max_iter, f_degree=1):
    solver = PxLaplaceSolver(mesh, p, f, u, f_degree=f_degree)
    res = solver.solve(tol=tol, max_iter=max_iter, stop="rel_u")
    fem = P1Space(mesh)
    ue = u(*mesh.vertices.T)
    e = res.u - ue
    return dict(nodes=mesh.n_nodes, it=res.iterations, conv=bool(res.converged),
                L2=fem.norm_L2(e) / fem.norm_L2(ue), sup=fem.norm_inf(e) / fem.norm_inf(ue))


def order(h, err, k):
    return float(np.polyfit(np.log(h[-k:]), np.log(err[-k:]), 1)[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["smooth", "radial", "example3"])
    ap.add_argument("--prad", choices=["jump", "lin"], default="jump")
    ap.add_argument("--h", type=float, nargs="+", default=None, help="disk: edge lengths; example3: number of cells n")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--fit", type=int, default=3)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    if a.case == "smooth":
        p, u, f = smooth_problem(); hs = a.h or [0.2, 0.1, 0.05, 0.025, 0.0125]; fdeg = 2
    elif a.case == "radial":
        p, u, f = radial_problem(a.prad); hs = a.h or [0.2, 0.1, 0.05, 0.025, 0.0125]; fdeg = 1
    else:
        p, u, f = example3_problem(); hs = a.h or [10, 20, 40, 80, 160]; fdeg = 1
    rows = []
    print(f"{'h':>8} {'nodes':>7} {'iter':>5} {'conv':>5} {'L2 rel':>10} {'ord':>5} {'sup rel':>10} {'ord':>5}")
    for i, h in enumerate(hs):
        mesh = rectangle_mesh(int(h), int(h), -1, 1, -1, 1) if a.case == "example3" else disk_mesh(h)
        hh = 2.0 / h if a.case == "example3" else h
        r = run(mesh, p, u, f, a.tol, a.max_iter, fdeg); r["h"] = hh; rows.append(r)
        o = lambda k: f"{np.log(rows[-2][k] / r[k]) / np.log(rows[-2]['h'] / hh):5.2f}" if i else f"{'-':>5}"
        print(f"{hh:8.4f} {r['nodes']:7d} {r['it']:5d} {str(r['conv']):>5} {r['L2']:10.2e} {o('L2')} {r['sup']:10.2e} {o('sup')}", flush=True)
    H = np.array([r["h"] for r in rows])
    print(f"fitted order (finest {a.fit}): L2 {order(H, [r['L2'] for r in rows], a.fit):.2f}   sup {order(H, [r['sup'] for r in rows], a.fit):.2f}")
