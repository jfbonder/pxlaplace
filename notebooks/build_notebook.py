"""Generates notebooks/pxlaplace_colab.ipynb (self-contained: it embeds the package source)."""
import nbformat as nbf, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# p(x)-Laplacian solver — decomposition-coordination method on general 2D domains

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfbonder/pxlaplace/blob/main/notebooks/pxlaplace_colab.ipynb)

Python port (vectorised, any triangulated domain) of the MatLab code of
A. Aragón, J. Fernández Bonder, D. Rubio, *Effective numerical computation of p(x)-Laplace equations in 2D*.

We solve
$$-\\operatorname{div}\\big(|\\nabla u|^{p(x)-2}\\nabla u\\big)=f \\ \\text{in }\\Omega,\\qquad u=g\\ \\text{on }\\partial\\Omega$$
with the iteration (r = 1 in the paper):

1. $r\\,(-\\Delta u_n)=f+\\operatorname{div}(\\eta_n-r\\nu_{n-1})$  (linear, same matrix at every step → factorised once),
2. $|\\nu_n|^{p-2}\\nu_n+r\\nu_n=\\eta_n+r\\nabla u_n$  (pointwise scalar equation),
3. $\\eta_{n+1}=\\eta_n+r(\\nabla u_n-\\nu_n)$.

**Run all cells.** The first cells write the package (`pxlaplace/`) to the Colab disk; the rest are examples.
""")
md("## 1. Package source (self-contained)")
code("!pip -q install triangle meshio 2>&1 | tail -1\n!mkdir -p pxlaplace")
for name in ("__init__", "mesh", "fem", "nonlinear", "solver"):
    src = (root / "pxlaplace" / f"{name}.py").read_text()
    code(f"%%writefile pxlaplace/{name}.py\n{src}")
code("import sys; sys.path.insert(0, '.')\nimport numpy as np, matplotlib.pyplot as plt\nfrom pxlaplace import *\nprint('ok')")

md("## 2. Quick start (rectangle, Example 3 of the paper: variable exponent, known exact solution)")
code("""
pf = lambda x, y: 1 + 1/((x + y)/2 + 2)                       # p(x,y)
ex = lambda x, y: np.sqrt(2)*np.e**2*(np.exp(0.5*(x + y)) - 1) # exact solution (and Dirichlet data)

mesh = rectangle_mesh(100, 100, -1, 1, -1, 1)                  # n x m cells, 2 triangles each
solver = PxLaplaceSolver(mesh, p=pf, f=0.0, g=ex)              # set-up + factorisation (once)
res = solver.solve(tol=1e-7)                                   # iterate

fem = P1Space(mesh)
e = res.u - ex(*mesh.vertices.T)
print(f"{res.iterations} iterations, converged={res.converged}, {res.time_iter:.1f}s")
print("||u-uh||_inf =", fem.norm_inf(e), "  ||u-uh||_p =", fem.norm_Lp(e, solver.p))
mesh.plot(res.u); plt.title("computed solution"); plt.show()
""")
md("""
### Diagnostics
`res.history` stores, per iteration: `rel_u` (relative change of $u$, the stopping criterion of the MatLab code),
`p_error` (the $p$-modular error of the paper), `residual` ($\\|\\nabla u_n-\\nu_n\\|_{L^2}$) and `energy`.
""")
code("""
h = res.history
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].semilogy(h['rel_u'], label='rel_u'); ax[0].semilogy(h['p_error'], label='p-error'); ax[0].legend(); ax[0].set_title('stopping criteria')
ax[1].semilogy(h['residual']); ax[1].set_title(r'$\\|\\nabla u-\\nu\\|_{L^2}$')
ax[2].plot(h['energy']); ax[2].set_title('energy'); plt.show()
""")

md("## 3. Paper examples (rectangles)")
code("""
# Example 1: p = 20, u = (x^2+y^2)^((p-2)/(2p-2)) is an exact p-harmonic function on (0,1)^2
p = 20.0
ex1 = lambda x, y: (x**2 + y**2)**((p-2)/(2*p-2))
m = rectangle_mesh(100, 100)
r1 = PxLaplaceSolver(m, p, 0.0, ex1).solve(tol=1e-6)
e = r1.u - ex1(*m.vertices.T); F = P1Space(m)
print(r1.iterations, 'it;', f'{r1.time_iter:.1f}s;', 'inf err', F.norm_inf(e), 'p err', F.norm_Lp(e, p))
m.plot(r1.u); plt.title('Example 1'); plt.show()
""")
code("""
# Example 4: torsion problem -Delta_p u = 1 on (-1,1)^2 for several p
m = rectangle_mesh(100, 100, -1, 1, -1, 1)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, p in zip(axs, (1.15, 4.0, 50.0)):
    r = PxLaplaceSolver(m, p, 1.0, 0.0).solve(tol=1e-6)
    m.plot(r.u, ax=ax); ax.set_title(f'torsion, p={p} ({r.iterations} it)')
plt.show()
""")
code("""
# Example 5: discontinuous exponent, Omega = [-2,2]x[-1,1]
pf5 = lambda x, y: np.where(x <= 0, 1.2, 4.0)
m = rectangle_mesh(200, 100, -2, 2, -1, 1)
r5 = PxLaplaceSolver(m, pf5, 1.0, 0.0).solve(tol=1e-6)
m.plot(r5.u); plt.title(f'Example 5 ({r5.iterations} it)'); plt.show()
""")

md("""
## 4. General domains
Only a triangulation is needed: `Mesh(vertices, triangles, dirichlet)`. Built-in ways to get one:

* `rectangle_mesh(n, m, xmin, xmax, ymin, ymax)`
* `polygon_mesh(outer, holes=[...], max_area=...)` — polygons with holes (uses `triangle`)
* `sdf_mesh(sdf, bbox, h0)` / `disk_mesh(h0)` — any domain given by a signed-distance function (no external dependency)
* `from_meshio(file)` (Gmsh `.msh`, VTK, …) and `from_txt(...)` (the 3 text files of the MatLab code)
""")
code("""
# Disk, torsion problem with p = 4 (exact solution known)
p = 4.0; c = (p-1)/p*2**(-1/(p-1))
ex = lambda x, y: c*(1 - (x**2 + y**2)**(p/(2*p-2)))
md_ = disk_mesh(0.03)
r = PxLaplaceSolver(md_, p, 1.0, 0.0).solve(tol=1e-8, stop='p_error')
print(md_.n_nodes, 'nodes;', r.iterations, 'it; inf error', np.abs(r.u - ex(*md_.vertices.T)).max())
md_.plot(r.u); plt.title('torsion on the disk, p=4'); plt.show()
""")
code("""
# L-shaped domain, variable exponent p(x,y) = 1.5 + x/2
L = [(0,0),(2,0),(2,1),(1,1),(1,2),(0,2)]
mL = polygon_mesh(L, max_area=0.001)
rL = PxLaplaceSolver(mL, lambda x, y: 1.5 + x/2, 1.0, 0.0).solve(tol=1e-7)
mL.plot(rL.u); plt.title(f'L-shape, p(x,y)=1.5+x/2 ({rL.iterations} it)'); plt.show()
""")
code("""
# Annulus with a hole, Dirichlet data 1 on the inner circle and 0 on the outer one (p = 3)
th = np.linspace(0, 2*np.pi, 100, endpoint=False)
outer = np.c_[2*np.cos(th), 2*np.sin(th)]; hole = np.c_[0.5*np.cos(th), 0.5*np.sin(th)]
mA = polygon_mesh(outer, holes=[hole], max_area=0.002)
gA = lambda x, y: np.where(x**2 + y**2 < 1, 1.0, 0.0)
rA = PxLaplaceSolver(mA, 3.0, 0.0, gA).solve(tol=1e-8)
mA.plot(rA.u); plt.title(f'annulus, p=3 ({rA.iterations} it)'); plt.show()
""")
code("""
# Any domain through a signed-distance function: a peanut-shaped region
def sdf(P):
    d1 = np.hypot(P[:,0] + 0.5, P[:,1]) - 0.8
    d2 = np.hypot(P[:,0] - 0.5, P[:,1]) - 0.8
    return np.minimum(d1, d2)                      # union of two discs
mS = sdf_mesh(sdf, (-1.4, -0.9, 1.4, 0.9), h0=0.03)
rS = PxLaplaceSolver(mS, lambda x, y: 1.3 + 3*(x > 0), 1.0, 0.0).solve(tol=1e-7)
mS.plot(rS.u); plt.title(f'union of discs, p = 1.3 | 4.3 ({rS.iterations} it)'); plt.show()
""")
md("""
## 5. Using your own mesh
```python
mesh = Mesh(vertices, triangles, dirichlet_node_indices)     # numpy arrays (0-based)
mesh = from_txt('vertex_coordinates.txt', 'elem_vertices.txt', 'dirichlet.txt')   # MatLab files (1-based)
mesh = from_meshio('domain.msh')                              # Gmsh etc.; boundary detected automatically
```
`p`, `f`, `g` can be constants, arrays, or functions `fun(x, y)` (vectorised with NumPy; scalar-only functions also work, just slower).
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["colab"] = {"provenance": []}
out = root / "notebooks" / "pxlaplace_colab.ipynb"
nbf.write(nb, out); print("wrote", out)
