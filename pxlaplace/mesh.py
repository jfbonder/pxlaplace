"""Triangular meshes for arbitrary 2D domains.

A :class:`Mesh` only needs ``vertices (Nn,2)``, ``triangles (Nt,3)`` and the
indices of the Dirichlet (boundary) nodes -- exactly the three arrays the
MatLab code reads (vertex_coordinates, elem_vertices, dirichlet).  Everything
else in the package is domain independent, so any mesher can be plugged in.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray          # (Nn, 2) float
    triangles: np.ndarray         # (Nt, 3) int, counter-clockwise
    dirichlet: np.ndarray         # (Nd,)   int, boundary node indices

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=float)
        self.triangles = np.ascontiguousarray(self.triangles, dtype=np.int64)
        self.dirichlet = np.unique(np.asarray(self.dirichlet, dtype=np.int64))
        self._orient()

    def _orient(self):
        """Make every triangle counter-clockwise (positive area)."""
        v = self.vertices[self.triangles]
        d = ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
             - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
        flip = d < 0
        if flip.any():
            self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]

    @property
    def n_nodes(self): return self.vertices.shape[0]
    @property
    def n_elems(self): return self.triangles.shape[0]

    def plot(self, u=None, ax=None, **kw):
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        tri = mtri.Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.triangles)
        if ax is None:
            ax = plt.gca()
        if u is None:
            ax.triplot(tri, lw=0.4, **kw)
        else:
            c = ax.tripcolor(tri, u, shading="gouraud", **kw)
            ax.figure.colorbar(c, ax=ax)
        ax.set_aspect("equal")
        return ax


def boundary_nodes_from_triangles(triangles) -> np.ndarray:
    """Nodes on edges that belong to exactly one triangle (the boundary)."""
    t = np.asarray(triangles)
    edges = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(uniq[counts == 1])


# --------------------------------------------------------------------------
# Rectangle (same layout as gen_mesh_rectangle.m, but in memory)
# --------------------------------------------------------------------------
def rectangle_mesh(n: int, m: int | None = None, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0) -> Mesh:
    """Uniform mesh of [xmin,xmax]x[ymin,ymax]: n x m cells, 2 triangles each."""
    m = n if m is None else m
    x = np.linspace(xmin, xmax, n + 1)
    y = np.linspace(ymin, ymax, m + 1)
    X, Y = np.meshgrid(x, y)                      # row i -> y_i, col j -> x_j
    V = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange((n + 1) * (m + 1)).reshape(m + 1, n + 1)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    T = np.vstack([np.column_stack([a, b, d]), np.column_stack([b, c, d])])
    bd = np.concatenate([idx[0, :], idx[-1, :], idx[:, 0], idx[:, -1]])
    return Mesh(V, T, bd)


# --------------------------------------------------------------------------
# Polygon (with holes) via Shewchuk's Triangle (optional dependency)
# --------------------------------------------------------------------------
def polygon_mesh(outer, holes=(), hole_points=(), max_area=None, min_angle=25.0) -> Mesh:
    """Quality triangulation of a polygon with optional polygonal holes.

    outer       : (k,2) vertices of the outer boundary (any orientation)
    holes       : iterable of (k_i,2) arrays, boundaries of the holes
    hole_points : (not needed; a point inside every hole is computed automatically)
    max_area    : maximum triangle area (controls resolution)
    Requires the ``triangle`` package (pip install triangle).
    """
    try:
        import triangle as tr
    except ImportError as e:                                   # pragma: no cover
        raise ImportError("polygon_mesh needs `pip install triangle`; "
                          "alternatively use sdf_mesh()") from e
    outer = np.asarray(outer, float)
    pts, segs, hpts = [outer], [], []
    off = len(outer)
    segs.append(np.column_stack([np.arange(off), (np.arange(off) + 1) % off]))
    for h in holes:
        h = np.asarray(h, float)
        k = len(h)
        segs.append(np.column_stack([off + np.arange(k), off + (np.arange(k) + 1) % k]))
        pts.append(h)
        hpts.append(_interior_point(h))
        off += k
    A = dict(vertices=np.vstack(pts), segments=np.vstack(segs))
    if hpts:
        A["holes"] = np.array(hpts)
    opts = f"pq{min_angle:g}"
    if max_area is not None:
        opts += f"a{max_area:.10g}"
    B = tr.triangulate(A, opts)
    T = B["triangles"]
    return Mesh(B["vertices"], T, boundary_nodes_from_triangles(T))


def _interior_point(poly):
    """A point strictly inside a simple polygon (centroid of an ear)."""
    n = len(poly)
    from matplotlib.path import Path
    path = Path(poly)
    c = poly.mean(axis=0)
    if path.contains_point(c):
        return c
    for i in range(n):
        tri = poly[[i, (i + 1) % n, (i + 2) % n]]
        p = tri.mean(axis=0)
        if path.contains_point(p):
            return p
    raise ValueError("could not find an interior point of the hole")


# --------------------------------------------------------------------------
# Implicit domains via signed distance function (DistMesh-lite, numpy/scipy)
# --------------------------------------------------------------------------
def sdf_mesh(sdf, bbox, h0, fh=None, seed=0, max_iter=300, fixed=None, verbose=False) -> Mesh:
    """Mesh {sdf<0} with a simplified DistMesh (Persson & Strang, 2004).

    sdf  : callable (N,2)->(N,) signed distance, negative inside
    bbox : (xmin, ymin, xmax, ymax)
    h0   : target edge length
    fh   : optional callable (N,2)->(N,) relative size function (default 1)
    """
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bbox
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    fh = fh or (lambda p: np.ones(len(p)))
    # hexagonal initial grid
    xs = np.arange(xmin, xmax + h0, h0)
    ys = np.arange(ymin, ymax + h0 * np.sqrt(3) / 2, h0 * np.sqrt(3) / 2)
    X, Y = np.meshgrid(xs, ys)
    X[1::2, :] += h0 / 2
    p = np.column_stack([X.ravel(), Y.ravel()])
    p = p[sdf(p) < geps]
    r0 = 1.0 / fh(p) ** 2
    p = p[rng.random(len(p)) < r0 / r0.max()]
    fixed = np.zeros((0, 2)) if fixed is None else np.asarray(fixed, float)
    if len(fixed):
        p = np.vstack([fixed, p])
    nfix = len(fixed)
    pold = np.full_like(p, np.inf)
    Fscale, dt, ttol = 1.2, 0.2, 0.1
    for it in range(max_iter):
        if np.max(np.linalg.norm(p - pold, axis=1) / h0) > ttol:
            pold = p.copy()
            tri = Delaunay(p).simplices
            cen = p[tri].mean(axis=1)
            tri = tri[sdf(cen) < -geps]
            bars = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
        bv = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.linalg.norm(bv, axis=1)
        hb = fh(p[bars].mean(axis=1))
        L0 = hb * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hb ** 2))
        F = np.maximum(L0 - L, 0)
        Fv = (F / np.maximum(L, 1e-300))[:, None] * bv
        Ftot = np.zeros_like(p)
        np.add.at(Ftot, bars[:, 0], Fv)
        np.add.at(Ftot, bars[:, 1], -Fv)
        Ftot[:nfix] = 0
        p = p + dt * Ftot
        d = sdf(p)
        out = d > 0                                   # project back onto boundary
        if out.any():
            q = p[out]
            dx = (sdf(q + [deps, 0]) - d[out]) / deps
            dy = (sdf(q + [0, deps]) - d[out]) / deps
            p[out] = q - np.column_stack([d[out] * dx, d[out] * dy])
        inside = d < -geps
        if it > 0 and np.max(np.linalg.norm(dt * Ftot[inside], axis=1) / h0, initial=0.0) < 1e-3:
            if verbose:
                print("sdf_mesh converged at iteration", it)
            break
    tri = Delaunay(p).simplices
    tri = tri[sdf(p[tri].mean(axis=1)) < -geps]
    # drop unused nodes
    used = np.unique(tri)
    remap = -np.ones(len(p), int); remap[used] = np.arange(len(used))
    p, tri = p[used], remap[tri]
    return Mesh(p, tri, boundary_nodes_from_triangles(tri))


def disk_mesh(h0=0.05, center=(0.0, 0.0), radius=1.0, **kw) -> Mesh:
    c = np.asarray(center, float)
    sdf = lambda p: np.linalg.norm(p - c, axis=1) - radius
    return sdf_mesh(sdf, (c[0] - radius, c[1] - radius, c[0] + radius, c[1] + radius), h0, **kw)


# --------------------------------------------------------------------------
# Importers
# --------------------------------------------------------------------------
def from_txt(vertex_file, elem_file, dirichlet_file=None, one_based=True) -> Mesh:
    """Read the three text files of the MatLab code (1-based indices by default)."""
    V = np.loadtxt(vertex_file)
    T = np.loadtxt(elem_file, dtype=int)
    s = 1 if one_based else 0
    T = T - s
    if dirichlet_file is None:
        D = boundary_nodes_from_triangles(T)
    else:
        D = np.loadtxt(dirichlet_file, dtype=int) - s
    return Mesh(V, T, D)


def from_meshio(path_or_mesh) -> Mesh:
    """Any format supported by meshio (gmsh .msh, .vtk, .xdmf, ...). Uses the triangle cells."""
    import meshio
    m = meshio.read(path_or_mesh) if isinstance(path_or_mesh, (str, bytes)) or hasattr(path_or_mesh, "__fspath__") else path_or_mesh
    T = m.get_cells_type("triangle")
    return Mesh(m.points[:, :2], T, boundary_nodes_from_triangles(T))
