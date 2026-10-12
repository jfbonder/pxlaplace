"""pxlaplace: p(x)-Laplacian solver (decomposition-coordination + P1 FEM).

Python port of the MatLab code accompanying
A. Aragon, J. Fernandez Bonder, D. Rubio,
"Effective numerical computation of p(x)-Laplace equations in 2D".
"""
from .mesh import (Mesh, rectangle_mesh, polygon_mesh, sdf_mesh, disk_mesh,
                   from_txt, from_meshio, boundary_nodes_from_triangles)
from .fem import P1Space
from .solver import PxLaplaceSolver, solve, SolveResult
from .nonlinear import solve_scalar

__all__ = ["Mesh", "rectangle_mesh", "polygon_mesh", "sdf_mesh", "disk_mesh",
           "from_txt", "from_meshio", "boundary_nodes_from_triangles",
           "P1Space", "PxLaplaceSolver", "solve", "SolveResult", "solve_scalar"]
