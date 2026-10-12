import numpy as np, pytest
from pxlaplace import (rectangle_mesh, polygon_mesh, disk_mesh, P1Space,
                       PxLaplaceSolver, solve_scalar)
from pxlaplace.nonlinear import update_nu


def test_stiffness_and_load():
    m = rectangle_mesh(8, 8)
    fem = P1Space(m)
    assert np.isclose(fem.area.sum(), 1.0)
    assert np.allclose(fem.K @ np.ones(m.n_nodes), 0)                 # constants in kernel
    x, y = m.vertices.T
    assert np.isclose((fem.K @ x) @ x, 1.0)                           # int |grad x|^2 = 1
    assert np.isclose(fem.load_vector(1.0).sum(), 1.0)
    assert np.isclose(fem.load_vector(lambda x, y: x * y, degree=2).sum(), 0.25)


@pytest.mark.parametrize("p", [1.05, 1.5, 2.0, 4.0, 50.0])
def test_scalar_equation(p):
    r = np.logspace(-12, 6, 50)
    t = solve_scalar(np.full_like(r, p), r)
    assert np.allclose(t ** (p - 1) + t, r, rtol=1e-10, atol=0)
    nx, ny = update_nu(np.full(3, p), np.array([1., 0, 0]), np.array([0., 2, 0]))
    assert np.allclose(np.hypot(nx, ny) ** (p - 1) + np.hypot(nx, ny), [1, 2, 0]) and nx[2] == 0


def test_p2_is_poisson():
    # p = 2: -Laplace u = f, compare with exact u = sin(pi x) sin(pi y)
    m = rectangle_mesh(40)
    ex = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    res = PxLaplaceSolver(m, 2.0, lambda x, y: 2 * np.pi**2 * ex(x, y), 0.0).solve(tol=1e-10)
    assert res.converged and np.abs(res.u - ex(*m.vertices.T)).max() < 2e-3


def test_example3_variable_p():
    pf = lambda x, y: 1 + 1 / ((x + y) / 2 + 2)
    ex = lambda x, y: np.sqrt(2) * np.e**2 * (np.exp(0.5 * (x + y)) - 1)
    m = rectangle_mesh(40, 40, -1, 1, -1, 1)
    res = PxLaplaceSolver(m, pf, 0.0, ex).solve(tol=1e-7)
    assert res.converged and np.abs(res.u - ex(*m.vertices.T)).max() < 2e-4


def test_disk_torsion_general_mesh():
    p = 4.0; c = (p - 1) / p * 2 ** (-1 / (p - 1))
    ex = lambda x, y: c * (1 - (x**2 + y**2) ** (p / (2 * p - 2)))
    for m in (disk_mesh(0.06),):
        res = PxLaplaceSolver(m, p, 1.0, 0.0).solve(tol=1e-8, stop="p_error")
        assert res.converged and np.abs(res.u - ex(*m.vertices.T)).max() < 2e-3


def test_polygon_with_hole_maximum_principle():
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    m = polygon_mesh(np.c_[2 * np.cos(th), 2 * np.sin(th)], holes=[np.c_[.5 * np.cos(th), .5 * np.sin(th)]], max_area=0.01)
    g = lambda x, y: np.where(x**2 + y**2 < 1, 1.0, 0.0)
    res = PxLaplaceSolver(m, 3.0, 0.0, g).solve(tol=1e-8)
    assert res.converged and res.u.min() > -1e-8 and res.u.max() < 1 + 1e-8


def test_random_init_same_limit():
    m = rectangle_mesh(20)
    a = PxLaplaceSolver(m, 3.0, 1.0, 0.0).solve(tol=1e-9)
    b = PxLaplaceSolver(m, 3.0, 1.0, 0.0).solve(tol=1e-9, init="random")
    assert np.abs(a.u - b.u).max() < 1e-6


def test_true_errors_quadrature():
    m = rectangle_mesh(10)
    fem = P1Space(m)
    # exact solution representable in P1 -> zero error
    lin = lambda x, y: 1 + 2 * x - 3 * y
    assert fem.l2_error(lin(*m.vertices.T), lin, relative=False) < 1e-14
    assert fem.grad_error_Lp(lin(*m.vertices.T), lambda x, y: (2 + 0 * x, -3 + 0 * x), 3.0, relative=False) < 1e-14
    # ||x^2||_{L2(0,1)^2} = 1/sqrt(5) ; uh = 0
    sq = lambda x, y: x**2 + 0 * y
    assert np.isclose(fem.l2_error(np.zeros(m.n_nodes), sq, relative=False), 1 / np.sqrt(5))
    # true error of the P1 interpolant of x^2 is O(h^2) and larger than the nodal error (which is 0)
    e = fem.l2_error(sq(*m.vertices.T), sq, relative=False)
    assert 0 < e < 0.02 and fem.norm_L2(sq(*m.vertices.T) - sq(*m.vertices.T)) == 0
