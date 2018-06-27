import numpy as np

from meshless.interpolate import (interpolate_lagrange_2D,
        interpolate_lagrange_3D, interpolate_lagrange_ND)

def test_interpolate_lagrange_2D():
    a = 1
    xs = np.linspace(0, a, 12)
    x_ref = np.array([0, a/3, 2*a/3, a])
    theta_ref = np.array([0, 45, 45, 0])
    thetas = interpolate_lagrange_2D(xs, xs*0, x_ref, x_ref*0, theta_ref)
    ans = [0., 16.73553719, 30.12396694, 40.16528926, 46.85950413,
           50.20661157, 50.20661157, 46.85950413, 40.16528926, 30.12396694,
           16.73553719, 0.]
    assert np.allclose(ans, thetas)

def test_interpolate_lagrange_3D_ND():
    a = 1
    b = 2
    c = 3
    xs = np.linspace(0, a, 2.)
    ys = np.linspace(0, b, 2.)
    zs = np.linspace(0, c, 2.)
    x = np.array(np.meshgrid(xs, ys, zs)).swapaxes(0, -1).reshape(-1, 3)
    x_ref = np.array([[0, 0, 0],
                      [a, 0, 0],
                      [0, b, 0],
                      [a, b, 0],
                      [0, 0, c],
                      [a, 0, c],
                      [0, b, c],
                      [a, b, c]])
    theta_ref = np.array([0, 1., 0., 1, 0, 1, 0, 1])
    thetas = interpolate_lagrange_ND(x, x_ref, theta_ref)
    assert np.allclose(thetas, [0., 1., 0., 1., 0., 1., 0., 1.])
    thetas = interpolate_lagrange_3D(x[:, 0], x[:, 1], x[:, 2], x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], theta_ref)
    assert np.allclose(thetas, [0., 1., 0., 1., 0., 1., 0., 1.])

