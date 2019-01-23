import numpy as np


def interpolate_lagrange_2D(x, y, x_nodes, y_nodes, field_nodes):
    """Interpolate values using 2D Lagrangian polynomials

    The order of the polynomial interpolation in each direction depends on the
    number control nodes passed for dimensions x and y.

    Parameters
    ----------
    x, y : (K,) array_like
        Coordinates used to interpolate
    x_nodes : (M,) array_like
        Coordinates of dimension 1
    y_nodes : (N,) array_like
        Coordinates of dimension 2
    field_nodes (M, N): array_like
        Nodal coordinates and field to be interpolated at corresponding control
        nodes

    Returns
    -------
    field : (K,) array_like
        Interpolated field

    """
    x_nodes = np.asarray(x_nodes)
    y_nodes = np.asarray(y_nodes)
    field_nodes = np.asarray(field_nodes)
    assert field_nodes.ndim == 2
    assert field_nodes.shape[0] == x_nodes.shape[0]
    assert field_nodes.shape[1] == y_nodes.shape[0]
    #TODO try to remove loops one day
    field = 0
    for m in range(len(x_nodes)):
        xm = x_nodes[m]
        fx = 1
        for i, xi in enumerate(x_nodes):
            if i != m and (xm - xi) != 0:
               fx *= (x - xi)/(xm - xi)
        for n in range(len(y_nodes)):
            yn = y_nodes[n]
            fy = 1
            for j, yj in enumerate(y_nodes):
                if j != n and (yn - yj) != 0:
                   fy *= (y - yj)/(yn - yj)
            field += fx * fy * field_nodes[m, n]
    return field


def interpolate_lagrange_3D(x, y, z, x_nodes, y_nodes, z_nodes, field_nodes):
    """Interpolate values using 3D Lagrangian polynomials

    The order of the polynomial interpolation in each direction depends on the
    number of field nodes passed to this function

    Parameters
    ----------
    x, y : (K,) array_like
        Coordinates used to interpolate
    x_nodes : (M,) array_like
        Coordinates of dimension 1
    y_nodes : (N,) array_like
        Coordinates of dimension 2
    z_nodes : (P,) array_like
        Coordinates of dimension 3
    field_nodes (M, N, P): array_like
        Nodal coordinates and field to be interpolated at corresponding control
        nodes

    Returns
    -------
    field : (N,) array_like
        Interpolated field

    """
    x_nodes = np.asarray(x_nodes)
    y_nodes = np.asarray(y_nodes)
    z_nodes = np.asarray(z_nodes)
    field_nodes = np.asarray(field_nodes)
    assert field_nodes.ndim == 3
    assert field_nodes.shape[0] == x_nodes.shape[0]
    assert field_nodes.shape[1] == y_nodes.shape[0]
    assert field_nodes.shape[2] == z_nodes.shape[0]
    #TODO try to remove loops one day
    field = 0
    for m in range(len(x_nodes)):
        xm = x_nodes[m]
        fx = 1
        for i, xi in enumerate(x_nodes):
            if i != m and (xm - xi) != 0:
               fx *= (x - xi)/(xm - xi)
        for n in range(len(y_nodes)):
            yn = y_nodes[n]
            fy = 1
            for j, yj in enumerate(y_nodes):
                if j != n and (yn - yj) != 0:
                   fy *= (y - yj)/(yn - yj)
            for p in range(len(z_nodes)):
                zp = z_nodes[p]
                fz = 1
                for k, zk in enumerate(y_nodes):
                    if k != p and (zp - zk) != 0:
                       fz *= (z - zk)/(zp - zk)
                field += fx * fy * fz * field_nodes[m, n, p]
    return field
