import numpy as np

def interpolate_lagrange_ND(x, x_nodes, field_nodes):
    """Interpolate values using ND Lagrangian polynomials

    The order of the polynomial interpolation in each direction depends on the
    number of field nodes passed to this function

    Parameters
    ----------
    x : (N, D) array_like
        Coordinates used to interpolate
    x_nodes : (K, D) array_like
        Nodal coordinates
    field_nodes : (K) array_like
        Field to be interpolated at corresponding nodes

    Returns
    -------
    field : (N,) array_like
        Interpolated field

    """
    x = np.asarray(x)
    x_nodes = np.asarray(x_nodes)
    x = np.atleast_2d(x).reshape(x.shape[0], -1)
    x_nodes = np.atleast_2d(x_nodes).reshape(x_nodes.shape[0], -1)
    #TODO remove loops one day
    field = 0
    for m, (xs, theta) in enumerate(zip(x_nodes, field_nodes)):
        tmp = 1
        for dim, xi in enumerate(xs):
            for k, xk in enumerate(x_nodes[:, dim]):
                if k != m and (xi - xk) != 0:
                    tmp *= (x[:, dim]- xk)/(xi - xk)
        field += tmp * theta
    return field


def interpolate_lagrange_2D(x, y, x_nodes, y_nodes, field_nodes):
    """Interpolate values using 2D Lagrangian polynomials

    The order of the polynomial interpolation in each direction depends on the
    number of field nodes passed to this function

    Parameters
    ----------
    x, y : (N,) array_like
        Coordinates used to interpolate
    x_nodes, y_nodes, field_nodes : array_like
        Nodal coordinates and field to be interpolated at corresponding nodes

    Returns
    -------
    field : (N,) array_like
        Interpolated field

    """
    #TODO try to remove loops one day
    field = 0
    for m, (xm, ym, theta) in enumerate(zip(x_nodes, y_nodes, field_nodes)):
        tmp = 1
        for k, xk in enumerate(x_nodes):
            if k != m and (xm - xk) != 0:
               tmp *= (x - xk)/(xm - xk)
        for p, yp in enumerate(y_nodes):
            if p != m and (ym - yp) != 0:
               tmp *= (y - yp)/(ym - yp)
        field += tmp * theta
    return field


def interpolate_lagrange_3D(x, y, z, x_nodes, y_nodes, z_nodes, field_nodes):
    """Interpolate values using 3D Lagrangian polynomials

    The order of the polynomial interpolation in each direction depends on the
    number of field nodes passed to this function

    Parameters
    ----------
    x, y, z : (N,) array_like
        Coordinates used to interpolate
    x_nodes, y_nodes, z_nodes, field_nodes : array_like
        Nodal coordinates and field to be interpolated at corresponding nodes

    Returns
    -------
    field : (N,) array_like
        Interpolated field

    """
    #TODO try to remove loops one day
    field = 0
    for m, (xm, ym, zm, theta) in enumerate(zip(x_nodes, y_nodes, z_nodes, field_nodes)):
        tmp = 1
        for k, xk in enumerate(x_nodes):
            if k != m and (xm - xk) != 0:
               tmp *= (x - xk)/(xm - xk)
        for p, yp in enumerate(y_nodes):
            if p != m and (ym - yp) != 0:
               tmp *= (y - yp)/(ym - yp)
        for p, zp in enumerate(z_nodes):
            if p != m and (zm - zp) != 0:
               tmp *= (z - zp)/(zm - zp)
        field += tmp * theta
    return field
