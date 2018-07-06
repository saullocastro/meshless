from meshless.espim.plate2d_calc_kCs_cell_based import calc_kCs as calc_kCs_cell
from meshless.espim.plate2d_calc_kCs_cell_based_no_smoothing import calc_kCs as calc_kCs_cell_no_smoothing
from meshless.espim.plate2d_calc_kCs_edge_based import calc_kCs as calc_kCs_edge

def calc_kCs(mesh, prop_from_node, method='cell-based', alpha=0.08):
    """Add the transverse shear stiffness to an existing consitutive stiffness
    matrix

    The transverse shear stiffness is computed using the Discrete Shear Gap
    method, with a correction that uses parameter `alpha`

    Parameters
    ----------
    k0 : (N, N) array-like
        Existing stiffness matrix. This object is modified in-place

    mesh : :class:`pyNastran.bdf.BDF` object
        The object must have the proper edge references as those returned by
        :func:`.read_mesh` or :func:`.read_delaunay`

    prop_from_node : bool
        If the constitutive properties are assigned per node. Otherwise they
        are considered assigned per element

    method : str, optional
        The smoothing method for the transverse shear

    alpha : float
        Positive constant used in the correction applied to the transverse
        shear stiffness

    """
    #alpha between 0. and 0.6, according to studies of Lyly et al.
    if method == 'cell-based':
        return calc_kCs_cell(mesh, prop_from_node, alpha=alpha)
    elif method == 'cell-based-no-smoothing':
        return calc_kCs_cell_no_smoothing(mesh, prop_from_node, alpha=alpha)
    elif method == 'edge-based':
        return calc_kCs_edge(mesh, prop_from_node, alpha=alpha)
    else:
        raise ValueError('Invalid method')

