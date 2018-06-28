r"""
Utilities (:mod:`meshless.utils`)
=================================
"""
import numpy as np

def area_of_polygon(x, y):
    """Area of an arbitrary 2D polygon given its vertices
    """
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0


def unitvec(vector):
    """Return the unit vector
    """
    return vector / np.linalg.norm(vector)


def getMid(elem):
    """Get mid xyz coordinates given a pyNastran Element
    """
    return elem.get_node_positions().mean(axis=0)


