import numpy as np

def area_of_polygon(x, y):
    """Area of an arbitrary 2D polygon given its verticies
    """
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0


def unitvec(vector):
    """Return the unit vector
    """
    return vector / np.linalg.norm(vector)


class Property(object):
    def __init__(self, A, B, D, E):
        self.A = A
        self.B = B
        self.D = D
        self.E = E


class IntegrationPoint(object):
    def __init__(self, tria, n1, n2, n3, f1, f2, f3, nx, ny, nz, le):
        self.xyz = f1*n1.xyz + f2*n2.xyz + f3*n3.xyz
        self.tria = tria
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.nx = nx
        self.ny = ny
        self.nz = nz # 3D case
        self.le = le # length of the line where the integration point lies on
