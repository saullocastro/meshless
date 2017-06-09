import numpy as np
from scipy.spatial import Delaunay

from meshless.espim.read_mesh import read_delaunay


def test_read_delaunay():
    xs = np.linspace(0, 0.5, 3)
    ys = np.linspace(0, 0.2, 3)
    points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
    tri = Delaunay(points)
    mesh = read_delaunay(points, tri)

    assert len(mesh.nodes) == 9
    assert len(mesh.elements) == 8
    assert len(mesh.edges) == 16


if __name__ == '__main__':
    test_read_delaunay()
