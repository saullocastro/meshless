import os
import inspect

from meshless.espim.read_mesh import read_mesh

THISDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))

def test_read_mesh():
    nodes, trias, edges = read_mesh(os.path.join(THISDIR, 'nastran_plate_16_nodes.dat'))
    assert len(nodes) == 16
    assert len(trias) == 18
    assert len(edges) == 33


