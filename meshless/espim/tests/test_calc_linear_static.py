import os
import inspect

import numpy as np
from scipy.sparse import coo_matrix

from meshless.composite.laminate import read_stack
from meshless.sparse import solve
from meshless.espim.read_mesh import read_mesh
from meshless.espim.plate2d_calc_k0 import calc_k0
from meshless.espim.plate2d_add_k0s import add_k0s

THISDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))

def test_calc_linear_static():
    nodes, trias, edges = read_mesh(os.path.join(THISDIR, 'nastran_plate_16_nodes.dat'))
    E11 = 71.e9
    nu = 0.33
    plyt = 0.007
    lam = read_stack([0], plyt=plyt, laminaprop=(E11, E11, nu))
    for tria in trias:
        tria.prop = lam
    for node in nodes:
        node.prop = lam
    for prop_from_nodes in [False, True]:
        for k0s_method in ['cell-based', 'cell-based-no-smoothing']: #, 'edge-based'
            k0 = calc_k0(nodes, trias, edges, prop_from_nodes)
            add_k0s(k0, edges, trias, prop_from_nodes, k0s_method)

            k0run = k0.copy()

            dof = 5
            n = k0.shape[0] // 5
            fext = np.zeros(n*dof, dtype=np.float64)
            fext[nodes[4].index*dof + 2] = 500.
            fext[nodes[5].index*dof + 2] = 1000.
            fext[nodes[6].index*dof + 2] = 1000.
            fext[nodes[7].index*dof + 2] = 500.
            i, j = np.indices(k0run.shape)

            # boundary conditions
            for i in [0, 9, 10, 11]:
                for j in [0, 1, 2, 3]:
                    k0run[nodes[i].index*dof+j, :] = 0
                    k0run[:, nodes[i].index*dof+j] = 0

            k0run = coo_matrix(k0run)
            u = solve(k0run, fext, silent=True)
            ans = np.loadtxt(os.path.join(THISDIR, 'nastran_plate_16_nodes.result.txt'),
                    dtype=float)
            assert np.allclose(u[2::5].reshape(4, 4).T, ans, rtol=0.02)


if __name__ == '__main__':
    test_calc_linear_static()

