import os
import inspect

import numpy as np
from scipy.sparse import coo_matrix
from composites.laminate import read_stack
from structsolve import solve

from meshless.espim.read_mesh import read_mesh
from meshless.espim.plate2d_calc_k0 import calc_k0
from meshless.espim.plate2d_add_k0s import add_k0s

THISDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))

def test_calc_linear_static():
    mesh = read_mesh(os.path.join(THISDIR, 'nastran_plate_16_nodes.dat'))
    E11 = 71.e9
    nu = 0.33
    plyt = 0.007
    lam = read_stack([0], plyt=plyt, laminaprop=(E11, E11, nu))
    for tria in mesh.elements.values():
        tria.prop = lam
    for node in mesh.nodes.values():
        node.prop = lam
    for prop_from_nodes in [False, True]:
        for k0s_method in ['cell-based', 'cell-based-no-smoothing']: #, 'edge-based'
            k0 = calc_k0(mesh, prop_from_nodes)
            add_k0s(k0, mesh, prop_from_nodes, k0s_method)

            k0run = k0.copy()

            dof = 5
            n = k0.shape[0] // dof
            fext = np.zeros(n*dof, dtype=np.float64)
            fext[mesh.nodes[4].index*dof + 2] = 500.
            fext[mesh.nodes[7].index*dof + 2] = 500.
            fext[mesh.nodes[5].index*dof + 2] = 1000.
            fext[mesh.nodes[6].index*dof + 2] = 1000.
            i, j = np.indices(k0run.shape)

            # boundary conditions
            for nid in [1, 10, 11, 12]:
                for j in [0, 1, 2, 3]:
                    k0run[mesh.nodes[nid].index*dof+j, :] = 0
                    k0run[:, mesh.nodes[nid].index*dof+j] = 0

            k0run = coo_matrix(k0run)
            u = solve(k0run, fext, silent=True)
            ans = np.loadtxt(os.path.join(THISDIR, 'nastran_plate_16_nodes.result.txt'),
                    dtype=float)
            xyz = np.array([n.xyz for n in mesh.nodes.values()])
            ind = np.lexsort((xyz[:, 1], xyz[:, 0]))
            xyz = xyz[ind]
            nodes = np.array(list(mesh.nodes.values()))[ind]
            pick = [n.index for n in nodes]
            assert np.allclose(u[2::5][pick].reshape(4, 4).T, ans, rtol=0.05)


if __name__ == '__main__':
    test_calc_linear_static()

