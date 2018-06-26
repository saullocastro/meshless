import os
import inspect

import numpy as np
from scipy.sparse import coo_matrix
from composites.laminate import read_stack
from structsolve import solve, lb

from meshless.espim.read_mesh import read_mesh
from meshless.espim.plate2d_calc_k0 import calc_k0
from meshless.espim.plate2d_calc_kG import calc_kG
from meshless.espim.plate2d_add_k0s import add_k0s

THISDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))

def test_calc_linear_buckling():
    E11 = 71.e9
    nu = 0.33
    plyt = 0.007
    lam = read_stack([0], plyt=plyt, laminaprop=(E11, E11, nu))
    ans = {'edge-based': 9.4115354, 'cell-based': 6.98852939,
            'cell-based-no-smoothing': 4.921956}
    for prop_from_nodes in [True, False]:
        for k0s_method in ['edge-based', 'cell-based', 'cell-based-no-smoothing']:
            mesh = read_mesh(os.path.join(THISDIR, 'nastran_plate_16_nodes.dat'))
            for tria in mesh.elements.values():
                tria.prop = lam
            for node in mesh.nodes.values():
                node.prop = lam
            k0 = calc_k0(mesh, prop_from_nodes)
            add_k0s(k0, mesh, prop_from_nodes, k0s_method)

            # running static subcase first
            dof = 5
            n = k0.shape[0] // 5
            fext = np.zeros(n*dof, dtype=np.float64)
            fext[mesh.nodes[4].index*dof + 0] = -500.
            fext[mesh.nodes[7].index*dof + 0] = -500.
            fext[mesh.nodes[5].index*dof + 0] = -1000.
            fext[mesh.nodes[6].index*dof + 0] = -1000.

            # boundary conditions
            def bc(K):
                for i in [1, 10, 11, 12]:
                    for j in [0, 1, 2]:
                        K[mesh.nodes[i].index*dof+j, :] = 0
                        K[:, mesh.nodes[i].index*dof+j] = 0

                for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                    for j in [1, 2]:
                        K[mesh.nodes[i].index*dof+j, :] = 0
                        K[:, mesh.nodes[i].index*dof+j] = 0

            bc(k0)
            k0 = coo_matrix(k0)
            d = solve(k0, fext, silent=True)
            kG = calc_kG(d, mesh, prop_from_nodes)
            bc(kG)
            kG = coo_matrix(kG)

            eigvals, eigvecs = lb(k0, kG, silent=True)
            print('k0s_method, eigvals[0]', k0s_method, eigvals[0])

            assert np.isclose(eigvals[0], ans[k0s_method])

if __name__ == '__main__':
    test_calc_linear_buckling()

