import os
import inspect

import numpy as np
from scipy.sparse import csr_matrix
from composites.laminate import read_stack
from structsolve import solve, lb

from meshless.espim.read_mesh import read_mesh
from meshless.espim.plate2d_calc_kC import calc_kC
from meshless.espim.plate2d_calc_kG import calc_kG
from meshless.espim.plate2d_calc_kCs import calc_kCs

THISDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))

def test_calc_linear_buckling():
    E11 = 71.e9
    nu = 0.33
    plyt = 0.007
    lam = read_stack([0], plyt=plyt, laminaprop=(E11, E11, nu))
    ans = {'edge-based': 9.4115354, 'cell-based': 6.98852939,
            'cell-based-no-smoothing': 4.921956}
    for prop_from_nodes in [True, False]:
        for kCs_method in ['cell-based', 'cell-based-no-smoothing', 'edge-based']:
            mesh = read_mesh(os.path.join(THISDIR, 'nastran_plate_16_nodes.dat'))
            for tria in mesh.elements.values():
                tria.prop = lam
            for node in mesh.nodes.values():
                node.prop = lam
            kC = calc_kC(mesh, prop_from_nodes)
            kC = csr_matrix(kC)
            kCs = calc_kCs(mesh, prop_from_nodes, kCs_method, alpha=0.2)
            kCs = csr_matrix(kCs)
            kC += kCs

            # running static subcase first
            dof = 5
            n = kC.shape[0] // 5
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

            bc(kC)
            d = solve(kC, fext, silent=True)
            kG = calc_kG(d, mesh, prop_from_nodes)
            kG = csr_matrix(kG)
            bc(kG)

            eigvals, eigvecs = lb(kC, kG, silent=True)
            print('kCs_method, eigvals[0]', kCs_method, eigvals[0])

            assert np.isclose(eigvals[0], ans[kCs_method])

if __name__ == '__main__':
    test_calc_linear_buckling()

