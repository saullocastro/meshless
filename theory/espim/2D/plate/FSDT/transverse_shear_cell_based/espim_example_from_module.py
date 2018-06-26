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

E11 = 71.e9
nu = 0.33
plyt = 0.007
laminaprop = (E11, E11, nu)
lam = read_stack([0], plyt=plyt, laminaprop=laminaprop, calc_scf=True)
prop_from_nodes = False
k0s_method = 'cell-based'
mesh = read_mesh(os.path.join(THISDIR, 'nastran_test.dat'))

nodes = mesh.nodes.values()
for tria in mesh.elements.values():
    tria.prop = lam
for node in nodes:
    node.prop = lam
k0 = calc_k0(mesh, prop_from_nodes)
add_k0s(k0, mesh, prop_from_nodes, k0s_method)

dof = 5
N = k0.shape[0] // 5

# boundary conditions
def bc(K):
    ids = [n.nid for n in nodes if np.isclose(n.xyz[0], 0.)]
    for i in ids:
        for j in [0, 1, 2, 3]:
            K[mesh.nodes[i].index*dof+j, :] = 0
            K[:, mesh.nodes[i].index*dof+j] = 0

    ids = ([n.nid for n in nodes if np.isclose(n.xyz[1], 3.)] +
           [n.nid for n in nodes if np.isclose(n.xyz[1], 5.)] +
           [n.nid for n in nodes if np.isclose(n.xyz[0], 7.)]
           )
    for i in ids:
        for j in [2]:
            K[mesh.nodes[i].index*dof+j, :] = 0
            K[:, mesh.nodes[i].index*dof+j] = 0

bc(k0)
k0 = coo_matrix(k0)

# running static subcase first
fext = np.zeros(N*dof, dtype=np.float64)
Nxx = -100
b = 2
fu = Nxx * b
ids_corner = [n.nid for n in nodes if ((np.isclose(n.xyz[1], 3.)
    or np.isclose(n.xyz[1], 5.)) and np.isclose(n.xyz[0], 7.))]
assert len(ids_corner) == 2, 'ids_corner %d' % len(ids_corner)
ids_middle = [n.nid for n in nodes if (not np.isclose(n.xyz[1], 3)
    and not np.isclose(n.xyz[1], 5) and np.isclose(n.xyz[0], 7.))]
val = fu/(N**0.5-1)
for i in ids_corner:
    fext[mesh.nodes[i].index*dof + 0] = val/2
for i in ids_middle:
    fext[mesh.nodes[i].index*dof + 0] = val
print('fext.sum()', fext.sum())
d = solve(k0, fext, silent=True)

# running linear buckling case
vec = d[0::dof]
print('u.min()', vec.min())
kG = calc_kG(d, mesh, prop_from_nodes)
bc(kG)
kG = coo_matrix(kG)

eigvals, eigvecs = lb(k0, kG, silent=True)
print('eigvals[0]', eigvals[0])


xyz = np.array([n.xyz for n in nodes])
ind = np.lexsort((xyz[:, 1], xyz[:, 0]))
xyz = xyz[ind]
nodes = np.array(list(nodes))[ind]
pick = [n.index for n in nodes]
vec = eigvecs[2::dof, 0]
levels = np.linspace(vec.min(), vec.max(), 16)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.figure(dpi=100, figsize=(6, 3))
nx = int(N**0.5)
ny = int(N**0.5)
plt.axis('off')
plt.axes().set_aspect('equal')

plt.contourf(xyz[:, 0].reshape(nx, ny), xyz[:, 1].reshape(nx, ny), vec[pick].reshape(nx, ny),
        levels=levels, cmap=cm.gist_rainbow_r)
plt.colorbar()
plt.show()
