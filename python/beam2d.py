from __future__ import division

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class P(object):
    """
          y, v

          4
          |
      1-- o -- 2  x, u
          |
          3

    """
    def __init__(self):
        self.rho = 5.e3
        self.i = 0
        self.j = 0
        #TODO use list of adjacent particles
        #TODO compute vector of positions among adjacent particles
        # adjacent particles
        self.padjs = []
        # initial distances
        self.dadjs = []

        self.E = 71e9
        self.nu = 0.33
        self.length = 0.01
        self.h = 0.01
        self.b = 0.01
        self.x = 0
        self.y = 0
        self.pos = np.array([0., 0.])

        self.u_1 = np.array([0, 0.]) # 2-D u, v
        self.u = np.array([0, 0.]) # 2-D u, v

        self.ut_1 = np.array([0, 0.]) # 2-D uxx, vxx, rzxx
        self.ut = np.array([0, 0.]) # 2-D uxx, vxx, rzxx

        self.utt_1 = np.array([0, 0.]) # 2-D uxx, vxx, rzxx
        self.utt = np.array([0, 0.]) # 2-D uxx, vxx, rzxx

        self.f = np.array([0., 0.]) # 2-D
        self.fext = np.array([0, 0]) # 2-D

        self.build()


    def build(self):
        self.A = self.h * self.b
        #kuu = self.E*self.A
        #kvv = self.E*self.A
        #self.k = np.array([[kuu, kuv],
                           #[kvu, kvv]])
        #self.kinv = np.linalg.inv(self.k)
        # mass matrix
        self.m = self.length * self.A * self.rho

    def __str__(self):
        return 'P%d%d' % (self.i, self.j)
    def __repr__(self):
        return self.__str__()

# building up particle connectivity
near = 4
lenx = 200
leny = 100
size = 10
numx = lenx//size
numy = leny//size
ps = [P() for _ in range(numx * numy)]

for i in range(numx):
    for j in range(numy):
        #TODO update position x and y
        # recalculate force components among particles according to the
        # updated x and y position
        p = ps[j*numx + i]
        p.pos[0] = lenx*i/(numx-1)
        p.pos[1] = leny*j/(numy-1)
        p.i = i
        p.j = j

posall = [p.pos for p in ps]
tree = cKDTree(posall)

for p in ps:
    distvec, posvec = tree.query(p.pos, k=(near+1))
    distvec = distvec[1:]
    posvec = posvec[1:]
    p.dadjs = distvec
    p.padjs = [ps[i] for i in posvec]

fig = plt.figure(dpi=500, figsize=(10, 10))
fig.clear()
ax = plt.gca()
#ax.set_xlim(-size, lenx+size)
#ax.set_ylim(-size, leny+size)
ax.plot([p.pos[0] for p in ps], [p.pos[1] for p in ps], 'ko')
ax.set_aspect('equal')
fig.savefig(filename='tmp_beam2d_points.png', bbox_inches='tight')

# integration
dt = 0.000001
n = 500

ps[-1].fext += [0, -1]

for step in range(n):
    for p in ps:
        # forces
        p.f *= 0
        p.f += p.fext
        dist_init = p.dadjs
        for i, padj in enumerate(p.padjs):
            pos_diff = p.pos - padj.pos
            if pos_diff.sum() == 0:
                continue
            dx = pos_diff[0]
            dy = pos_diff[1]
            d = (dx**2 + dy**2)**0.5
            #TODO not considering area reduction with dist variation
            k = p.E * p.A / dist_init[i]
            #FIXME treat component-wise
            fres = (d - dist_init[i]) * k
            fx = fres * dy / d
            fy = fres * dx / d
            p.f += [fx, fy]

        p.u = p.u_1 + p.ut_1*dt + p.utt_1*dt**2
        p.ut = p.ut_1 + p.utt_1*dt
        p.utt = 1./p.m*p.f
        if np.any(np.isnan(p.u)):
            print p.u
            raise

        p.u_1 = p.u.copy()
        p.ut_1 = p.ut.copy()
        p.utt_1 = p.utt.copy()

        # clamping one side
        if p.i in [0]:
            p.u_1 *= 0
            p.ut_1 *= 0
            p.utt_1 *= 0
            p.u *= 0
            p.ut *= 0
            p.utt *= 0

        p.pos += p.u

ax.plot([p.pos[0] for p in ps], [p.pos[1] for p in ps], 'r^', mfc='None')
fig.savefig(filename='tmp_beam2d_deformed.png', bbox_inches='tight')
