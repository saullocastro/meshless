from __future__ import division

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class P(object):
    r"""
          y, v

        8 4 6
         \|/
      1-- o -- 2   x, u
         /|\
        5 3 7

    """
    def __init__(self, size):
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
        self.size = size
        self.h = 0.001
        self.x = 0
        self.y = 0
        self.pos_1 = np.array([0., 0.])
        self.pos = np.array([0., 0.])

        self.ut = np.array([0, 0.]) # 2-D uxx, vxx, rzxx

        self.utt = np.array([0, 0.]) # 2-D uxx, vxx, rzxx

        self.f = np.array([0., 0.]) # 2-D
        self.fext = np.array([0., 0.]) # 2-D

        self.build()


    def build(self):
        self.A = self.h * self.size
        self.m = self.size * self.A * self.rho

    def __str__(self):
        return 'P%02d%02d' % (self.i, self.j)
    def __repr__(self):
        return self.__str__()

# building particles
lenx = 1
leny = 0.5
size = 0.1
numx = int(lenx/size)
numy = int(leny/size)
ps = [P(size) for _ in range(numx * numy)]

# positions
psdict = {}
for i in range(numx):
    psdict[i] = {}
    for j in range(numy):
        p = ps[j*numx + i]
        psdict[i][j] = p
        p.x = lenx*i/(numx-1)
        p.y = leny*j/(numy-1)
        p.pos[0] = p.x
        p.pos[1] = p.y
        p.pos_1[:] = p.pos
        p.i = i
        p.j = j

# connectivity
for i in range(numx):
    for j in range(numy):
        p = psdict[i][j]
        if i != 0:
            p.padjs.append(psdict[i-1][j])
            if j > 0:
                p.padjs.append(psdict[i-1][j-1])
            if j < numy-1:
                p.padjs.append(psdict[i-1][j+1])
        if i != numx-1:
            p.padjs.append(psdict[i+1][j])
            if j > 0:
                p.padjs.append(psdict[i+1][j-1])
            if j < numy-1:
                p.padjs.append(psdict[i+1][j+1])
        if j != 0:
            p.padjs.append(psdict[i][j-1])
        if j != numy-1:
            p.padjs.append(psdict[i][j+1])

for p in ps:
    for padj in p.padjs:
        dist = ((p.pos - padj.pos)**2).sum()**0.5
        p.dadjs.append(dist)

fig = plt.figure(dpi=500, figsize=(10, 10))
fig.clear()
ax = plt.gca()
#ax.set_xlim(-2*size, lenx+2*size)
#ax.set_ylim(-2*size, leny+2*size)
ax.plot([p.pos[0] for p in ps], [p.pos[1] for p in ps], 'ko')
ax.set_aspect('equal')
fig.savefig(filename='tmp_beam2d_points.png', bbox_inches='tight')

# integration
dt = 0.0001
n = 200

psdict[numx-1][numy-1].fext += [0, 100]

for step in range(n):
    for p in ps:
        # forces
        p.f *= 0
        p.f += p.fext
        dist_init = p.dadjs
        for i, padj in enumerate(p.padjs):
            #TODO not considering area reduction deformedwith dist variation
            k = p.E * p.A / dist_init[i]
            pos_diff = p.pos_1 - padj.pos_1
            if pos_diff.sum() == 0:
                continue
            dx = pos_diff[0]
            dy = pos_diff[1]
            d = (dx**2 + dy**2)**0.5
            fres = (d - dist_init[i]) * k
            fx = -fres * dx / d
            fy = -fres * dy / d
            p.f += [fx, fy]

        p.utt = 1./p.m*p.f
        p.ut = p.ut + p.utt*dt
        p.pos = p.pos_1 + p.ut*dt

    for p in ps:
        if np.any(np.isnan(p.pos)):
            print(p.pos)
            raise RuntimeError()
        p.pos_1[:] = p.pos

        # clamping one side
        if p.i in [0]:
            p.pos[:] = [p.x, p.y]
            p.pos_1[:] = [p.x, p.y]
            p.ut *= 0
            p.utt *= 0

xplot = [p.pos[0] for p in ps]
yplot = [p.pos[1] for p in ps]
ax.plot(xplot, yplot, 'r^', mfc='None')
fig.savefig(filename='tmp_beam2d_deformed.png', bbox_inches='tight')

if True:
    fig.clear()
    ax = fig.gca()
    ax.set_aspect('equal')
    xplot = np.array([p.x for p in ps]).reshape(numy, numx)
    yplot = np.array([p.y for p in ps]).reshape(numy, numx)
    fxplot = np.array([p.f[1] for p in ps]).reshape(numy, numx)
    levels = np.linspace(fxplot.min(), fxplot.max(), 400)
    ax.contourf(xplot, yplot, fxplot, levels=levels)
    fig.savefig(filename='tmp_beam2d_fx.png', bbox_inches='tight')

