from __future__ import division

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class P(object):
    r"""Single Particle
    """
    def __init__(self, size):
        self.rho = 5.e3
        self.size = size
        self.h = 0.001
        self.pos = np.array([0., 0.])
        self.ut = np.array([0., 0.]) # 2-D uxx, vxx, rzxx
        self.utt = np.array([0., 0.]) # 2-D uxx, vxx, rzxx
        self.fext = np.array([0., 0.]) # 2-D
        self.A = self.h * self.size
        self.m = self.size * self.A * self.rho

fig = plt.figure(dpi=500, figsize=(10, 10))
fig.clear()
ax = plt.gca()
#ax.set_aspect('equal')

# integration
dt = 10.
n = 400000

p = P(0.01)
# verifying Earth's scape velocity, try to slightly increate the 11200 value
# below
p.ut[:] = [10, 11200]
xs = []
ys = []

re = 6.3710088e6 # Earth's mean radius
g0 = 9.80665 # Standard gravitation acceleration

for step in range(n):
    p.fext[1] = -(g0*(re/(re + p.pos[1]))**2) * p.m
    p.utt = 1/p.m*p.fext
    p.ut = p.ut + p.utt*dt
    p.pos = p.pos + p.ut*dt #+ p.utt*dt**2
    if np.any(np.isnan(p.pos)):
        print(p.pos)
        raise
    xs.append(p.pos[0])
    ys.append(p.pos[1])
    if p.pos[1] < 0:
        break

ax.plot(xs, ys, '-o')
fig.savefig(filename='tmp_single_trajectory.png', bbox_inches='tight')
