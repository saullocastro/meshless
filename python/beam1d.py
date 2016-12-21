import numpy as np
import matplotlib.pyplot as plt


class P(object):
    def __init__(self):
        self.rho = 5.e3

        #TODO use list of adjacent particles
        #TODO compute vector of positions among adjacent particles
        self.p1 = None
        self.p2 = None
        #self.p3 = None
        #self.p4 = None
        self.E = 71e9
        self.nu = 0.33
        self.length = 0.01
        self.h = 0.01
        self.b = 0.01

        self.u_1 = np.array([0, 0, 0.]) # 2-D u, v, rz
        self.u = np.array([0, 0, 0.]) # 2-D u, v, rz

        self.ut_1 = np.array([0, 0, 0.]) # 2-D uxx, vxx, rzxx
        self.ut = np.array([0, 0, 0.]) # 2-D uxx, vxx, rzxx

        self.utt = np.array([0, 0, 0.]) # 2-D uxx, vxx, rzxx

        self.f = np.array([0, 0, 0.]) # 2-D
        self.fext = np.array([0, 0, 0.]) # 2-D

        #self.f = np.array([0, 0, 0, 0, 0, 0]) # 3-D


        self.build()


    def build(self):
        self.A = self.h * self.b
        self.I = self.h * self.b**3 / 12
        ku = self.E*self.A / self.length
        if self.p1 and self.p2:
            kv = kb*self.length
        if (self.p1 and not self.p2) or (not self.p1 and self.p2):
            kv = kb*self.length / 2
        else:
            #TODO this is the case of a loose particle
            ku = 1e-6
            kv = 1e-6
        kb = self.E*self.I
        self.k = np.array([[ku, 0, 0],
                           [0, kv, 0],
                           [0, 0, kb]])
        self.kinv = np.linalg.inv(self.k)

        # mass matrix
        self.m = (self.length * self.A * self.rho *
                np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, self.b**2/3]]))


ps = [P() for _ in range(100)]
for i, p in enumerate(ps[1:]):
    p.p1 = ps[i]
for i, p in enumerate(ps[:-1]):
    p.p2 = ps[i+1]

# integration
dt = 0.001
n = 10000

ps[49].fext = np.array([0, 1, 0])

for i in range(n):
    for ip, p in enumerate(ps):
        if ip in (0, 1):
            p.u *= 0 # clamping one side
            continue
        if ip in (98, 99):
            p.u *= 0 # clamping one side
            continue
        # forces
        p.f *= 0
        p.f += p.fext
        #TODO use list of particles
        if p.p1:
            p.f += p.k.dot(p.u_1 - p.p1.u_1)/2
        if p.p2:
            p.f += p.k.dot(p.p2.u_1 - p.u_1)/2

        p.ut = (p.u - p.u_1)/dt
        p.utt = (p.ut - p.ut_1)/dt
        du = p.kinv.dot(p.f - p.m.dot(p.utt))
        p.u = p.u_1 + du

        p.u_1 = p.u.copy()
        p.ut_1 = p.ut.copy()
        p.utt_1 = p.utt.copy()

plt.plot([p.u[1] for p in ps])
plt.savefig(filename='tmp_beam1d.png')


    #self.u
    #k x + m*d2x/dt2 = f
    #f = k*x

#u = u + du/dt * deltat


