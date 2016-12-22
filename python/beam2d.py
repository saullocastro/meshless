import numpy as np
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

        #TODO use list of adjacent particles
        #TODO compute vector of positions among adjacent particles
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p4 = None
        self.E = 71e9
        self.nu = 0.33
        self.length = 0.01
        self.h = 0.01
        self.b = 0.01
        self.x = 0
        self.y = 0
        self.pos_1 = np.array([0., 0.])
        self.pos = np.array([0., 0.])

        self.u_1 = np.array([0, 0.]) # 2-D u, v
        self.u = np.array([0, 0.]) # 2-D u, v

        self.ut_1 = np.array([0, 0.]) # 2-D uxx, vxx, rzxx
        self.ut = np.array([0, 0.]) # 2-D uxx, vxx, rzxx

        self.utt = np.array([0, 0.]) # 2-D uxx, vxx, rzxx

        self.f1 = np.array([0., 0.]) # 2-D
        self.f2= np.array([0., 0.]) # 2-D
        self.f3= np.array([0., 0.]) # 2-D
        self.f4= np.array([0., 0.]) # 2-D
        self.fext = np.array([0, 0]) # 2-D

        self.build()


    def build(self):
        self.A = self.h * self.b
        kuu = self.E*self.A
        kvv = self.E*self.A
        self.k = np.array([[kuu, kuv],
                           [kvu, kvv]])
        self.kinv = np.linalg.inv(self.k)
        # mass matrix
        self.m = (self.length * self.A * self.rho * np.array([[1, 0],
                                                              [0, 1]]))

# building up particle connectivity
lenx = 1000
leny = 10
numx = 100
numy = 5
ps = [P() for _ in range(numx * numy)]
psy0 = []

for i in range(numx):
    psy0.append(ps[i])
    for j in range(numy):
        #TODO update position x and y
        # recalculate force components among particles according to the
        # updated x and y position
        p = ps[j*numx + i]
        p.pos[0] = lenx*i/(numx-1)
        p.pos[1] = leny*j/(numy-1)
        p.pos_1[:] = p.pos[:]

for j in range(numy):
    for i in range(numx-1):
        ps[j*numx + i].p2 = ps[j*numx + i+1]
        ps[j*numx + i+1].p1 = ps[j*numx + i]
for i in range(numx):
    for j in range(numy-1):
        ps[i*numy + j].p4 = ps[i*numy + j+1]
        ps[i*numy + j+1].p3 = ps[i*numy + j]

# integration
dt = 0.001
n = 10000

ps[-1].fext = np.array([0, 1])

for step in range(n):
    for i in range(numx):
        for j in range(numy):
            p = ps[j*numx + j]
            if i in (0, 1):
                p.u *= 0 # clamping one side
                continue
            # forces
            p.f *= 0
            p.f += p.fext
            #TODO use list of particles
            #TODO finish this force and displacement calculation...
            if p.p1:
                p.f1 = p.k.dot(p.u_1 - p.p1.u_1)/2
                vec1 = p.pos - p1.pos
                d1 = (vec1**2).sum()**0.5
                k1 = p.E * p.A / d1
                p.u += p.f / k1
            if p.p2:
                p.f += p.k.dot(p.u_1 - p.p2.u_1)/2
            if p.p3:
                p.f += p.k.dot(p.u_1 - p.p3.u_1)/2
            if p.p4:
                p.f += p.k.dot(p.u_1 - p.p4.u_1)/2

            p.ut = (p.u - p.u_1)/dt
            p.utt = (p.ut - p.ut_1)/dt
            du = p.kinv.dot(p.f - p.m.dot(p.utt))
            p.u = p.u_1 + du

            p.u_1 = p.u.copy()
            p.ut_1 = p.ut.copy()
            p.utt_1 = p.utt.copy()

plt.plot([p.u[1] for p in ps])
plt.savefig(filename='tmp_beam2d.png')


    #self.u
    #k x + m*d2x/dt2 = f
    #f = k*x

#u = u + du/dt * deltat


