"""ES-PIM

- prevents non-zero energy spurious modes for vibration / buckling analysis as
  NS-PIM
- compared to SFEM using TRIA3, this approach is very similar, but much easily
  extented to n-sided elements, since the integration is performed edge-wise
- results more precise than FEM using QUAD4 elements

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

from pim.composite.laminate import read_stack
from pim.sparse import solve, is_symmetric


XGLOBAL = np.array([1., 0, 0])
YGLOBAL = np.array([0, 1., 0])
ZGLOBAL = np.array([0, 0, 1.])


def area_of_polygon(x, y):
    """Area of an arbitrary 2D polygon given its verticies
    """
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0


def unitvec(vector):
    """Return the unit vector
    """
    return vector / np.linalg.norm(vector)


class Property(object):
    def __init__(self, A, B, D, E):
        self.A = A
        self.B = B
        self.D = D
        self.E = E


class IntegrationPoint(object):
    def __init__(self, tria, n1, n2, n3, f1, f2, f3, nx, ny, nz, le):
        self.pos = (f1*n1 + f2*n2 + f3*n3).pos
        self.tria = tria
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.nx = nx
        self.ny = ny
        self.nz = nz # 3D case
        self.le = le # length of the line where the integration point lies on


class Tria(object):
    def __init__(self, n1, n2, n3):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.edges = []
        self.nodes = [n1, n2, n3]
        self.Ae = area_of_polygon([n1.pos[0], n2.pos[0], n3.pos[0]],
                                  [n1.pos[1], n2.pos[1], n3.pos[1]])
        self.prop = None # either define here or in Node


    def getMid(self):
        return 1/3*(self.n1 + self.n2 + self.n3)


class Edge(object):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.nodes = [n1, n2]
        self.trias = []
        self.sdomain = []
        self.ipts = []
        self.Ac = None

    def getMid(self):
        return 0.5*(self.n1 + self.n2)


class Node(object):
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.edges = set()
        self.trias = set()
        self.sdomain = None
        self.index = None
        self.prop = None # either define here or in Tria

    def __add__(self, val):
        if isinstance(val, Node):
            return Node(*(self.pos + val.pos))
        else:
            return Node(*(self.pos + val))

    def __sub__(self, val):
        if isinstance(val, Node):
            return Node(*(self.pos - val.pos))
        else:
            return Node(*(self.pos - val))

    def __rmul__(self, val):
        if isinstance(val, Node):
            return Node(*(self.pos * val.pos))
        else:
            return Node(*(self.pos * val))

    def __lmul__(self, val):
        return self.__rmul__(val)

a = 10.
b = 5.


plt.figure(dpi=300)

nodes = np.array([
        Node(0, 0, 0),
        Node(a/3, 0, 0),
        Node(2*a/3, 0, 0),
        Node(a, 0, 0),

        Node(0, b/3, 0),
        Node(a/3, b/3, 0),
        Node(2*a/3, b/3, 0),
        Node(a, b/3, 0),

        Node(0, 2*b/3, 0),
        Node(a/3, 2*b/3, 0),
        Node(2*a/3, 2*b/3, 0),
        Node(a, 2*b/3, 0),

        Node(0, b, 0),
        Node(a/3, b, 0),
        Node(2*a/3, b, 0),
        Node(a, b, 0),
        ])

trias = [
        Tria(nodes[0], nodes[1], nodes[5]),
        Tria(nodes[1], nodes[2], nodes[6]),
        Tria(nodes[2], nodes[3], nodes[7]),
        Tria(nodes[0], nodes[5], nodes[4]),
        Tria(nodes[1], nodes[6], nodes[5]),
        Tria(nodes[2], nodes[7], nodes[6]),

        Tria(nodes[4], nodes[5], nodes[9]),
        Tria(nodes[5], nodes[6], nodes[10]),
        Tria(nodes[6], nodes[7], nodes[11]),
        Tria(nodes[4], nodes[9], nodes[8]),
        Tria(nodes[5], nodes[10], nodes[9]),
        Tria(nodes[6], nodes[11], nodes[10]),

        Tria(nodes[8], nodes[9], nodes[13]),
        Tria(nodes[9], nodes[10], nodes[14]),
        Tria(nodes[10], nodes[11], nodes[15]),
        Tria(nodes[8], nodes[13], nodes[12]),
        Tria(nodes[9], nodes[14], nodes[13]),
        Tria(nodes[10], nodes[15], nodes[14]),
       ]

edges = np.array([
    Edge(nodes[0], nodes[1]),
    Edge(nodes[1], nodes[2]),
    Edge(nodes[2], nodes[3]),

    Edge(nodes[0], nodes[4]),
    Edge(nodes[1], nodes[5]),
    Edge(nodes[2], nodes[6]),
    Edge(nodes[3], nodes[7]),

    Edge(nodes[0], nodes[5]),
    Edge(nodes[1], nodes[6]),
    Edge(nodes[2], nodes[7]),


    Edge(nodes[4], nodes[5]),
    Edge(nodes[5], nodes[6]),
    Edge(nodes[6], nodes[7]),

    Edge(nodes[4], nodes[8]),
    Edge(nodes[5], nodes[9]),
    Edge(nodes[6], nodes[10]),
    Edge(nodes[7], nodes[11]),

    Edge(nodes[4], nodes[9]),
    Edge(nodes[5], nodes[10]),
    Edge(nodes[6], nodes[11]),

    Edge(nodes[8], nodes[9]),
    Edge(nodes[9], nodes[10]),
    Edge(nodes[10], nodes[11]),

    Edge(nodes[8], nodes[12]),
    Edge(nodes[9], nodes[13]),
    Edge(nodes[10], nodes[14]),
    Edge(nodes[11], nodes[15]),

    Edge(nodes[8], nodes[13]),
    Edge(nodes[9], nodes[14]),
    Edge(nodes[10], nodes[15]),

    Edge(nodes[12], nodes[13]),
    Edge(nodes[13], nodes[14]),
    Edge(nodes[14], nodes[15]),
    ])


for edge in edges:
    edge.n1.edges.add(edge)
    edge.n2.edges.add(edge)
    plt.plot([edge.n1.pos[0], edge.n2.pos[0]],
             [edge.n1.pos[1], edge.n2.pos[1]], '--k', lw=0.5, mfc=None)
    for tria in trias:
        if len(set(edge.nodes) & set(tria.nodes)) == 2:
            tria.edges.append(edge)
            edge.trias.append(tria)
            for node in tria.nodes:
                node.trias.add(tria)

# __________________________________________________________________
#
# the code above will come from an external triangulation algorithm
# __________________________________________________________________
#
#  FOCUS HERE AND BELOW
# __________________________________________________________________

for edge in edges:
    if len(edge.trias) == 1:
        tria1 = edge.trias[0]
        tria2 = None
        othernode1 = (set(tria1.nodes) - set(edge.nodes)).pop()
        mid1 = tria1.getMid()
    elif len(edge.trias) == 2:
        tria1 = edge.trias[0]
        tria2 = edge.trias[1]
        othernode1 = (set(tria1.nodes) - set(edge.nodes)).pop()
        othernode2 = (set(tria2.nodes) - set(edge.nodes)).pop()
        mid1 = tria1.getMid()
        mid2 = tria2.getMid()
    else:
        raise NotImplementedError('ntrias != 1 or 2 for an edge')

    node1 = edge.nodes[0]
    node2 = edge.nodes[1]

    ipts = []
    sdomain = []

    # to guarantee outward normals
    sign = 1
    if np.dot(np.cross((mid1 - node2).pos, (node1 - node2).pos), ZGLOBAL) < 0:
        sign = -1

    tmpvec = (node1 - mid1).pos
    nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
    sdomain.append(node1)
    sdomain.append(mid1)
    le = np.sqrt(((node1.pos - mid1.pos)**2).sum())
    ipt = IntegrationPoint(tria1, node1, node2, othernode1, 2/3, 1/6, 1/6, nx, ny, nz, le)
    ipts.append(ipt)

    #NOTE check only if for distorted meshes these ratios 2/3, 1/6, 1/6 are
    #     still valid
    #ipt = ipts[-1]
    #A1 = area_of_polygon([node1.pos[0], node2.pos[0], othernode1.pos[0]],
                         #[node1.pos[1], node2.pos[1], othernode1.pos[1]])
    #fA1 = area_of_polygon([ipt.pos[0], node2.pos[0], othernode1.pos[0]],
                          #[ipt.pos[1], node2.pos[1], othernode1.pos[1]])
    #print('DEBUG area: %1.3f = %1.3f' % (fA1/A1, 2/3))


    tmpvec = (mid1 - node2).pos
    nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
    sdomain.append(node2)
    le = np.sqrt(((mid1.pos - node2.pos)**2).sum())
    ipt = IntegrationPoint(tria1, node1, node2, othernode1, 1/6, 2/3, 1/6, nx, ny, nz, le)
    ipts.append(ipt)

    if tria2 is None:
        tmpvec = (node2 - node1).pos
        nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
        sdomain.append(node1)
        le = np.sqrt(((node2.pos - node1.pos)**2).sum())
        ipt = IntegrationPoint(tria1, node1, node2, othernode1, 1/2, 1/2, 0, nx, ny, nz, le)
        ipts.append(ipt)
    else:
        tmpvec = (node2 - mid2).pos
        nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
        sdomain.append(mid2)
        le = np.sqrt(((node2.pos - mid2.pos)**2).sum())
        ipt = IntegrationPoint(tria2, node1, node2, othernode2, 1/6, 2/3, 1/6, nx, ny, nz, le)
        ipts.append(ipt)

        tmpvec = (mid2 - node1).pos
        nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
        sdomain.append(node1)
        le = np.sqrt(((mid2.pos - node1.pos)**2).sum())
        ipt = IntegrationPoint(tria2, node1, node2, othernode2, 2/3, 1/6, 1/6, nx, ny, nz, le)
        ipts.append(ipt)

    edge.sdomain = sdomain
    edge.Ac = area_of_polygon([sr.pos[0] for sr in sdomain[:-1]],
                              [sr.pos[1] for sr in sdomain[:-1]])
    edge.ipts = ipts

    xcoord = [pt.pos[0] for pt in sdomain]
    ycoord = [pt.pos[1] for pt in sdomain]
    plt.plot(xcoord, ycoord, '-k', lw=0.25)

    xcoord = [ipt.pos[0] for ipt in ipts]
    ycoord = [ipt.pos[1] for ipt in ipts]
    plt.plot(xcoord, ycoord, 'xk', mew=0.25, mfc='None')


xcord = [node.pos[0] for node in nodes]
ycord = [node.pos[1] for node in nodes]
plt.scatter(xcord, ycord)
plt.gca().set_aspect('equal')

# ASSEMBLYING GLOBAL MATRICES

# renumbering nodes using Liu's suggested algorithm
# - sorting nodes from a minimum spatial position to a maximum one
# - Node oject will carry an index that will position it in the global
#   stiffness matrix
nodes_xyz = np.array([n.pos for n in nodes])
index_ref_point = nodes_xyz.min(axis=0)

index_dist = ((nodes_xyz - index_ref_point)**2).sum(axis=-1)
indices = np.argsort(index_dist)
ind2node = {}
for i, node in enumerate(nodes):
    node.index = i
    #node.index = indices[i]
    #ind2node[node.index] = node

n = nodes.shape[0]
dof = 5

# material properties
E11 = 71.e9
nu = 0.33
plyt = 0.01
lam = read_stack([0], plyt=plyt, laminaprop=(E11, E11, nu))
prop = Property(lam.A, lam.B, lam.D, lam.E)
for tria in trias:
    tria.prop = prop

#TODO allocate less memory here...
k0 = np.zeros((n*dof, n*dof), dtype=np.float64)

prop_from_node = False
count = 0
Atotal = 0


for edge in edges:
    n1 = edge.nodes[0]
    n2 = edge.nodes[1]
    Ac = edge.Ac
    Atotal += Ac
    ipts = edge.ipts
    for ipt in ipts:
        if True:
            # plotting arrows indicating normals for each integration point
            A1 = area_of_polygon([tria.n1.pos[0], tria.n2.pos[0], tria.n3.pos[0]],
                                 [tria.n1.pos[1], tria.n2.pos[1], tria.n3.pos[1]])
            f = 0.05
            la = A1**0.5/8
            plt.arrow(ipt.pos[0]-la*ipt.nx, ipt.pos[1]-la*ipt.ny,
                      f*ipt.nx, f*ipt.ny, head_width=0.05,
                      head_length=0.05, fc='k', ec='k')

    indices = set()
    for ipt in ipts:
        indices.add(ipt.n1.index)
        indices.add(ipt.n2.index)
        indices.add(ipt.n3.index)
    indices = sorted(list(indices))
    if len(ipts) == 3:
        indices.append(-1) # fourth dummy index
    indexpos = dict([[ind, i] for i, ind in enumerate(indices)])
    i1, i2, i3, i4 = indices
    f1 = [0, 0, 0, 0]
    f2 = [0, 0, 0, 0]
    f3 = [0, 0, 0, 0]
    f4 = [0, 0, 0, 0]

    nx1 = ipts[0].nx
    ny1 = ipts[0].ny
    le1 = ipts[0].le
    f1[indexpos[ipts[0].n1.index]] = ipts[0].f1
    f1[indexpos[ipts[0].n2.index]] = ipts[0].f2
    f1[indexpos[ipts[0].n3.index]] = ipts[0].f3

    nx2 = ipts[1].nx
    ny2 = ipts[1].ny
    le2 = ipts[1].le
    f21 = ipts[1].f1
    f22 = ipts[1].f2
    f23 = ipts[1].f3
    f2[indexpos[ipts[1].n1.index]] = ipts[1].f1
    f2[indexpos[ipts[1].n2.index]] = ipts[1].f2
    f2[indexpos[ipts[1].n3.index]] = ipts[1].f3

    nx3 = ipts[2].nx
    ny3 = ipts[2].ny
    le3 = ipts[2].le
    f3[indexpos[ipts[2].n1.index]] = ipts[2].f1
    f3[indexpos[ipts[2].n2.index]] = ipts[2].f2
    f3[indexpos[ipts[2].n3.index]] = ipts[2].f3

    if len(ipts) == 3:
        nx4 = 0
        ny4 = 0
        le4 = 0
    else:
        nx4 = ipts[3].nx
        ny4 = ipts[3].ny
        le4 = ipts[3].le
        f4[indexpos[ipts[3].n1.index]] = ipts[3].f1
        f4[indexpos[ipts[3].n2.index]] = ipts[3].f2
        f4[indexpos[ipts[3].n3.index]] = ipts[3].f3

    f11, f12, f13, f14 = f1
    f21, f22, f23, f24 = f2
    f31, f32, f33, f34 = f3
    f41, f42, f43, f44 = f4

    #FIXME do some weighted average on A, B, D
    # either use properties from tria or nodes
    if prop_from_node:
        pass
        #A = f1*ipt.n1.prop.A + f2*ipt.n2.prop.A + f3*ipt.n3.prop.A
        #B = f1*ipt.n1.prop.B + f2*ipt.n2.prop.B + f3*ipt.n3.prop.B
        #D = f1*ipt.n1.prop.D + f2*ipt.n2.prop.D + f3*ipt.n3.prop.D
        #E = f1*ipt.n1.prop.E + f2*ipt.n2.prop.E + f3*ipt.n3.prop.E
    else:
        A = ipts[0].tria.prop.A
        B = ipts[0].tria.prop.B
        D = ipts[0].tria.prop.D

    A11 = A[0, 0]
    A12 = A[0, 1]
    A16 = A[0, 2]
    A22 = A[1, 1]
    A26 = A[1, 2]
    A66 = A[2, 2]
    B11 = B[0, 0]
    B12 = B[0, 1]
    B16 = B[0, 2]
    B22 = B[1, 1]
    B26 = B[1, 2]
    B66 = B[2, 2]
    D11 = D[0, 0]
    D12 = D[0, 1]
    D16 = D[0, 2]
    D22 = D[1, 1]
    D26 = D[1, 2]
    D66 = D[2, 2]

    #TODO calculate only upper triangle
    k0[i1*dof+0, i1*dof+0] += Ac*((A11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+0, i1*dof+1] += Ac*((A12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+0, i1*dof+3] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+0, i1*dof+4] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+0, i2*dof+0] += Ac*((A11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+0, i2*dof+1] += Ac*((A12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+0, i2*dof+3] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+0, i2*dof+4] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+0, i3*dof+0] += Ac*((A11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+0, i3*dof+1] += Ac*((A12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+0, i3*dof+3] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+0, i3*dof+4] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+0, i4*dof+0] += Ac*((A11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+0, i4*dof+1] += Ac*((A12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + A66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i1*dof+0, i4*dof+3] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+0, i4*dof+4] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i1*dof+1, i1*dof+0] += Ac*((A12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+1, i1*dof+1] += Ac*((A22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+1, i1*dof+3] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+1, i1*dof+4] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+1, i2*dof+0] += Ac*((A12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+1, i2*dof+1] += Ac*((A22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+1, i2*dof+3] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+1, i2*dof+4] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+1, i3*dof+0] += Ac*((A12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+1, i3*dof+1] += Ac*((A22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+1, i3*dof+3] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+1, i3*dof+4] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+1, i4*dof+0] += Ac*((A12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+1, i4*dof+1] += Ac*((A22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + A66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i1*dof+1, i4*dof+3] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+1, i4*dof+4] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i1*dof+3, i1*dof+0] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+3, i1*dof+1] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+3, i1*dof+3] += Ac*((D11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+3, i1*dof+4] += Ac*((D12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+3, i2*dof+0] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+3, i2*dof+1] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+3, i2*dof+3] += Ac*((D11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+3, i2*dof+4] += Ac*((D12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+3, i3*dof+0] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+3, i3*dof+1] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+3, i3*dof+3] += Ac*((D11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+3, i3*dof+4] += Ac*((D12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+3, i4*dof+0] += Ac*((B11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+3, i4*dof+1] += Ac*((B12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + B66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i1*dof+3, i4*dof+3] += Ac*((D11*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D16*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+3, i4*dof+4] += Ac*((D12*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + D66*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i1*dof+4, i1*dof+0] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+4, i1*dof+1] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+4, i1*dof+3] += Ac*((D12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i1*dof+4, i1*dof+4] += Ac*((D22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i1*dof+4, i2*dof+0] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+4, i2*dof+1] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+4, i2*dof+3] += Ac*((D12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i1*dof+4, i2*dof+4] += Ac*((D22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i1*dof+4, i3*dof+0] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+4, i3*dof+1] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+4, i3*dof+3] += Ac*((D12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i1*dof+4, i3*dof+4] += Ac*((D22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i1*dof+4, i4*dof+0] += Ac*((B12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+4, i4*dof+1] += Ac*((B22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + B66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i1*dof+4, i4*dof+3] += Ac*((D12*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D16*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i1*dof+4, i4*dof+4] += Ac*((D22*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D26*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D26*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + D66*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+0, i1*dof+0] += Ac*((A11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+0, i1*dof+1] += Ac*((A12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+0, i1*dof+3] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+0, i1*dof+4] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+0, i2*dof+0] += Ac*((A11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+0, i2*dof+1] += Ac*((A12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+0, i2*dof+3] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+0, i2*dof+4] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+0, i3*dof+0] += Ac*((A11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+0, i3*dof+1] += Ac*((A12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+0, i3*dof+3] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+0, i3*dof+4] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+0, i4*dof+0] += Ac*((A11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+0, i4*dof+1] += Ac*((A12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + A66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+0, i4*dof+3] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+0, i4*dof+4] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+1, i1*dof+0] += Ac*((A12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+1, i1*dof+1] += Ac*((A22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+1, i1*dof+3] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+1, i1*dof+4] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+1, i2*dof+0] += Ac*((A12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+1, i2*dof+1] += Ac*((A22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+1, i2*dof+3] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+1, i2*dof+4] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+1, i3*dof+0] += Ac*((A12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+1, i3*dof+1] += Ac*((A22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+1, i3*dof+3] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+1, i3*dof+4] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+1, i4*dof+0] += Ac*((A12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+1, i4*dof+1] += Ac*((A22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + A66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+1, i4*dof+3] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+1, i4*dof+4] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+3, i1*dof+0] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+3, i1*dof+1] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+3, i1*dof+3] += Ac*((D11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+3, i1*dof+4] += Ac*((D12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+3, i2*dof+0] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+3, i2*dof+1] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+3, i2*dof+3] += Ac*((D11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+3, i2*dof+4] += Ac*((D12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+3, i3*dof+0] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+3, i3*dof+1] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+3, i3*dof+3] += Ac*((D11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+3, i3*dof+4] += Ac*((D12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+3, i4*dof+0] += Ac*((B11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+3, i4*dof+1] += Ac*((B12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + B66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+3, i4*dof+3] += Ac*((D11*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D16*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+3, i4*dof+4] += Ac*((D12*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + D66*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+4, i1*dof+0] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+4, i1*dof+1] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+4, i1*dof+3] += Ac*((D12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i2*dof+4, i1*dof+4] += Ac*((D22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i2*dof+4, i2*dof+0] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+4, i2*dof+1] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+4, i2*dof+3] += Ac*((D12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i2*dof+4, i2*dof+4] += Ac*((D22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i2*dof+4, i3*dof+0] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+4, i3*dof+1] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+4, i3*dof+3] += Ac*((D12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i2*dof+4, i3*dof+4] += Ac*((D22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i2*dof+4, i4*dof+0] += Ac*((B12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+4, i4*dof+1] += Ac*((B22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + B66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i2*dof+4, i4*dof+3] += Ac*((D12*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D16*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i2*dof+4, i4*dof+4] += Ac*((D22*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D26*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D26*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + D66*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+0, i1*dof+0] += Ac*((A11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+0, i1*dof+1] += Ac*((A12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+0, i1*dof+3] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+0, i1*dof+4] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+0, i2*dof+0] += Ac*((A11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+0, i2*dof+1] += Ac*((A12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+0, i2*dof+3] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+0, i2*dof+4] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+0, i3*dof+0] += Ac*((A11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+0, i3*dof+1] += Ac*((A12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+0, i3*dof+3] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+0, i3*dof+4] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+0, i4*dof+0] += Ac*((A11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+0, i4*dof+1] += Ac*((A12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + A66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+0, i4*dof+3] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+0, i4*dof+4] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+1, i1*dof+0] += Ac*((A12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+1, i1*dof+1] += Ac*((A22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+1, i1*dof+3] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+1, i1*dof+4] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+1, i2*dof+0] += Ac*((A12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+1, i2*dof+1] += Ac*((A22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+1, i2*dof+3] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+1, i2*dof+4] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+1, i3*dof+0] += Ac*((A12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+1, i3*dof+1] += Ac*((A22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+1, i3*dof+3] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+1, i3*dof+4] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+1, i4*dof+0] += Ac*((A12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+1, i4*dof+1] += Ac*((A22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + A66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+1, i4*dof+3] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+1, i4*dof+4] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+3, i1*dof+0] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+3, i1*dof+1] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+3, i1*dof+3] += Ac*((D11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+3, i1*dof+4] += Ac*((D12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+3, i2*dof+0] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+3, i2*dof+1] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+3, i2*dof+3] += Ac*((D11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+3, i2*dof+4] += Ac*((D12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+3, i3*dof+0] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+3, i3*dof+1] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+3, i3*dof+3] += Ac*((D11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+3, i3*dof+4] += Ac*((D12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+3, i4*dof+0] += Ac*((B11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+3, i4*dof+1] += Ac*((B12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + B66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+3, i4*dof+3] += Ac*((D11*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D16*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+3, i4*dof+4] += Ac*((D12*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + D66*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+4, i1*dof+0] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+4, i1*dof+1] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+4, i1*dof+3] += Ac*((D12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i3*dof+4, i1*dof+4] += Ac*((D22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i3*dof+4, i2*dof+0] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+4, i2*dof+1] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+4, i2*dof+3] += Ac*((D12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i3*dof+4, i2*dof+4] += Ac*((D22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i3*dof+4, i3*dof+0] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+4, i3*dof+1] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+4, i3*dof+3] += Ac*((D12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i3*dof+4, i3*dof+4] += Ac*((D22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i3*dof+4, i4*dof+0] += Ac*((B12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+4, i4*dof+1] += Ac*((B22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + B66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i3*dof+4, i4*dof+3] += Ac*((D12*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D16*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i3*dof+4, i4*dof+4] += Ac*((D22*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D26*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D26*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + D66*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+0, i1*dof+0] += Ac*((A11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+0, i1*dof+1] += Ac*((A12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+0, i1*dof+3] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+0, i1*dof+4] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+0, i2*dof+0] += Ac*((A11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+0, i2*dof+1] += Ac*((A12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+0, i2*dof+3] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+0, i2*dof+4] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+0, i3*dof+0] += Ac*((A11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+0, i3*dof+1] += Ac*((A12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+0, i3*dof+3] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+0, i3*dof+4] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+0, i4*dof+0] += Ac*((A11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+0, i4*dof+1] += Ac*((A12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + A66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+0, i4*dof+3] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+0, i4*dof+4] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+1, i1*dof+0] += Ac*((A12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+1, i1*dof+1] += Ac*((A22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+1, i1*dof+3] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+1, i1*dof+4] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+1, i2*dof+0] += Ac*((A12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+1, i2*dof+1] += Ac*((A22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+1, i2*dof+3] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+1, i2*dof+4] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+1, i3*dof+0] += Ac*((A12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+1, i3*dof+1] += Ac*((A22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+1, i3*dof+3] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+1, i3*dof+4] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+1, i4*dof+0] += Ac*((A12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+1, i4*dof+1] += Ac*((A22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (A26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + A66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+1, i4*dof+3] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+1, i4*dof+4] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+3, i1*dof+0] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+3, i1*dof+1] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+3, i1*dof+3] += Ac*((D11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+3, i1*dof+4] += Ac*((D12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+3, i2*dof+0] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+3, i2*dof+1] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+3, i2*dof+3] += Ac*((D11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+3, i2*dof+4] += Ac*((D12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+3, i3*dof+0] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+3, i3*dof+1] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+3, i3*dof+3] += Ac*((D11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+3, i3*dof+4] += Ac*((D12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+3, i4*dof+0] += Ac*((B11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+3, i4*dof+1] += Ac*((B12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + B66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+3, i4*dof+3] += Ac*((D11*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D16*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+3, i4*dof+4] += Ac*((D12*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + D66*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+4, i1*dof+0] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+4, i1*dof+1] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+4, i1*dof+3] += Ac*((D12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
    k0[i4*dof+4, i1*dof+4] += Ac*((D22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac)
    k0[i4*dof+4, i2*dof+0] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+4, i2*dof+1] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+4, i2*dof+3] += Ac*((D12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
    k0[i4*dof+4, i2*dof+4] += Ac*((D22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac)
    k0[i4*dof+4, i3*dof+0] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+4, i3*dof+1] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+4, i3*dof+3] += Ac*((D12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
    k0[i4*dof+4, i3*dof+4] += Ac*((D22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac)
    k0[i4*dof+4, i4*dof+0] += Ac*((B12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+4, i4*dof+1] += Ac*((B22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (B26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + B66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)
    k0[i4*dof+4, i4*dof+3] += Ac*((D12*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D16*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
    k0[i4*dof+4, i4*dof+4] += Ac*((D22*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D26*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + (D26*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac + D66*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac)

    if len(edge.trias) == 1:
        tria = edge.trias[0]
        if not tria.n1 in edge.nodes:
            other1 = tria.n1
        elif not tria.n2 in edge.nodes:
            other1 = tria.n2
        elif not tria.n3 in edge.nodes:
            other1 = tria.n3
        else:
            raise RuntimeError('Invalid connectivity tria-edge')

        if np.dot(np.cross((n2 - n1).pos, (other1 - n1).pos), ZGLOBAL) > 0:
            tn1, tn2 = n1, n2
        else:
            tn1, tn2 = n2, n1

        x1, y1, z1 = tn1.pos
        x2, y2, z2 = tn2.pos
        x3, y3, z3 = other1.pos
        a = x2 - x1
        b = y2 - y1
        c = x3 - x1
        d = y3 - y1
        Ae = tria.Ae

        #TODO interpolate nodal properties when used
        if prop_from_node:
            raise NotImplementedError('')
        k = 5/6 #TODO!!!

        E = k * tria.prop.E
        E44 = E[0, 0]
        E45 = E[0, 1]
        E55 = E[1, 1]

        i1 = tn1.index
        i2 = tn2.index
        i3 = other1.index

        k0[i1*dof+2, i1*dof+2] += (-(a - c)*(E45*(b - d) - E55*(a - c)) + (b - d)*(E44*(b - d) - E45*(a - c)))/(4*(Ae*Ae))
        k0[i1*dof+2, i1*dof+3] += (E44*(b - d) - E45*(a - c))/(4*Ae)
        k0[i1*dof+2, i1*dof+4] += (E45*(b - d) - E55*(a - c))/(4*Ae)
        k0[i1*dof+2, i2*dof+2] += (-c*(E45*(b - d) - E55*(a - c)) + d*(E44*(b - d) - E45*(a - c)))/(4*(Ae*Ae))
        k0[i1*dof+2, i2*dof+3] += a*(-c*(E45*(b - d) - E55*(a - c)) + d*(E44*(b - d) - E45*(a - c)))/(8*(Ae*Ae))
        k0[i1*dof+2, i2*dof+4] += b*(-c*(E45*(b - d) - E55*(a - c)) + d*(E44*(b - d) - E45*(a - c)))/(8*(Ae*Ae))
        k0[i1*dof+2, i3*dof+2] += (a*(E45*(b - d) - E55*(a - c)) - b*(E44*(b - d) - E45*(a - c)))/(4*(Ae*Ae))
        k0[i1*dof+2, i3*dof+3] += c*(a*(E45*(b - d) - E55*(a - c)) - b*(E44*(b - d) - E45*(a - c)))/(8*(Ae*Ae))
        k0[i1*dof+2, i3*dof+4] += d*(a*(E45*(b - d) - E55*(a - c)) - b*(E44*(b - d) - E45*(a - c)))/(8*(Ae*Ae))
        k0[i1*dof+3, i1*dof+2] += (E44*(b - d) - E45*(a - c))/(4*Ae)
        k0[i1*dof+3, i1*dof+3] += E44/4
        k0[i1*dof+3, i1*dof+4] += E45/4
        k0[i1*dof+3, i2*dof+2] += (E44*d - E45*c)/(4*Ae)
        k0[i1*dof+3, i2*dof+3] += a*(E44*d - E45*c)/(8*Ae)
        k0[i1*dof+3, i2*dof+4] += b*(E44*d - E45*c)/(8*Ae)
        k0[i1*dof+3, i3*dof+2] += (-E44*b + E45*a)/(4*Ae)
        k0[i1*dof+3, i3*dof+3] += c*(-E44*b + E45*a)/(8*Ae)
        k0[i1*dof+3, i3*dof+4] += d*(-E44*b + E45*a)/(8*Ae)
        k0[i1*dof+4, i1*dof+2] += (E45*(b - d) - E55*(a - c))/(4*Ae)
        k0[i1*dof+4, i1*dof+3] += E45/4
        k0[i1*dof+4, i1*dof+4] += E55/4
        k0[i1*dof+4, i2*dof+2] += (E45*d - E55*c)/(4*Ae)
        k0[i1*dof+4, i2*dof+3] += a*(E45*d - E55*c)/(8*Ae)
        k0[i1*dof+4, i2*dof+4] += b*(E45*d - E55*c)/(8*Ae)
        k0[i1*dof+4, i3*dof+2] += (-E45*b + E55*a)/(4*Ae)
        k0[i1*dof+4, i3*dof+3] += c*(-E45*b + E55*a)/(8*Ae)
        k0[i1*dof+4, i3*dof+4] += d*(-E45*b + E55*a)/(8*Ae)
        k0[i2*dof+2, i1*dof+2] += (-(a - c)*(E45*d - E55*c) + (b - d)*(E44*d - E45*c))/(4*(Ae*Ae))
        k0[i2*dof+2, i1*dof+3] += (E44*d - E45*c)/(4*Ae)
        k0[i2*dof+2, i1*dof+4] += (E45*d - E55*c)/(4*Ae)
        k0[i2*dof+2, i2*dof+2] += (-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(4*(Ae*Ae))
        k0[i2*dof+2, i2*dof+3] += a*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+2, i2*dof+4] += b*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+2, i3*dof+2] += (a*(E45*d - E55*c) - b*(E44*d - E45*c))/(4*(Ae*Ae))
        k0[i2*dof+2, i3*dof+3] += c*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+2, i3*dof+4] += d*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+3, i1*dof+2] += a*(-(a - c)*(E45*d - E55*c) + (b - d)*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+3, i1*dof+3] += a*(E44*d - E45*c)/(8*Ae)
        k0[i2*dof+3, i1*dof+4] += a*(E45*d - E55*c)/(8*Ae)
        k0[i2*dof+3, i2*dof+2] += a*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+3, i2*dof+3] += (a*a)*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i2*dof+3, i2*dof+4] += a*b*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i2*dof+3, i3*dof+2] += a*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+3, i3*dof+3] += a*c*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i2*dof+3, i3*dof+4] += a*d*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i2*dof+4, i1*dof+2] += b*(-(a - c)*(E45*d - E55*c) + (b - d)*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+4, i1*dof+3] += b*(E44*d - E45*c)/(8*Ae)
        k0[i2*dof+4, i1*dof+4] += b*(E45*d - E55*c)/(8*Ae)
        k0[i2*dof+4, i2*dof+2] += b*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+4, i2*dof+3] += a*b*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i2*dof+4, i2*dof+4] += (b*b)*(-c*(E45*d - E55*c) + d*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i2*dof+4, i3*dof+2] += b*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(8*(Ae*Ae))
        k0[i2*dof+4, i3*dof+3] += b*c*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i2*dof+4, i3*dof+4] += b*d*(a*(E45*d - E55*c) - b*(E44*d - E45*c))/(16*(Ae*Ae))
        k0[i3*dof+2, i1*dof+2] += ((a - c)*(E45*b - E55*a) - (b - d)*(E44*b - E45*a))/(4*(Ae*Ae))
        k0[i3*dof+2, i1*dof+3] += (-E44*b + E45*a)/(4*Ae)
        k0[i3*dof+2, i1*dof+4] += (-E45*b + E55*a)/(4*Ae)
        k0[i3*dof+2, i2*dof+2] += (c*(E45*b - E55*a) - d*(E44*b - E45*a))/(4*(Ae*Ae))
        k0[i3*dof+2, i2*dof+3] += a*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+2, i2*dof+4] += b*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+2, i3*dof+2] += (-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(4*(Ae*Ae))
        k0[i3*dof+2, i3*dof+3] += c*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+2, i3*dof+4] += d*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+3, i1*dof+2] += c*((a - c)*(E45*b - E55*a) - (b - d)*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+3, i1*dof+3] += c*(-E44*b + E45*a)/(8*Ae)
        k0[i3*dof+3, i1*dof+4] += c*(-E45*b + E55*a)/(8*Ae)
        k0[i3*dof+3, i2*dof+2] += c*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+3, i2*dof+3] += a*c*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(16*(Ae*Ae))
        k0[i3*dof+3, i2*dof+4] += b*c*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(16*(Ae*Ae))
        k0[i3*dof+3, i3*dof+2] += c*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+3, i3*dof+3] += (c*c)*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(16*(Ae*Ae))
        k0[i3*dof+3, i3*dof+4] += c*d*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(16*(Ae*Ae))
        k0[i3*dof+4, i1*dof+2] += d*((a - c)*(E45*b - E55*a) - (b - d)*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+4, i1*dof+3] += d*(-E44*b + E45*a)/(8*Ae)
        k0[i3*dof+4, i1*dof+4] += d*(-E45*b + E55*a)/(8*Ae)
        k0[i3*dof+4, i2*dof+2] += d*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+4, i2*dof+3] += a*d*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(16*(Ae*Ae))
        k0[i3*dof+4, i2*dof+4] += b*d*(c*(E45*b - E55*a) - d*(E44*b - E45*a))/(16*(Ae*Ae))
        k0[i3*dof+4, i3*dof+2] += d*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(8*(Ae*Ae))
        k0[i3*dof+4, i3*dof+3] += c*d*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(16*(Ae*Ae))
        k0[i3*dof+4, i3*dof+4] += (d*d)*(-a*(E45*b - E55*a) + b*(E44*b - E45*a))/(16*(Ae*Ae))

    else:
        k1 = k2 = 5/6
        tria1 = edge.trias[0]
        tria2 = edge.trias[1]

        if not tria1.n1 in edge.nodes:
            other1 = tria1.n1
        elif not tria1.n2 in edge.nodes:
            other1 = tria1.n2
        elif not tria1.n3 in edge.nodes:
            other1 = tria1.n3
        else:
            raise RuntimeError('Invalid connectivity tria-edge')

        if not tria2.n1 in edge.nodes:
            other2 = tria2.n1
        elif not tria2.n2 in edge.nodes:
            other2 = tria2.n2
        elif not tria2.n3 in edge.nodes:
            other2 = tria2.n3
        else:
            raise RuntimeError('Invalid connectivity tria-edge')

        if np.dot(np.cross((n2 - n1).pos, (other1 - n1).pos), ZGLOBAL) > 0:
            t1n1, t1n2 = n1, n2
        else:
            t1n1, t1n2 = n2, n1

        x1, y1, z1 = t1n1.pos
        x2, y2, z2 = t1n2.pos
        x3, y3, z3 = other1.pos
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = x3 - x1
        d1 = y3 - y1
        Ae1 = tria1.Ae

        tmp = [n1, n2, tria1.getMid()]
        Ac1 = area_of_polygon([n.pos[0] for n in tmp], [n.pos[1] for n in tmp])

        if np.dot(np.cross((n2 - n1).pos, (other2 - n1).pos), ZGLOBAL) > 0:
            t2n1, t2n2 = n1, n2
        else:
            t2n1, t2n2 = n2, n1
        x1, y1, z1 = t2n1.pos
        x2, y2, z2 = t2n2.pos
        x3, y3, z3 = other2.pos
        a2 = x2 - x1
        b2 = y2 - y1
        c2 = x3 - x1
        d2 = y3 - y1
        Ae2 = tria2.Ae

        tmp = [n1, n2, tria2.getMid()]
        Ac2 = area_of_polygon([n.pos[0] for n in tmp], [n.pos[1] for n in tmp])

        E = k1*Ac1/Ac*tria1.prop.E + k2*Ac2/Ac*tria2.prop.E
        E44 = E[0, 0]
        E45 = E[0, 1]
        E55 = E[1, 1]

        i1 = n1.index
        i2 = n2.index
        i3 = other1.index
        i4 = other2.index

        k0[i1*dof+2, i1*dof+2] += ((E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - (E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i1*dof+2, i1*dof+3] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+2, i1*dof+4] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+2, i2*dof+2] += ((E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - (E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i1*dof+2, i2*dof+3] += ((E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - (E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i1*dof+2, i2*dof+4] += ((E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - (E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i1*dof+2, i3*dof+2] += Ac1*(a1*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))) - b1*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))))/(4*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i1*dof+2, i3*dof+3] += Ac1*c1*(a1*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))) - b1*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i1*dof+2, i3*dof+4] += Ac1*d1*(a1*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))) - b1*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i1*dof+2, i4*dof+2] += Ac2*(a2*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))) - b2*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))))/(4*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i1*dof+2, i4*dof+3] += Ac2*c2*(a2*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))) - b2*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i1*dof+2, i4*dof+4] += Ac2*d2*(a2*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))) - b2*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2))))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i1*dof+3, i1*dof+2] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E45*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+3, i1*dof+3] += E44*(Ac1 + Ac2)**2/(4*(Ac*Ac))
        k0[i1*dof+3, i1*dof+4] += E45*(Ac1 + Ac2)**2/(4*(Ac*Ac))
        k0[i1*dof+3, i2*dof+2] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+3, i2*dof+3] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+3, i2*dof+4] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+3, i3*dof+2] += Ac1*(Ac1 + Ac2)*(-E44*b1 + E45*a1)/(4*(Ac*Ac)*Ae1)
        k0[i1*dof+3, i3*dof+3] += Ac1*c1*(Ac1 + Ac2)*(-E44*b1 + E45*a1)/(8*(Ac*Ac)*Ae1)
        k0[i1*dof+3, i3*dof+4] += Ac1*d1*(Ac1 + Ac2)*(-E44*b1 + E45*a1)/(8*(Ac*Ac)*Ae1)
        k0[i1*dof+3, i4*dof+2] += Ac2*(Ac1 + Ac2)*(-E44*b2 + E45*a2)/(4*(Ac*Ac)*Ae2)
        k0[i1*dof+3, i4*dof+3] += Ac2*c2*(Ac1 + Ac2)*(-E44*b2 + E45*a2)/(8*(Ac*Ac)*Ae2)
        k0[i1*dof+3, i4*dof+4] += Ac2*d2*(Ac1 + Ac2)*(-E44*b2 + E45*a2)/(8*(Ac*Ac)*Ae2)
        k0[i1*dof+4, i1*dof+2] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - E55*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+4, i1*dof+3] += E45*(Ac1 + Ac2)**2/(4*(Ac*Ac))
        k0[i1*dof+4, i1*dof+4] += E55*(Ac1 + Ac2)**2/(4*(Ac*Ac))
        k0[i1*dof+4, i2*dof+2] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+4, i2*dof+3] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+4, i2*dof+4] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i1*dof+4, i3*dof+2] += Ac1*(Ac1 + Ac2)*(-E45*b1 + E55*a1)/(4*(Ac*Ac)*Ae1)
        k0[i1*dof+4, i3*dof+3] += Ac1*c1*(Ac1 + Ac2)*(-E45*b1 + E55*a1)/(8*(Ac*Ac)*Ae1)
        k0[i1*dof+4, i3*dof+4] += Ac1*d1*(Ac1 + Ac2)*(-E45*b1 + E55*a1)/(8*(Ac*Ac)*Ae1)
        k0[i1*dof+4, i4*dof+2] += Ac2*(Ac1 + Ac2)*(-E45*b2 + E55*a2)/(4*(Ac*Ac)*Ae2)
        k0[i1*dof+4, i4*dof+3] += Ac2*c2*(Ac1 + Ac2)*(-E45*b2 + E55*a2)/(8*(Ac*Ac)*Ae2)
        k0[i1*dof+4, i4*dof+4] += Ac2*d2*(Ac1 + Ac2)*(-E45*b2 + E55*a2)/(8*(Ac*Ac)*Ae2)
        k0[i2*dof+2, i1*dof+2] += ((E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - (E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+2, i1*dof+3] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i2*dof+2, i1*dof+4] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i2*dof+2, i2*dof+2] += ((E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - (E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+2, i2*dof+3] += ((E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - (E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+2, i2*dof+4] += ((E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - (E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2))*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+2, i3*dof+2] += Ac1*(a1*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2)) - b1*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2)))/(4*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+2, i3*dof+3] += Ac1*c1*(a1*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2)) - b1*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+2, i3*dof+4] += Ac1*d1*(a1*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2)) - b1*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+2, i4*dof+2] += Ac2*(a2*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2)) - b2*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2)))/(4*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+2, i4*dof+3] += Ac2*c2*(a2*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2)) - b2*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2)))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+2, i4*dof+4] += Ac2*d2*(a2*(E45*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E55*(Ac1*Ae2*c1 + Ac2*Ae1*c2)) - b2*(E44*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - E45*(Ac1*Ae2*c1 + Ac2*Ae1*c2)))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+3, i1*dof+2] += ((E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - (E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+3, i1*dof+3] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i2*dof+3, i1*dof+4] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i2*dof+3, i2*dof+2] += ((E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - (E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+3, i2*dof+3] += ((E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - (E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+3, i2*dof+4] += ((E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - (E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+3, i3*dof+2] += Ac1*(a1*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)) - b1*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+3, i3*dof+3] += Ac1*c1*(a1*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)) - b1*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+3, i3*dof+4] += Ac1*d1*(a1*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)) - b1*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+3, i4*dof+2] += Ac2*(a2*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)) - b2*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+3, i4*dof+3] += Ac2*c2*(a2*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)) - b2*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+3, i4*dof+4] += Ac2*d2*(a2*(E45*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E55*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)) - b2*(E44*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - E45*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2)))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+4, i1*dof+2] += ((E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) - (E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+4, i1*dof+3] += (Ac1 + Ac2)*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i2*dof+4, i1*dof+4] += (Ac1 + Ac2)*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i2*dof+4, i2*dof+2] += ((E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*d1 + Ac2*Ae1*d2) - (E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+4, i2*dof+3] += ((E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) - (E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+4, i2*dof+4] += ((E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - (E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*(Ae2*Ae2))
        k0[i2*dof+4, i3*dof+2] += Ac1*(a1*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)) - b1*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+4, i3*dof+3] += Ac1*c1*(a1*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)) - b1*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+4, i3*dof+4] += Ac1*d1*(a1*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)) - b1*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i2*dof+4, i4*dof+2] += Ac2*(a2*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)) - b2*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+4, i4*dof+3] += Ac2*c2*(a2*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)) - b2*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i2*dof+4, i4*dof+4] += Ac2*d2*(a2*(E45*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E55*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)) - b2*(E44*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) - E45*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2)))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i3*dof+2, i1*dof+2] += Ac1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) + (E45*b1 - E55*a1)*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+2, i1*dof+3] += -Ac1*(Ac1 + Ac2)*(E44*b1 - E45*a1)/(4*(Ac*Ac)*Ae1)
        k0[i3*dof+2, i1*dof+4] += -Ac1*(Ac1 + Ac2)*(E45*b1 - E55*a1)/(4*(Ac*Ac)*Ae1)
        k0[i3*dof+2, i2*dof+2] += Ac1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*d1 + Ac2*Ae1*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+2, i2*dof+3] += Ac1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+2, i2*dof+4] += Ac1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+2, i3*dof+2] += (Ac1*Ac1)*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(4*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+2, i3*dof+3] += (Ac1*Ac1)*c1*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(8*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+2, i3*dof+4] += (Ac1*Ac1)*d1*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(8*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+2, i4*dof+2] += Ac1*Ac2*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+2, i4*dof+3] += Ac1*Ac2*c2*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+2, i4*dof+4] += Ac1*Ac2*d2*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+3, i1*dof+2] += Ac1*c1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) + (E45*b1 - E55*a1)*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+3, i1*dof+3] += -Ac1*c1*(Ac1 + Ac2)*(E44*b1 - E45*a1)/(8*(Ac*Ac)*Ae1)
        k0[i3*dof+3, i1*dof+4] += -Ac1*c1*(Ac1 + Ac2)*(E45*b1 - E55*a1)/(8*(Ac*Ac)*Ae1)
        k0[i3*dof+3, i2*dof+2] += Ac1*c1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*d1 + Ac2*Ae1*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+3, i2*dof+3] += Ac1*c1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+3, i2*dof+4] += Ac1*c1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+3, i3*dof+2] += (Ac1*Ac1)*c1*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(8*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+3, i3*dof+3] += (Ac1*Ac1)*(c1*c1)*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(16*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+3, i3*dof+4] += (Ac1*Ac1)*c1*d1*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(16*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+3, i4*dof+2] += Ac1*Ac2*c1*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+3, i4*dof+3] += Ac1*Ac2*c1*c2*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+3, i4*dof+4] += Ac1*Ac2*c1*d2*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+4, i1*dof+2] += Ac1*d1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) + (E45*b1 - E55*a1)*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+4, i1*dof+3] += -Ac1*d1*(Ac1 + Ac2)*(E44*b1 - E45*a1)/(8*(Ac*Ac)*Ae1)
        k0[i3*dof+4, i1*dof+4] += -Ac1*d1*(Ac1 + Ac2)*(E45*b1 - E55*a1)/(8*(Ac*Ac)*Ae1)
        k0[i3*dof+4, i2*dof+2] += Ac1*d1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*d1 + Ac2*Ae1*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(8*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+4, i2*dof+3] += Ac1*d1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+4, i2*dof+4] += Ac1*d1*(-(E44*b1 - E45*a1)*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) + (E45*b1 - E55*a1)*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(16*(Ac*Ac)*(Ae1*Ae1)*Ae2)
        k0[i3*dof+4, i3*dof+2] += (Ac1*Ac1)*d1*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(8*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+4, i3*dof+3] += (Ac1*Ac1)*c1*d1*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(16*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+4, i3*dof+4] += (Ac1*Ac1)*(d1*d1)*(-a1*(E45*b1 - E55*a1) + b1*(E44*b1 - E45*a1))/(16*(Ac*Ac)*(Ae1*Ae1))
        k0[i3*dof+4, i4*dof+2] += Ac1*Ac2*d1*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+4, i4*dof+3] += Ac1*Ac2*c2*d1*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i3*dof+4, i4*dof+4] += Ac1*Ac2*d1*d2*(-a2*(E45*b1 - E55*a1) + b2*(E44*b1 - E45*a1))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+2, i1*dof+2] += Ac2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) + (E45*b2 - E55*a2)*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(4*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+2, i1*dof+3] += -Ac2*(Ac1 + Ac2)*(E44*b2 - E45*a2)/(4*(Ac*Ac)*Ae2)
        k0[i4*dof+2, i1*dof+4] += -Ac2*(Ac1 + Ac2)*(E45*b2 - E55*a2)/(4*(Ac*Ac)*Ae2)
        k0[i4*dof+2, i2*dof+2] += Ac2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*d1 + Ac2*Ae1*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(4*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+2, i2*dof+3] += Ac2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+2, i2*dof+4] += Ac2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+2, i3*dof+2] += Ac1*Ac2*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(4*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+2, i3*dof+3] += Ac1*Ac2*c1*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+2, i3*dof+4] += Ac1*Ac2*d1*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+2, i4*dof+2] += (Ac2*Ac2)*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(4*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+2, i4*dof+3] += (Ac2*Ac2)*c2*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(8*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+2, i4*dof+4] += (Ac2*Ac2)*d2*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(8*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+3, i1*dof+2] += Ac2*c2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) + (E45*b2 - E55*a2)*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+3, i1*dof+3] += -Ac2*c2*(Ac1 + Ac2)*(E44*b2 - E45*a2)/(8*(Ac*Ac)*Ae2)
        k0[i4*dof+3, i1*dof+4] += -Ac2*c2*(Ac1 + Ac2)*(E45*b2 - E55*a2)/(8*(Ac*Ac)*Ae2)
        k0[i4*dof+3, i2*dof+2] += Ac2*c2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*d1 + Ac2*Ae1*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+3, i2*dof+3] += Ac2*c2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+3, i2*dof+4] += Ac2*c2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+3, i3*dof+2] += Ac1*Ac2*c2*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+3, i3*dof+3] += Ac1*Ac2*c1*c2*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+3, i3*dof+4] += Ac1*Ac2*c2*d1*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+3, i4*dof+2] += (Ac2*Ac2)*c2*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(8*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+3, i4*dof+3] += (Ac2*Ac2)*(c2*c2)*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(16*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+3, i4*dof+4] += (Ac2*Ac2)*c2*d2*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(16*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+4, i1*dof+2] += Ac2*d2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*(b1 - d1) + Ac2*Ae1*(b2 - d2)) + (E45*b2 - E55*a2)*(Ac1*Ae2*(a1 - c1) + Ac2*Ae1*(a2 - c2)))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+4, i1*dof+3] += -Ac2*d2*(Ac1 + Ac2)*(E44*b2 - E45*a2)/(8*(Ac*Ac)*Ae2)
        k0[i4*dof+4, i1*dof+4] += -Ac2*d2*(Ac1 + Ac2)*(E45*b2 - E55*a2)/(8*(Ac*Ac)*Ae2)
        k0[i4*dof+4, i2*dof+2] += Ac2*d2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*d1 + Ac2*Ae1*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*c1 + Ac2*Ae1*c2))/(8*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+4, i2*dof+3] += Ac2*d2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*a1*d1 + Ac2*Ae1*a2*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*a1*c1 + Ac2*Ae1*a2*c2))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+4, i2*dof+4] += Ac2*d2*(-(E44*b2 - E45*a2)*(Ac1*Ae2*b1*d1 + Ac2*Ae1*b2*d2) + (E45*b2 - E55*a2)*(Ac1*Ae2*b1*c1 + Ac2*Ae1*b2*c2))/(16*(Ac*Ac)*Ae1*(Ae2*Ae2))
        k0[i4*dof+4, i3*dof+2] += Ac1*Ac2*d2*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(8*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+4, i3*dof+3] += Ac1*Ac2*c1*d2*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+4, i3*dof+4] += Ac1*Ac2*d1*d2*(-a1*(E45*b2 - E55*a2) + b1*(E44*b2 - E45*a2))/(16*(Ac*Ac)*Ae1*Ae2)
        k0[i4*dof+4, i4*dof+2] += (Ac2*Ac2)*d2*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(8*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+4, i4*dof+3] += (Ac2*Ac2)*c2*d2*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(16*(Ac*Ac)*(Ae2*Ae2))
        k0[i4*dof+4, i4*dof+4] += (Ac2*Ac2)*(d2*d2)*(-a2*(E45*b2 - E55*a2) + b2*(E44*b2 - E45*a2))/(16*(Ac*Ac)*(Ae2*Ae2))


print('Atotal:', Atotal)

puvw = 2
# force vector
fext = np.zeros(n*dof, dtype=np.float64)
fext[nodes[3].index*dof + puvw] = 500.
fext[nodes[7].index*dof + puvw] = 1000.
fext[nodes[11].index*dof + puvw] = 1000.
fext[nodes[15].index*dof + puvw] = 500.

#k0[i<j] = k0[i>j]
i, j = np.indices(k0.shape)
print('is_symmetric:', np.allclose(k0[i>j], k0[i<j]))
print('is_symmetric:', k0[i>j].sum(), k0[i<j].sum())
test = is_symmetric(coo_matrix(k0))
print('is_symmetric:', test)

# boundary conditions
for i in [0, 4, 8, 12]:
    for j in [0, 1, 2]:
        k0[nodes[i].index*dof+j, :] = 0
        k0[:, nodes[i].index*dof+j] = 0

k0 = coo_matrix(k0)
u = solve(k0, fext, silent=True)

xcord = np.array([node.pos[0] for node in nodes])
ycord = np.array([node.pos[1] for node in nodes])
wmin = u[puvw::dof].min()
wmax = u[puvw::dof].max()
levels = np.linspace(wmin, wmax, 400)
print(u[puvw::dof].reshape(4, 4))
plt.contourf(xcord.reshape(4, 4), ycord.reshape(4, 4), u[puvw::dof].reshape(4, 4),
        levels=levels)

plt.savefig('plot_edge_based_smoothing_domain.png', bbox_inches='tight')

