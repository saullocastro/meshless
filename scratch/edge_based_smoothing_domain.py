from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

from meshless.composite.laminate import read_stack
from meshless.sparse import solve

XGLOBAL = np.array([1., 0, 0])
YGLOBAL = np.array([0, 1., 0])


def unit_vector(vector):
    """Return the unit vector
    """
    return vector / np.linalg.norm(vector)


def cosvec(v1, v2):
    """Return the cos between vectors 'v1' and 'v2'

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


class Property(object):
    def __init__(self, A, B, D, E):
        self.A = A
        self.B = B
        self.D = D
        self.E = E


class IntegrationPoint(object):
    def __init__(self, pos, tria, n1, n2, n3, f1, f2, f3):
        self.pos = pos
        self.tria = tria
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3


class Tria(object):
    def __init__(self, n1, n2, n3):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.edges = []
        self.nodes = [n1, n2, n3]
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
        self.nx = 0
        self.ny = 0
        self.nz = 0 # 3D case

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


nodes = np.array([
        Node(0, 0, 0),
        Node(1.5, -0.5, 0),
        Node(3, 0, 0),

        Node(0.25, 1, 0),
        Node(1.5, 1.5, 0),
        Node(2.5, 1, 0),

        Node(0, 3, 0),
        Node(1.5, 3, 0),
        Node(3, 3, 0),
        ])

trias = [
        Tria(nodes[0], nodes[1], nodes[4]),
        Tria(nodes[1], nodes[2], nodes[5]),
        Tria(nodes[0], nodes[4], nodes[3]),
        Tria(nodes[1], nodes[5], nodes[4]),

        Tria(nodes[3], nodes[4], nodes[7]),
        Tria(nodes[4], nodes[5], nodes[8]),
        Tria(nodes[3], nodes[7], nodes[6]),
        Tria(nodes[4], nodes[8], nodes[7]),
       ]


edges = np.array([
        Edge(nodes[0], nodes[1]),
        Edge(nodes[1], nodes[2]),
        Edge(nodes[0], nodes[3]),
        Edge(nodes[0], nodes[4]),
        Edge(nodes[1], nodes[4]),
        Edge(nodes[1], nodes[5]),
        Edge(nodes[2], nodes[5]),

        Edge(nodes[3], nodes[4]),
        Edge(nodes[4], nodes[5]),
        Edge(nodes[3], nodes[6]),
        Edge(nodes[3], nodes[7]),
        Edge(nodes[4], nodes[7]),
        Edge(nodes[4], nodes[8]),
        Edge(nodes[5], nodes[8]),

        Edge(nodes[6], nodes[7]),
        Edge(nodes[7], nodes[8]),
        ])

for edge in edges:
    edge.n1.edges.add(edge)
    edge.n2.edges.add(edge)
    plt.plot([edge.n1.pos[0], edge.n2.pos[0]],
             [edge.n1.pos[1], edge.n2.pos[1]], '--r', mfc=None)
    for tria in trias:
        if len(set(edge.nodes) & set(tria.nodes)) == 2:
            tria.edges.append(edge)
            edge.trias.append(tria)
            for node in tria.nodes:
                node.trias.add(tria)

# the code above will come from an external triangulation algorithm

colors = cycle(['g', 'k', 'y', 'b'])

for edge in edges[...]:

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

    edgevec = (node2 - node1).pos
    edge.nx = cosvec(XGLOBAL, edgevec)
    edge.ny = cosvec(edgevec, YGLOBAL)

    color = next(colors)

    ipts = []
    sdomain = []

    sdomain.append(node1)
    sdomain.append(mid1)
    ipts.append(IntegrationPoint(0.5*(node1.pos + mid1.pos), tria1,
            node1, node2, othernode1, 2/3, 1/6, 1/6))

    sdomain.append(node2)
    ipts.append(IntegrationPoint(0.5*(mid1.pos + node2.pos), tria1,
            node1, node2, othernode1, 1/6, 2/3, 1/6))

    if tria2 is None:
        sdomain.append(node1)
        ipts.append(IntegrationPoint(0.5*(node2.pos + node1.pos), tria1,
            node1, node2, othernode1, 1/2, 1/2, 0))
    else:
        sdomain.append(mid2)
        ipts.append(IntegrationPoint(0.5*(node2.pos + mid2.pos), tria2,
            node1, node2, othernode2, 1/6, 2/3, 1/6))
        sdomain.append(node1)
        ipts.append(IntegrationPoint(0.5*(mid2.pos + node1.pos), tria2,
            node1, node2, othernode2, 2/3, 1/6, 1/6))

    edge.sdomain = sdomain
    edge.ipts = ipts

    xcoord = [pt.pos[0] for pt in sdomain]
    ycoord = [pt.pos[1] for pt in sdomain]
    plt.plot(xcoord, ycoord, '-' + color)

    xcoord = [ipt.pos[0] for ipt in ipts]
    ycoord = [ipt.pos[1] for ipt in ipts]
    plt.plot(xcoord, ycoord, 'og', mew=1., mfc='None')


xcord = [node.pos[0] for node in nodes]
ycord = [node.pos[1] for node in nodes]
plt.scatter(xcord, ycord)
plt.gca().set_aspect('equal')

plt.savefig('plot_edge_based_smoothing_domain.png', bbox_inches='tight')

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
    node.index = indices[i]
    ind2node[node.index] = node

n = nodes.shape[0]
dof = 5

# material properties
lam = read_stack([0], plyt=0.001, laminaprop=(71e9, 71e9, 0.33))
prop = Property(lam.A, lam.B, lam.D, lam.E)
for tria in trias:
    tria.prop = prop

# force vector
F = np.zeros(n*dof, dtype=np.float64)
F[nodes[0].index*dof + 0] = 100.

# boundary conditions


#TODO allocate less memory here...
K0 = np.zeros((n*dof, n*dof), dtype=np.float64)

prop_from_node = False
for edge in edges:
    nx = edge.nx
    ny = edge.ny
    for ipt in edge.ipts:
        i1 = ipt.n1.index
        i2 = ipt.n2.index
        i3 = ipt.n3.index
        f1 = ipt.f1
        f2 = ipt.f2
        f3 = ipt.f3

        # either use properties from tria or nodes
        if prop_from_node:
            A = f1*ipt.n1.prop.A + f2*ipt.n2.prop.A + f3*ipt.n3.prop.A
            B = f1*ipt.n1.prop.B + f2*ipt.n2.prop.B + f3*ipt.n3.prop.B
            D = f1*ipt.n1.prop.D + f2*ipt.n2.prop.D + f3*ipt.n3.prop.D
            E = f1*ipt.n1.prop.E + f2*ipt.n2.prop.E + f3*ipt.n3.prop.E
        else:
            A = ipt.tria.prop.A
            B = ipt.tria.prop.B
            D = ipt.tria.prop.D
            E = ipt.tria.prop.E
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

        K0[i1*dof+0, i1*dof+0] += f1*nx*(A11*f1*nx + A16*f1*ny) + f1*ny*(A16*f1*nx + A66*f1*ny)
        K0[i1*dof+0, i1*dof+1] += f1*nx*(A16*f1*nx + A66*f1*ny) + f1*ny*(A12*f1*nx + A26*f1*ny)
        K0[i1*dof+0, i1*dof+3] += f1*nx*(B11*f1*nx + B16*f1*ny) + f1*ny*(B16*f1*nx + B66*f1*ny)
        K0[i1*dof+0, i1*dof+4] += f1*nx*(B16*f1*nx + B66*f1*ny) + f1*ny*(B12*f1*nx + B26*f1*ny)
        K0[i1*dof+0, i2*dof+0] += f2*nx*(A11*f1*nx + A16*f1*ny) + f2*ny*(A16*f1*nx + A66*f1*ny)
        K0[i1*dof+0, i2*dof+1] += f2*nx*(A16*f1*nx + A66*f1*ny) + f2*ny*(A12*f1*nx + A26*f1*ny)
        K0[i1*dof+0, i2*dof+3] += f2*nx*(B11*f1*nx + B16*f1*ny) + f2*ny*(B16*f1*nx + B66*f1*ny)
        K0[i1*dof+0, i2*dof+4] += f2*nx*(B16*f1*nx + B66*f1*ny) + f2*ny*(B12*f1*nx + B26*f1*ny)
        K0[i1*dof+0, i3*dof+0] += f3*nx*(A11*f1*nx + A16*f1*ny) + f3*ny*(A16*f1*nx + A66*f1*ny)
        K0[i1*dof+0, i3*dof+1] += f3*nx*(A16*f1*nx + A66*f1*ny) + f3*ny*(A12*f1*nx + A26*f1*ny)
        K0[i1*dof+0, i3*dof+3] += f3*nx*(B11*f1*nx + B16*f1*ny) + f3*ny*(B16*f1*nx + B66*f1*ny)
        K0[i1*dof+0, i3*dof+4] += f3*nx*(B16*f1*nx + B66*f1*ny) + f3*ny*(B12*f1*nx + B26*f1*ny)
        K0[i1*dof+1, i1*dof+0] += f1*nx*(A12*f1*ny + A16*f1*nx) + f1*ny*(A26*f1*ny + A66*f1*nx)
        K0[i1*dof+1, i1*dof+1] += f1*nx*(A26*f1*ny + A66*f1*nx) + f1*ny*(A22*f1*ny + A26*f1*nx)
        K0[i1*dof+1, i1*dof+3] += f1*nx*(B12*f1*ny + B16*f1*nx) + f1*ny*(B26*f1*ny + B66*f1*nx)
        K0[i1*dof+1, i1*dof+4] += f1*nx*(B26*f1*ny + B66*f1*nx) + f1*ny*(B22*f1*ny + B26*f1*nx)
        K0[i1*dof+1, i2*dof+0] += f2*nx*(A12*f1*ny + A16*f1*nx) + f2*ny*(A26*f1*ny + A66*f1*nx)
        K0[i1*dof+1, i2*dof+1] += f2*nx*(A26*f1*ny + A66*f1*nx) + f2*ny*(A22*f1*ny + A26*f1*nx)
        K0[i1*dof+1, i2*dof+3] += f2*nx*(B12*f1*ny + B16*f1*nx) + f2*ny*(B26*f1*ny + B66*f1*nx)
        K0[i1*dof+1, i2*dof+4] += f2*nx*(B26*f1*ny + B66*f1*nx) + f2*ny*(B22*f1*ny + B26*f1*nx)
        K0[i1*dof+1, i3*dof+0] += f3*nx*(A12*f1*ny + A16*f1*nx) + f3*ny*(A26*f1*ny + A66*f1*nx)
        K0[i1*dof+1, i3*dof+1] += f3*nx*(A26*f1*ny + A66*f1*nx) + f3*ny*(A22*f1*ny + A26*f1*nx)
        K0[i1*dof+1, i3*dof+3] += f3*nx*(B12*f1*ny + B16*f1*nx) + f3*ny*(B26*f1*ny + B66*f1*nx)
        K0[i1*dof+1, i3*dof+4] += f3*nx*(B26*f1*ny + B66*f1*nx) + f3*ny*(B22*f1*ny + B26*f1*nx)
        K0[i1*dof+3, i1*dof+0] += f1*nx*(B11*f1*nx + B16*f1*ny) + f1*ny*(B16*f1*nx + B66*f1*ny)
        K0[i1*dof+3, i1*dof+1] += f1*nx*(B16*f1*nx + B66*f1*ny) + f1*ny*(B12*f1*nx + B26*f1*ny)
        K0[i1*dof+3, i1*dof+3] += f1*nx*(D11*f1*nx + D16*f1*ny) + f1*ny*(D16*f1*nx + D66*f1*ny)
        K0[i1*dof+3, i1*dof+4] += f1*nx*(D16*f1*nx + D66*f1*ny) + f1*ny*(D12*f1*nx + D26*f1*ny)
        K0[i1*dof+3, i2*dof+0] += f2*nx*(B11*f1*nx + B16*f1*ny) + f2*ny*(B16*f1*nx + B66*f1*ny)
        K0[i1*dof+3, i2*dof+1] += f2*nx*(B16*f1*nx + B66*f1*ny) + f2*ny*(B12*f1*nx + B26*f1*ny)
        K0[i1*dof+3, i2*dof+3] += f2*nx*(D11*f1*nx + D16*f1*ny) + f2*ny*(D16*f1*nx + D66*f1*ny)
        K0[i1*dof+3, i2*dof+4] += f2*nx*(D16*f1*nx + D66*f1*ny) + f2*ny*(D12*f1*nx + D26*f1*ny)
        K0[i1*dof+3, i3*dof+0] += f3*nx*(B11*f1*nx + B16*f1*ny) + f3*ny*(B16*f1*nx + B66*f1*ny)
        K0[i1*dof+3, i3*dof+1] += f3*nx*(B16*f1*nx + B66*f1*ny) + f3*ny*(B12*f1*nx + B26*f1*ny)
        K0[i1*dof+3, i3*dof+3] += f3*nx*(D11*f1*nx + D16*f1*ny) + f3*ny*(D16*f1*nx + D66*f1*ny)
        K0[i1*dof+3, i3*dof+4] += f3*nx*(D16*f1*nx + D66*f1*ny) + f3*ny*(D12*f1*nx + D26*f1*ny)
        K0[i1*dof+4, i1*dof+0] += f1*nx*(B12*f1*ny + B16*f1*nx) + f1*ny*(B26*f1*ny + B66*f1*nx)
        K0[i1*dof+4, i1*dof+1] += f1*nx*(B26*f1*ny + B66*f1*nx) + f1*ny*(B22*f1*ny + B26*f1*nx)
        K0[i1*dof+4, i1*dof+3] += f1*nx*(D12*f1*ny + D16*f1*nx) + f1*ny*(D26*f1*ny + D66*f1*nx)
        K0[i1*dof+4, i1*dof+4] += f1*nx*(D26*f1*ny + D66*f1*nx) + f1*ny*(D22*f1*ny + D26*f1*nx)
        K0[i1*dof+4, i2*dof+0] += f2*nx*(B12*f1*ny + B16*f1*nx) + f2*ny*(B26*f1*ny + B66*f1*nx)
        K0[i1*dof+4, i2*dof+1] += f2*nx*(B26*f1*ny + B66*f1*nx) + f2*ny*(B22*f1*ny + B26*f1*nx)
        K0[i1*dof+4, i2*dof+3] += f2*nx*(D12*f1*ny + D16*f1*nx) + f2*ny*(D26*f1*ny + D66*f1*nx)
        K0[i1*dof+4, i2*dof+4] += f2*nx*(D26*f1*ny + D66*f1*nx) + f2*ny*(D22*f1*ny + D26*f1*nx)
        K0[i1*dof+4, i3*dof+0] += f3*nx*(B12*f1*ny + B16*f1*nx) + f3*ny*(B26*f1*ny + B66*f1*nx)
        K0[i1*dof+4, i3*dof+1] += f3*nx*(B26*f1*ny + B66*f1*nx) + f3*ny*(B22*f1*ny + B26*f1*nx)
        K0[i1*dof+4, i3*dof+3] += f3*nx*(D12*f1*ny + D16*f1*nx) + f3*ny*(D26*f1*ny + D66*f1*nx)
        K0[i1*dof+4, i3*dof+4] += f3*nx*(D26*f1*ny + D66*f1*nx) + f3*ny*(D22*f1*ny + D26*f1*nx)
        K0[i2*dof+0, i1*dof+0] += f1*nx*(A11*f2*nx + A16*f2*ny) + f1*ny*(A16*f2*nx + A66*f2*ny)
        K0[i2*dof+0, i1*dof+1] += f1*nx*(A16*f2*nx + A66*f2*ny) + f1*ny*(A12*f2*nx + A26*f2*ny)
        K0[i2*dof+0, i1*dof+3] += f1*nx*(B11*f2*nx + B16*f2*ny) + f1*ny*(B16*f2*nx + B66*f2*ny)
        K0[i2*dof+0, i1*dof+4] += f1*nx*(B16*f2*nx + B66*f2*ny) + f1*ny*(B12*f2*nx + B26*f2*ny)
        K0[i2*dof+0, i2*dof+0] += f2*nx*(A11*f2*nx + A16*f2*ny) + f2*ny*(A16*f2*nx + A66*f2*ny)
        K0[i2*dof+0, i2*dof+1] += f2*nx*(A16*f2*nx + A66*f2*ny) + f2*ny*(A12*f2*nx + A26*f2*ny)
        K0[i2*dof+0, i2*dof+3] += f2*nx*(B11*f2*nx + B16*f2*ny) + f2*ny*(B16*f2*nx + B66*f2*ny)
        K0[i2*dof+0, i2*dof+4] += f2*nx*(B16*f2*nx + B66*f2*ny) + f2*ny*(B12*f2*nx + B26*f2*ny)
        K0[i2*dof+0, i3*dof+0] += f3*nx*(A11*f2*nx + A16*f2*ny) + f3*ny*(A16*f2*nx + A66*f2*ny)
        K0[i2*dof+0, i3*dof+1] += f3*nx*(A16*f2*nx + A66*f2*ny) + f3*ny*(A12*f2*nx + A26*f2*ny)
        K0[i2*dof+0, i3*dof+3] += f3*nx*(B11*f2*nx + B16*f2*ny) + f3*ny*(B16*f2*nx + B66*f2*ny)
        K0[i2*dof+0, i3*dof+4] += f3*nx*(B16*f2*nx + B66*f2*ny) + f3*ny*(B12*f2*nx + B26*f2*ny)
        K0[i2*dof+1, i1*dof+0] += f1*nx*(A12*f2*ny + A16*f2*nx) + f1*ny*(A26*f2*ny + A66*f2*nx)
        K0[i2*dof+1, i1*dof+1] += f1*nx*(A26*f2*ny + A66*f2*nx) + f1*ny*(A22*f2*ny + A26*f2*nx)
        K0[i2*dof+1, i1*dof+3] += f1*nx*(B12*f2*ny + B16*f2*nx) + f1*ny*(B26*f2*ny + B66*f2*nx)
        K0[i2*dof+1, i1*dof+4] += f1*nx*(B26*f2*ny + B66*f2*nx) + f1*ny*(B22*f2*ny + B26*f2*nx)
        K0[i2*dof+1, i2*dof+0] += f2*nx*(A12*f2*ny + A16*f2*nx) + f2*ny*(A26*f2*ny + A66*f2*nx)
        K0[i2*dof+1, i2*dof+1] += f2*nx*(A26*f2*ny + A66*f2*nx) + f2*ny*(A22*f2*ny + A26*f2*nx)
        K0[i2*dof+1, i2*dof+3] += f2*nx*(B12*f2*ny + B16*f2*nx) + f2*ny*(B26*f2*ny + B66*f2*nx)
        K0[i2*dof+1, i2*dof+4] += f2*nx*(B26*f2*ny + B66*f2*nx) + f2*ny*(B22*f2*ny + B26*f2*nx)
        K0[i2*dof+1, i3*dof+0] += f3*nx*(A12*f2*ny + A16*f2*nx) + f3*ny*(A26*f2*ny + A66*f2*nx)
        K0[i2*dof+1, i3*dof+1] += f3*nx*(A26*f2*ny + A66*f2*nx) + f3*ny*(A22*f2*ny + A26*f2*nx)
        K0[i2*dof+1, i3*dof+3] += f3*nx*(B12*f2*ny + B16*f2*nx) + f3*ny*(B26*f2*ny + B66*f2*nx)
        K0[i2*dof+1, i3*dof+4] += f3*nx*(B26*f2*ny + B66*f2*nx) + f3*ny*(B22*f2*ny + B26*f2*nx)
        K0[i2*dof+3, i1*dof+0] += f1*nx*(B11*f2*nx + B16*f2*ny) + f1*ny*(B16*f2*nx + B66*f2*ny)
        K0[i2*dof+3, i1*dof+1] += f1*nx*(B16*f2*nx + B66*f2*ny) + f1*ny*(B12*f2*nx + B26*f2*ny)
        K0[i2*dof+3, i1*dof+3] += f1*nx*(D11*f2*nx + D16*f2*ny) + f1*ny*(D16*f2*nx + D66*f2*ny)
        K0[i2*dof+3, i1*dof+4] += f1*nx*(D16*f2*nx + D66*f2*ny) + f1*ny*(D12*f2*nx + D26*f2*ny)
        K0[i2*dof+3, i2*dof+0] += f2*nx*(B11*f2*nx + B16*f2*ny) + f2*ny*(B16*f2*nx + B66*f2*ny)
        K0[i2*dof+3, i2*dof+1] += f2*nx*(B16*f2*nx + B66*f2*ny) + f2*ny*(B12*f2*nx + B26*f2*ny)
        K0[i2*dof+3, i2*dof+3] += f2*nx*(D11*f2*nx + D16*f2*ny) + f2*ny*(D16*f2*nx + D66*f2*ny)
        K0[i2*dof+3, i2*dof+4] += f2*nx*(D16*f2*nx + D66*f2*ny) + f2*ny*(D12*f2*nx + D26*f2*ny)
        K0[i2*dof+3, i3*dof+0] += f3*nx*(B11*f2*nx + B16*f2*ny) + f3*ny*(B16*f2*nx + B66*f2*ny)
        K0[i2*dof+3, i3*dof+1] += f3*nx*(B16*f2*nx + B66*f2*ny) + f3*ny*(B12*f2*nx + B26*f2*ny)
        K0[i2*dof+3, i3*dof+3] += f3*nx*(D11*f2*nx + D16*f2*ny) + f3*ny*(D16*f2*nx + D66*f2*ny)
        K0[i2*dof+3, i3*dof+4] += f3*nx*(D16*f2*nx + D66*f2*ny) + f3*ny*(D12*f2*nx + D26*f2*ny)
        K0[i2*dof+4, i1*dof+0] += f1*nx*(B12*f2*ny + B16*f2*nx) + f1*ny*(B26*f2*ny + B66*f2*nx)
        K0[i2*dof+4, i1*dof+1] += f1*nx*(B26*f2*ny + B66*f2*nx) + f1*ny*(B22*f2*ny + B26*f2*nx)
        K0[i2*dof+4, i1*dof+3] += f1*nx*(D12*f2*ny + D16*f2*nx) + f1*ny*(D26*f2*ny + D66*f2*nx)
        K0[i2*dof+4, i1*dof+4] += f1*nx*(D26*f2*ny + D66*f2*nx) + f1*ny*(D22*f2*ny + D26*f2*nx)
        K0[i2*dof+4, i2*dof+0] += f2*nx*(B12*f2*ny + B16*f2*nx) + f2*ny*(B26*f2*ny + B66*f2*nx)
        K0[i2*dof+4, i2*dof+1] += f2*nx*(B26*f2*ny + B66*f2*nx) + f2*ny*(B22*f2*ny + B26*f2*nx)
        K0[i2*dof+4, i2*dof+3] += f2*nx*(D12*f2*ny + D16*f2*nx) + f2*ny*(D26*f2*ny + D66*f2*nx)
        K0[i2*dof+4, i2*dof+4] += f2*nx*(D26*f2*ny + D66*f2*nx) + f2*ny*(D22*f2*ny + D26*f2*nx)
        K0[i2*dof+4, i3*dof+0] += f3*nx*(B12*f2*ny + B16*f2*nx) + f3*ny*(B26*f2*ny + B66*f2*nx)
        K0[i2*dof+4, i3*dof+1] += f3*nx*(B26*f2*ny + B66*f2*nx) + f3*ny*(B22*f2*ny + B26*f2*nx)
        K0[i2*dof+4, i3*dof+3] += f3*nx*(D12*f2*ny + D16*f2*nx) + f3*ny*(D26*f2*ny + D66*f2*nx)
        K0[i2*dof+4, i3*dof+4] += f3*nx*(D26*f2*ny + D66*f2*nx) + f3*ny*(D22*f2*ny + D26*f2*nx)
        K0[i3*dof+0, i1*dof+0] += f1*nx*(A11*f3*nx + A16*f3*ny) + f1*ny*(A16*f3*nx + A66*f3*ny)
        K0[i3*dof+0, i1*dof+1] += f1*nx*(A16*f3*nx + A66*f3*ny) + f1*ny*(A12*f3*nx + A26*f3*ny)
        K0[i3*dof+0, i1*dof+3] += f1*nx*(B11*f3*nx + B16*f3*ny) + f1*ny*(B16*f3*nx + B66*f3*ny)
        K0[i3*dof+0, i1*dof+4] += f1*nx*(B16*f3*nx + B66*f3*ny) + f1*ny*(B12*f3*nx + B26*f3*ny)
        K0[i3*dof+0, i2*dof+0] += f2*nx*(A11*f3*nx + A16*f3*ny) + f2*ny*(A16*f3*nx + A66*f3*ny)
        K0[i3*dof+0, i2*dof+1] += f2*nx*(A16*f3*nx + A66*f3*ny) + f2*ny*(A12*f3*nx + A26*f3*ny)
        K0[i3*dof+0, i2*dof+3] += f2*nx*(B11*f3*nx + B16*f3*ny) + f2*ny*(B16*f3*nx + B66*f3*ny)
        K0[i3*dof+0, i2*dof+4] += f2*nx*(B16*f3*nx + B66*f3*ny) + f2*ny*(B12*f3*nx + B26*f3*ny)
        K0[i3*dof+0, i3*dof+0] += f3*nx*(A11*f3*nx + A16*f3*ny) + f3*ny*(A16*f3*nx + A66*f3*ny)
        K0[i3*dof+0, i3*dof+1] += f3*nx*(A16*f3*nx + A66*f3*ny) + f3*ny*(A12*f3*nx + A26*f3*ny)
        K0[i3*dof+0, i3*dof+3] += f3*nx*(B11*f3*nx + B16*f3*ny) + f3*ny*(B16*f3*nx + B66*f3*ny)
        K0[i3*dof+0, i3*dof+4] += f3*nx*(B16*f3*nx + B66*f3*ny) + f3*ny*(B12*f3*nx + B26*f3*ny)
        K0[i3*dof+1, i1*dof+0] += f1*nx*(A12*f3*ny + A16*f3*nx) + f1*ny*(A26*f3*ny + A66*f3*nx)
        K0[i3*dof+1, i1*dof+1] += f1*nx*(A26*f3*ny + A66*f3*nx) + f1*ny*(A22*f3*ny + A26*f3*nx)
        K0[i3*dof+1, i1*dof+3] += f1*nx*(B12*f3*ny + B16*f3*nx) + f1*ny*(B26*f3*ny + B66*f3*nx)
        K0[i3*dof+1, i1*dof+4] += f1*nx*(B26*f3*ny + B66*f3*nx) + f1*ny*(B22*f3*ny + B26*f3*nx)
        K0[i3*dof+1, i2*dof+0] += f2*nx*(A12*f3*ny + A16*f3*nx) + f2*ny*(A26*f3*ny + A66*f3*nx)
        K0[i3*dof+1, i2*dof+1] += f2*nx*(A26*f3*ny + A66*f3*nx) + f2*ny*(A22*f3*ny + A26*f3*nx)
        K0[i3*dof+1, i2*dof+3] += f2*nx*(B12*f3*ny + B16*f3*nx) + f2*ny*(B26*f3*ny + B66*f3*nx)
        K0[i3*dof+1, i2*dof+4] += f2*nx*(B26*f3*ny + B66*f3*nx) + f2*ny*(B22*f3*ny + B26*f3*nx)
        K0[i3*dof+1, i3*dof+0] += f3*nx*(A12*f3*ny + A16*f3*nx) + f3*ny*(A26*f3*ny + A66*f3*nx)
        K0[i3*dof+1, i3*dof+1] += f3*nx*(A26*f3*ny + A66*f3*nx) + f3*ny*(A22*f3*ny + A26*f3*nx)
        K0[i3*dof+1, i3*dof+3] += f3*nx*(B12*f3*ny + B16*f3*nx) + f3*ny*(B26*f3*ny + B66*f3*nx)
        K0[i3*dof+1, i3*dof+4] += f3*nx*(B26*f3*ny + B66*f3*nx) + f3*ny*(B22*f3*ny + B26*f3*nx)
        K0[i3*dof+3, i1*dof+0] += f1*nx*(B11*f3*nx + B16*f3*ny) + f1*ny*(B16*f3*nx + B66*f3*ny)
        K0[i3*dof+3, i1*dof+1] += f1*nx*(B16*f3*nx + B66*f3*ny) + f1*ny*(B12*f3*nx + B26*f3*ny)
        K0[i3*dof+3, i1*dof+3] += f1*nx*(D11*f3*nx + D16*f3*ny) + f1*ny*(D16*f3*nx + D66*f3*ny)
        K0[i3*dof+3, i1*dof+4] += f1*nx*(D16*f3*nx + D66*f3*ny) + f1*ny*(D12*f3*nx + D26*f3*ny)
        K0[i3*dof+3, i2*dof+0] += f2*nx*(B11*f3*nx + B16*f3*ny) + f2*ny*(B16*f3*nx + B66*f3*ny)
        K0[i3*dof+3, i2*dof+1] += f2*nx*(B16*f3*nx + B66*f3*ny) + f2*ny*(B12*f3*nx + B26*f3*ny)
        K0[i3*dof+3, i2*dof+3] += f2*nx*(D11*f3*nx + D16*f3*ny) + f2*ny*(D16*f3*nx + D66*f3*ny)
        K0[i3*dof+3, i2*dof+4] += f2*nx*(D16*f3*nx + D66*f3*ny) + f2*ny*(D12*f3*nx + D26*f3*ny)
        K0[i3*dof+3, i3*dof+0] += f3*nx*(B11*f3*nx + B16*f3*ny) + f3*ny*(B16*f3*nx + B66*f3*ny)
        K0[i3*dof+3, i3*dof+1] += f3*nx*(B16*f3*nx + B66*f3*ny) + f3*ny*(B12*f3*nx + B26*f3*ny)
        K0[i3*dof+3, i3*dof+3] += f3*nx*(D11*f3*nx + D16*f3*ny) + f3*ny*(D16*f3*nx + D66*f3*ny)
        K0[i3*dof+3, i3*dof+4] += f3*nx*(D16*f3*nx + D66*f3*ny) + f3*ny*(D12*f3*nx + D26*f3*ny)
        K0[i3*dof+4, i1*dof+0] += f1*nx*(B12*f3*ny + B16*f3*nx) + f1*ny*(B26*f3*ny + B66*f3*nx)
        K0[i3*dof+4, i1*dof+1] += f1*nx*(B26*f3*ny + B66*f3*nx) + f1*ny*(B22*f3*ny + B26*f3*nx)
        K0[i3*dof+4, i1*dof+3] += f1*nx*(D12*f3*ny + D16*f3*nx) + f1*ny*(D26*f3*ny + D66*f3*nx)
        K0[i3*dof+4, i1*dof+4] += f1*nx*(D26*f3*ny + D66*f3*nx) + f1*ny*(D22*f3*ny + D26*f3*nx)
        K0[i3*dof+4, i2*dof+0] += f2*nx*(B12*f3*ny + B16*f3*nx) + f2*ny*(B26*f3*ny + B66*f3*nx)
        K0[i3*dof+4, i2*dof+1] += f2*nx*(B26*f3*ny + B66*f3*nx) + f2*ny*(B22*f3*ny + B26*f3*nx)
        K0[i3*dof+4, i2*dof+3] += f2*nx*(D12*f3*ny + D16*f3*nx) + f2*ny*(D26*f3*ny + D66*f3*nx)
        K0[i3*dof+4, i2*dof+4] += f2*nx*(D26*f3*ny + D66*f3*nx) + f2*ny*(D22*f3*ny + D26*f3*nx)
        K0[i3*dof+4, i3*dof+0] += f3*nx*(B12*f3*ny + B16*f3*nx) + f3*ny*(B26*f3*ny + B66*f3*nx)
        K0[i3*dof+4, i3*dof+1] += f3*nx*(B26*f3*ny + B66*f3*nx) + f3*ny*(B22*f3*ny + B26*f3*nx)
        K0[i3*dof+4, i3*dof+3] += f3*nx*(D12*f3*ny + D16*f3*nx) + f3*ny*(D26*f3*ny + D66*f3*nx)
        K0[i3*dof+4, i3*dof+4] += f3*nx*(D26*f3*ny + D66*f3*nx) + f3*ny*(D22*f3*ny + D26*f3*nx)

K0[2*dof+0, :] = 0
K0[:, 2*dof+0] = 0
K0[2*dof+1, :] = 0
K0[:, 2*dof+1] = 0


K0 = coo_matrix(K0)

u = solve(K0, F)
print(u)

for nindi, node in ind2node.items():
    for i in range(3):
        node.pos[i] += u[nindi*dof+i]

xcord = [node.pos[0] for node in nodes]
ycord = [node.pos[1] for node in nodes]
plt.scatter(xcord, ycord)
plt.show()


#compute Kcell = sum(Kedge)  :  strain smoothing


# integration edge: 2 nodes (only cells at boundary)
# integration

# Lagrangian interpolation
# - at any cell, the evaluation of the integrand will be a function of three
#   nodes
# - points at smoothing domain edges a function of 3 nodes
# - points at edge a function of 4 nodes
# - an internal cell will have 13 integration points at least, one for each
# edge

# - there should be a way to quickly find the nodes belonging to any
# integration point

# example of integrand
# (du/dx)^T * E (du/dx)
# for a given integration point find the interpolation of u

# u = u1 * c1 + u2 * c2 + u3 * c3
# ci is a weighting factor that is a function of the distance between
# integration points and nodes of influence

#


# constitutive stiffness matrix FSDT

# u = u1*f1 + u2*f2 + u3*f3
# v = v1*f1 + v2*f2 + v3*f3
# w = w1*f1 + w2*f2 + w3*f3
# phix = phix1*f1 + phix2*f2 + phix3*f3
# phiy = phiy1*f1 + phiy2*f2 + phiy3*f3



# u = unodal * area_ratios(integration point position, nodal coordinates)



# generated matrices are expected to be sparse since there will be a maximum of
# six integration points connecting two degrees of freedom

# plate equation
# exx = ux
# eyy = vy
# gxy = uy + vx
# kxx = phix,x
# kyy = phiy,y
# kxy = phix,y + phiy,x







