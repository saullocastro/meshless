import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

from meshfree.composite.laminate import read_stack
from meshfree.sparse import solve, is_symmetric

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
    def __init__(self, pos, tria, n1, n2, n3, f1, f2, f3, nx, ny, nz=0):
        self.pos = pos
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
        self.As = None

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
        Node(1.5, 0, 0),
        Node(3, 0, 0),

        Node(0, 1.5, 0),
        Node(1.5, 1.5, 0),
        Node(3, 1.5, 0),

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
             [edge.n1.pos[1], edge.n2.pos[1]], '--k', lw=0.5, mfc=None)
    for tria in trias:
        if len(set(edge.nodes) & set(tria.nodes)) == 2:
            tria.edges.append(edge)
            edge.trias.append(tria)
            for node in tria.nodes:
                node.trias.add(tria)

# the code above will come from an external triangulation algorithm

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
    ipts.append(IntegrationPoint(0.5*(node1.pos + mid1.pos), tria1,
            node1, node2, othernode1, 2/3, 1/6, 1/6, nx, ny, nz))

    #NOTE check only if for distorted meshes these ratios 2/3, 1/6, 1/6 are
    #     still valid
    A1 = area_of_polygon([node1.pos[0], node2.pos[0], othernode1.pos[0]],
                         [node1.pos[1], node2.pos[1], othernode1.pos[1]])
    ipt = ipts[-1]
    fA1 = area_of_polygon([ipt.pos[0], node2.pos[0], othernode1.pos[0]],
                          [ipt.pos[1], node2.pos[1], othernode1.pos[1]])
    print('CHECK area: %1.3f = %1.3f' % (fA1/A1, 2/3))


    tmpvec = (mid1 - node2).pos
    nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
    sdomain.append(node2)
    ipts.append(IntegrationPoint(0.5*(mid1.pos + node2.pos), tria1,
            node1, node2, othernode1, 1/6, 2/3, 1/6, nx, ny, nz))

    if tria2 is None:
        tmpvec = (node2 - node1).pos
        nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
        sdomain.append(node1)
        ipts.append(IntegrationPoint(0.5*(node2.pos + node1.pos), tria1,
            node1, node2, othernode1, 1/2, 1/2, 0, nx, ny, nz))
    else:
        tmpvec = (node2 - mid2).pos
        nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
        sdomain.append(mid2)
        ipts.append(IntegrationPoint(0.5*(node2.pos + mid2.pos), tria2,
            node1, node2, othernode2, 1/6, 2/3, 1/6, nx, ny, nz))
        tmpvec = (mid2 - node1).pos
        nx, ny, nz = unitvec(np.cross(tmpvec, sign*ZGLOBAL))
        sdomain.append(node1)
        ipts.append(IntegrationPoint(0.5*(mid2.pos + node1.pos), tria2,
            node1, node2, othernode2, 2/3, 1/6, 1/6, nx, ny, nz))

    edge.sdomain = sdomain
    edge.As = area_of_polygon([sr.pos[0] for sr in sdomain[:-1]],
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
lam = read_stack([0], plyt=0.001, laminaprop=(71.e9, 71.e9, 0.33))
prop = Property(lam.A, lam.B, lam.D, lam.E)
for tria in trias:
    tria.prop = prop

#TODO allocate less memory here...
k0 = np.zeros((n*dof, n*dof), dtype=np.float64)

prop_from_node = False
for edge in edges:
    As = edge.As
    for ipt in edge.ipts:
        i1 = ipt.n1.index
        i2 = ipt.n2.index
        i3 = ipt.n3.index
        f1 = ipt.f1
        f2 = ipt.f2
        f3 = ipt.f3
        nx = ipt.nx
        ny = ipt.ny

        # either use properties from tria or nodes
        if prop_from_node:
            A = f1*ipt.n1.prop.A + f2*ipt.n2.prop.A + f3*ipt.n3.prop.A
            B = f1*ipt.n1.prop.B + f2*ipt.n2.prop.B + f3*ipt.n3.prop.B
            D = f1*ipt.n1.prop.D + f2*ipt.n2.prop.D + f3*ipt.n3.prop.D
            #E = f1*ipt.n1.prop.E + f2*ipt.n2.prop.E + f3*ipt.n3.prop.E
        else:
            A = ipt.tria.prop.A
            B = ipt.tria.prop.B
            D = ipt.tria.prop.D
            #E = ipt.tria.prop.E
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
        k0[i1*dof+0, i1*dof+0] += (f1*nx*(A11*f1*nx + A16*f1*ny) + f1*ny*(A16*f1*nx + A66*f1*ny))/As
        k0[i1*dof+0, i1*dof+1] += (f1*nx*(A16*f1*nx + A66*f1*ny) + f1*ny*(A12*f1*nx + A26*f1*ny))/As
        k0[i1*dof+0, i1*dof+3] += (f1*nx*(B11*f1*nx + B16*f1*ny) + f1*ny*(B16*f1*nx + B66*f1*ny))/As
        k0[i1*dof+0, i1*dof+4] += (f1*nx*(B16*f1*nx + B66*f1*ny) + f1*ny*(B12*f1*nx + B26*f1*ny))/As
        k0[i1*dof+0, i2*dof+0] += (f2*nx*(A11*f1*nx + A16*f1*ny) + f2*ny*(A16*f1*nx + A66*f1*ny))/As
        k0[i1*dof+0, i2*dof+1] += (f2*nx*(A16*f1*nx + A66*f1*ny) + f2*ny*(A12*f1*nx + A26*f1*ny))/As
        k0[i1*dof+0, i2*dof+3] += (f2*nx*(B11*f1*nx + B16*f1*ny) + f2*ny*(B16*f1*nx + B66*f1*ny))/As
        k0[i1*dof+0, i2*dof+4] += (f2*nx*(B16*f1*nx + B66*f1*ny) + f2*ny*(B12*f1*nx + B26*f1*ny))/As
        k0[i1*dof+0, i3*dof+0] += (f3*nx*(A11*f1*nx + A16*f1*ny) + f3*ny*(A16*f1*nx + A66*f1*ny))/As
        k0[i1*dof+0, i3*dof+1] += (f3*nx*(A16*f1*nx + A66*f1*ny) + f3*ny*(A12*f1*nx + A26*f1*ny))/As
        k0[i1*dof+0, i3*dof+3] += (f3*nx*(B11*f1*nx + B16*f1*ny) + f3*ny*(B16*f1*nx + B66*f1*ny))/As
        k0[i1*dof+0, i3*dof+4] += (f3*nx*(B16*f1*nx + B66*f1*ny) + f3*ny*(B12*f1*nx + B26*f1*ny))/As
        k0[i1*dof+1, i1*dof+0] += (f1*nx*(A12*f1*ny + A16*f1*nx) + f1*ny*(A26*f1*ny + A66*f1*nx))/As
        k0[i1*dof+1, i1*dof+1] += (f1*nx*(A26*f1*ny + A66*f1*nx) + f1*ny*(A22*f1*ny + A26*f1*nx))/As
        k0[i1*dof+1, i1*dof+3] += (f1*nx*(B12*f1*ny + B16*f1*nx) + f1*ny*(B26*f1*ny + B66*f1*nx))/As
        k0[i1*dof+1, i1*dof+4] += (f1*nx*(B26*f1*ny + B66*f1*nx) + f1*ny*(B22*f1*ny + B26*f1*nx))/As
        k0[i1*dof+1, i2*dof+0] += (f2*nx*(A12*f1*ny + A16*f1*nx) + f2*ny*(A26*f1*ny + A66*f1*nx))/As
        k0[i1*dof+1, i2*dof+1] += (f2*nx*(A26*f1*ny + A66*f1*nx) + f2*ny*(A22*f1*ny + A26*f1*nx))/As
        k0[i1*dof+1, i2*dof+3] += (f2*nx*(B12*f1*ny + B16*f1*nx) + f2*ny*(B26*f1*ny + B66*f1*nx))/As
        k0[i1*dof+1, i2*dof+4] += (f2*nx*(B26*f1*ny + B66*f1*nx) + f2*ny*(B22*f1*ny + B26*f1*nx))/As
        k0[i1*dof+1, i3*dof+0] += (f3*nx*(A12*f1*ny + A16*f1*nx) + f3*ny*(A26*f1*ny + A66*f1*nx))/As
        k0[i1*dof+1, i3*dof+1] += (f3*nx*(A26*f1*ny + A66*f1*nx) + f3*ny*(A22*f1*ny + A26*f1*nx))/As
        k0[i1*dof+1, i3*dof+3] += (f3*nx*(B12*f1*ny + B16*f1*nx) + f3*ny*(B26*f1*ny + B66*f1*nx))/As
        k0[i1*dof+1, i3*dof+4] += (f3*nx*(B26*f1*ny + B66*f1*nx) + f3*ny*(B22*f1*ny + B26*f1*nx))/As
        k0[i1*dof+3, i1*dof+0] += (f1*nx*(B11*f1*nx + B16*f1*ny) + f1*ny*(B16*f1*nx + B66*f1*ny))/As
        k0[i1*dof+3, i1*dof+1] += (f1*nx*(B16*f1*nx + B66*f1*ny) + f1*ny*(B12*f1*nx + B26*f1*ny))/As
        k0[i1*dof+3, i1*dof+3] += (f1*nx*(D11*f1*nx + D16*f1*ny) + f1*ny*(D16*f1*nx + D66*f1*ny))/As
        k0[i1*dof+3, i1*dof+4] += (f1*nx*(D16*f1*nx + D66*f1*ny) + f1*ny*(D12*f1*nx + D26*f1*ny))/As
        k0[i1*dof+3, i2*dof+0] += (f2*nx*(B11*f1*nx + B16*f1*ny) + f2*ny*(B16*f1*nx + B66*f1*ny))/As
        k0[i1*dof+3, i2*dof+1] += (f2*nx*(B16*f1*nx + B66*f1*ny) + f2*ny*(B12*f1*nx + B26*f1*ny))/As
        k0[i1*dof+3, i2*dof+3] += (f2*nx*(D11*f1*nx + D16*f1*ny) + f2*ny*(D16*f1*nx + D66*f1*ny))/As
        k0[i1*dof+3, i2*dof+4] += (f2*nx*(D16*f1*nx + D66*f1*ny) + f2*ny*(D12*f1*nx + D26*f1*ny))/As
        k0[i1*dof+3, i3*dof+0] += (f3*nx*(B11*f1*nx + B16*f1*ny) + f3*ny*(B16*f1*nx + B66*f1*ny))/As
        k0[i1*dof+3, i3*dof+1] += (f3*nx*(B16*f1*nx + B66*f1*ny) + f3*ny*(B12*f1*nx + B26*f1*ny))/As
        k0[i1*dof+3, i3*dof+3] += (f3*nx*(D11*f1*nx + D16*f1*ny) + f3*ny*(D16*f1*nx + D66*f1*ny))/As
        k0[i1*dof+3, i3*dof+4] += (f3*nx*(D16*f1*nx + D66*f1*ny) + f3*ny*(D12*f1*nx + D26*f1*ny))/As
        k0[i1*dof+4, i1*dof+0] += (f1*nx*(B12*f1*ny + B16*f1*nx) + f1*ny*(B26*f1*ny + B66*f1*nx))/As
        k0[i1*dof+4, i1*dof+1] += (f1*nx*(B26*f1*ny + B66*f1*nx) + f1*ny*(B22*f1*ny + B26*f1*nx))/As
        k0[i1*dof+4, i1*dof+3] += (f1*nx*(D12*f1*ny + D16*f1*nx) + f1*ny*(D26*f1*ny + D66*f1*nx))/As
        k0[i1*dof+4, i1*dof+4] += (f1*nx*(D26*f1*ny + D66*f1*nx) + f1*ny*(D22*f1*ny + D26*f1*nx))/As
        k0[i1*dof+4, i2*dof+0] += (f2*nx*(B12*f1*ny + B16*f1*nx) + f2*ny*(B26*f1*ny + B66*f1*nx))/As
        k0[i1*dof+4, i2*dof+1] += (f2*nx*(B26*f1*ny + B66*f1*nx) + f2*ny*(B22*f1*ny + B26*f1*nx))/As
        k0[i1*dof+4, i2*dof+3] += (f2*nx*(D12*f1*ny + D16*f1*nx) + f2*ny*(D26*f1*ny + D66*f1*nx))/As
        k0[i1*dof+4, i2*dof+4] += (f2*nx*(D26*f1*ny + D66*f1*nx) + f2*ny*(D22*f1*ny + D26*f1*nx))/As
        k0[i1*dof+4, i3*dof+0] += (f3*nx*(B12*f1*ny + B16*f1*nx) + f3*ny*(B26*f1*ny + B66*f1*nx))/As
        k0[i1*dof+4, i3*dof+1] += (f3*nx*(B26*f1*ny + B66*f1*nx) + f3*ny*(B22*f1*ny + B26*f1*nx))/As
        k0[i1*dof+4, i3*dof+3] += (f3*nx*(D12*f1*ny + D16*f1*nx) + f3*ny*(D26*f1*ny + D66*f1*nx))/As
        k0[i1*dof+4, i3*dof+4] += (f3*nx*(D26*f1*ny + D66*f1*nx) + f3*ny*(D22*f1*ny + D26*f1*nx))/As
        k0[i2*dof+0, i1*dof+0] += (f1*nx*(A11*f2*nx + A16*f2*ny) + f1*ny*(A16*f2*nx + A66*f2*ny))/As
        k0[i2*dof+0, i1*dof+1] += (f1*nx*(A16*f2*nx + A66*f2*ny) + f1*ny*(A12*f2*nx + A26*f2*ny))/As
        k0[i2*dof+0, i1*dof+3] += (f1*nx*(B11*f2*nx + B16*f2*ny) + f1*ny*(B16*f2*nx + B66*f2*ny))/As
        k0[i2*dof+0, i1*dof+4] += (f1*nx*(B16*f2*nx + B66*f2*ny) + f1*ny*(B12*f2*nx + B26*f2*ny))/As
        k0[i2*dof+0, i2*dof+0] += (f2*nx*(A11*f2*nx + A16*f2*ny) + f2*ny*(A16*f2*nx + A66*f2*ny))/As
        k0[i2*dof+0, i2*dof+1] += (f2*nx*(A16*f2*nx + A66*f2*ny) + f2*ny*(A12*f2*nx + A26*f2*ny))/As
        k0[i2*dof+0, i2*dof+3] += (f2*nx*(B11*f2*nx + B16*f2*ny) + f2*ny*(B16*f2*nx + B66*f2*ny))/As
        k0[i2*dof+0, i2*dof+4] += (f2*nx*(B16*f2*nx + B66*f2*ny) + f2*ny*(B12*f2*nx + B26*f2*ny))/As
        k0[i2*dof+0, i3*dof+0] += (f3*nx*(A11*f2*nx + A16*f2*ny) + f3*ny*(A16*f2*nx + A66*f2*ny))/As
        k0[i2*dof+0, i3*dof+1] += (f3*nx*(A16*f2*nx + A66*f2*ny) + f3*ny*(A12*f2*nx + A26*f2*ny))/As
        k0[i2*dof+0, i3*dof+3] += (f3*nx*(B11*f2*nx + B16*f2*ny) + f3*ny*(B16*f2*nx + B66*f2*ny))/As
        k0[i2*dof+0, i3*dof+4] += (f3*nx*(B16*f2*nx + B66*f2*ny) + f3*ny*(B12*f2*nx + B26*f2*ny))/As
        k0[i2*dof+1, i1*dof+0] += (f1*nx*(A12*f2*ny + A16*f2*nx) + f1*ny*(A26*f2*ny + A66*f2*nx))/As
        k0[i2*dof+1, i1*dof+1] += (f1*nx*(A26*f2*ny + A66*f2*nx) + f1*ny*(A22*f2*ny + A26*f2*nx))/As
        k0[i2*dof+1, i1*dof+3] += (f1*nx*(B12*f2*ny + B16*f2*nx) + f1*ny*(B26*f2*ny + B66*f2*nx))/As
        k0[i2*dof+1, i1*dof+4] += (f1*nx*(B26*f2*ny + B66*f2*nx) + f1*ny*(B22*f2*ny + B26*f2*nx))/As
        k0[i2*dof+1, i2*dof+0] += (f2*nx*(A12*f2*ny + A16*f2*nx) + f2*ny*(A26*f2*ny + A66*f2*nx))/As
        k0[i2*dof+1, i2*dof+1] += (f2*nx*(A26*f2*ny + A66*f2*nx) + f2*ny*(A22*f2*ny + A26*f2*nx))/As
        k0[i2*dof+1, i2*dof+3] += (f2*nx*(B12*f2*ny + B16*f2*nx) + f2*ny*(B26*f2*ny + B66*f2*nx))/As
        k0[i2*dof+1, i2*dof+4] += (f2*nx*(B26*f2*ny + B66*f2*nx) + f2*ny*(B22*f2*ny + B26*f2*nx))/As
        k0[i2*dof+1, i3*dof+0] += (f3*nx*(A12*f2*ny + A16*f2*nx) + f3*ny*(A26*f2*ny + A66*f2*nx))/As
        k0[i2*dof+1, i3*dof+1] += (f3*nx*(A26*f2*ny + A66*f2*nx) + f3*ny*(A22*f2*ny + A26*f2*nx))/As
        k0[i2*dof+1, i3*dof+3] += (f3*nx*(B12*f2*ny + B16*f2*nx) + f3*ny*(B26*f2*ny + B66*f2*nx))/As
        k0[i2*dof+1, i3*dof+4] += (f3*nx*(B26*f2*ny + B66*f2*nx) + f3*ny*(B22*f2*ny + B26*f2*nx))/As
        k0[i2*dof+3, i1*dof+0] += (f1*nx*(B11*f2*nx + B16*f2*ny) + f1*ny*(B16*f2*nx + B66*f2*ny))/As
        k0[i2*dof+3, i1*dof+1] += (f1*nx*(B16*f2*nx + B66*f2*ny) + f1*ny*(B12*f2*nx + B26*f2*ny))/As
        k0[i2*dof+3, i1*dof+3] += (f1*nx*(D11*f2*nx + D16*f2*ny) + f1*ny*(D16*f2*nx + D66*f2*ny))/As
        k0[i2*dof+3, i1*dof+4] += (f1*nx*(D16*f2*nx + D66*f2*ny) + f1*ny*(D12*f2*nx + D26*f2*ny))/As
        k0[i2*dof+3, i2*dof+0] += (f2*nx*(B11*f2*nx + B16*f2*ny) + f2*ny*(B16*f2*nx + B66*f2*ny))/As
        k0[i2*dof+3, i2*dof+1] += (f2*nx*(B16*f2*nx + B66*f2*ny) + f2*ny*(B12*f2*nx + B26*f2*ny))/As
        k0[i2*dof+3, i2*dof+3] += (f2*nx*(D11*f2*nx + D16*f2*ny) + f2*ny*(D16*f2*nx + D66*f2*ny))/As
        k0[i2*dof+3, i2*dof+4] += (f2*nx*(D16*f2*nx + D66*f2*ny) + f2*ny*(D12*f2*nx + D26*f2*ny))/As
        k0[i2*dof+3, i3*dof+0] += (f3*nx*(B11*f2*nx + B16*f2*ny) + f3*ny*(B16*f2*nx + B66*f2*ny))/As
        k0[i2*dof+3, i3*dof+1] += (f3*nx*(B16*f2*nx + B66*f2*ny) + f3*ny*(B12*f2*nx + B26*f2*ny))/As
        k0[i2*dof+3, i3*dof+3] += (f3*nx*(D11*f2*nx + D16*f2*ny) + f3*ny*(D16*f2*nx + D66*f2*ny))/As
        k0[i2*dof+3, i3*dof+4] += (f3*nx*(D16*f2*nx + D66*f2*ny) + f3*ny*(D12*f2*nx + D26*f2*ny))/As
        k0[i2*dof+4, i1*dof+0] += (f1*nx*(B12*f2*ny + B16*f2*nx) + f1*ny*(B26*f2*ny + B66*f2*nx))/As
        k0[i2*dof+4, i1*dof+1] += (f1*nx*(B26*f2*ny + B66*f2*nx) + f1*ny*(B22*f2*ny + B26*f2*nx))/As
        k0[i2*dof+4, i1*dof+3] += (f1*nx*(D12*f2*ny + D16*f2*nx) + f1*ny*(D26*f2*ny + D66*f2*nx))/As
        k0[i2*dof+4, i1*dof+4] += (f1*nx*(D26*f2*ny + D66*f2*nx) + f1*ny*(D22*f2*ny + D26*f2*nx))/As
        k0[i2*dof+4, i2*dof+0] += (f2*nx*(B12*f2*ny + B16*f2*nx) + f2*ny*(B26*f2*ny + B66*f2*nx))/As
        k0[i2*dof+4, i2*dof+1] += (f2*nx*(B26*f2*ny + B66*f2*nx) + f2*ny*(B22*f2*ny + B26*f2*nx))/As
        k0[i2*dof+4, i2*dof+3] += (f2*nx*(D12*f2*ny + D16*f2*nx) + f2*ny*(D26*f2*ny + D66*f2*nx))/As
        k0[i2*dof+4, i2*dof+4] += (f2*nx*(D26*f2*ny + D66*f2*nx) + f2*ny*(D22*f2*ny + D26*f2*nx))/As
        k0[i2*dof+4, i3*dof+0] += (f3*nx*(B12*f2*ny + B16*f2*nx) + f3*ny*(B26*f2*ny + B66*f2*nx))/As
        k0[i2*dof+4, i3*dof+1] += (f3*nx*(B26*f2*ny + B66*f2*nx) + f3*ny*(B22*f2*ny + B26*f2*nx))/As
        k0[i2*dof+4, i3*dof+3] += (f3*nx*(D12*f2*ny + D16*f2*nx) + f3*ny*(D26*f2*ny + D66*f2*nx))/As
        k0[i2*dof+4, i3*dof+4] += (f3*nx*(D26*f2*ny + D66*f2*nx) + f3*ny*(D22*f2*ny + D26*f2*nx))/As
        k0[i3*dof+0, i1*dof+0] += (f1*nx*(A11*f3*nx + A16*f3*ny) + f1*ny*(A16*f3*nx + A66*f3*ny))/As
        k0[i3*dof+0, i1*dof+1] += (f1*nx*(A16*f3*nx + A66*f3*ny) + f1*ny*(A12*f3*nx + A26*f3*ny))/As
        k0[i3*dof+0, i1*dof+3] += (f1*nx*(B11*f3*nx + B16*f3*ny) + f1*ny*(B16*f3*nx + B66*f3*ny))/As
        k0[i3*dof+0, i1*dof+4] += (f1*nx*(B16*f3*nx + B66*f3*ny) + f1*ny*(B12*f3*nx + B26*f3*ny))/As
        k0[i3*dof+0, i2*dof+0] += (f2*nx*(A11*f3*nx + A16*f3*ny) + f2*ny*(A16*f3*nx + A66*f3*ny))/As
        k0[i3*dof+0, i2*dof+1] += (f2*nx*(A16*f3*nx + A66*f3*ny) + f2*ny*(A12*f3*nx + A26*f3*ny))/As
        k0[i3*dof+0, i2*dof+3] += (f2*nx*(B11*f3*nx + B16*f3*ny) + f2*ny*(B16*f3*nx + B66*f3*ny))/As
        k0[i3*dof+0, i2*dof+4] += (f2*nx*(B16*f3*nx + B66*f3*ny) + f2*ny*(B12*f3*nx + B26*f3*ny))/As
        k0[i3*dof+0, i3*dof+0] += (f3*nx*(A11*f3*nx + A16*f3*ny) + f3*ny*(A16*f3*nx + A66*f3*ny))/As
        k0[i3*dof+0, i3*dof+1] += (f3*nx*(A16*f3*nx + A66*f3*ny) + f3*ny*(A12*f3*nx + A26*f3*ny))/As
        k0[i3*dof+0, i3*dof+3] += (f3*nx*(B11*f3*nx + B16*f3*ny) + f3*ny*(B16*f3*nx + B66*f3*ny))/As
        k0[i3*dof+0, i3*dof+4] += (f3*nx*(B16*f3*nx + B66*f3*ny) + f3*ny*(B12*f3*nx + B26*f3*ny))/As
        k0[i3*dof+1, i1*dof+0] += (f1*nx*(A12*f3*ny + A16*f3*nx) + f1*ny*(A26*f3*ny + A66*f3*nx))/As
        k0[i3*dof+1, i1*dof+1] += (f1*nx*(A26*f3*ny + A66*f3*nx) + f1*ny*(A22*f3*ny + A26*f3*nx))/As
        k0[i3*dof+1, i1*dof+3] += (f1*nx*(B12*f3*ny + B16*f3*nx) + f1*ny*(B26*f3*ny + B66*f3*nx))/As
        k0[i3*dof+1, i1*dof+4] += (f1*nx*(B26*f3*ny + B66*f3*nx) + f1*ny*(B22*f3*ny + B26*f3*nx))/As
        k0[i3*dof+1, i2*dof+0] += (f2*nx*(A12*f3*ny + A16*f3*nx) + f2*ny*(A26*f3*ny + A66*f3*nx))/As
        k0[i3*dof+1, i2*dof+1] += (f2*nx*(A26*f3*ny + A66*f3*nx) + f2*ny*(A22*f3*ny + A26*f3*nx))/As
        k0[i3*dof+1, i2*dof+3] += (f2*nx*(B12*f3*ny + B16*f3*nx) + f2*ny*(B26*f3*ny + B66*f3*nx))/As
        k0[i3*dof+1, i2*dof+4] += (f2*nx*(B26*f3*ny + B66*f3*nx) + f2*ny*(B22*f3*ny + B26*f3*nx))/As
        k0[i3*dof+1, i3*dof+0] += (f3*nx*(A12*f3*ny + A16*f3*nx) + f3*ny*(A26*f3*ny + A66*f3*nx))/As
        k0[i3*dof+1, i3*dof+1] += (f3*nx*(A26*f3*ny + A66*f3*nx) + f3*ny*(A22*f3*ny + A26*f3*nx))/As
        k0[i3*dof+1, i3*dof+3] += (f3*nx*(B12*f3*ny + B16*f3*nx) + f3*ny*(B26*f3*ny + B66*f3*nx))/As
        k0[i3*dof+1, i3*dof+4] += (f3*nx*(B26*f3*ny + B66*f3*nx) + f3*ny*(B22*f3*ny + B26*f3*nx))/As
        k0[i3*dof+3, i1*dof+0] += (f1*nx*(B11*f3*nx + B16*f3*ny) + f1*ny*(B16*f3*nx + B66*f3*ny))/As
        k0[i3*dof+3, i1*dof+1] += (f1*nx*(B16*f3*nx + B66*f3*ny) + f1*ny*(B12*f3*nx + B26*f3*ny))/As
        k0[i3*dof+3, i1*dof+3] += (f1*nx*(D11*f3*nx + D16*f3*ny) + f1*ny*(D16*f3*nx + D66*f3*ny))/As
        k0[i3*dof+3, i1*dof+4] += (f1*nx*(D16*f3*nx + D66*f3*ny) + f1*ny*(D12*f3*nx + D26*f3*ny))/As
        k0[i3*dof+3, i2*dof+0] += (f2*nx*(B11*f3*nx + B16*f3*ny) + f2*ny*(B16*f3*nx + B66*f3*ny))/As
        k0[i3*dof+3, i2*dof+1] += (f2*nx*(B16*f3*nx + B66*f3*ny) + f2*ny*(B12*f3*nx + B26*f3*ny))/As
        k0[i3*dof+3, i2*dof+3] += (f2*nx*(D11*f3*nx + D16*f3*ny) + f2*ny*(D16*f3*nx + D66*f3*ny))/As
        k0[i3*dof+3, i2*dof+4] += (f2*nx*(D16*f3*nx + D66*f3*ny) + f2*ny*(D12*f3*nx + D26*f3*ny))/As
        k0[i3*dof+3, i3*dof+0] += (f3*nx*(B11*f3*nx + B16*f3*ny) + f3*ny*(B16*f3*nx + B66*f3*ny))/As
        k0[i3*dof+3, i3*dof+1] += (f3*nx*(B16*f3*nx + B66*f3*ny) + f3*ny*(B12*f3*nx + B26*f3*ny))/As
        k0[i3*dof+3, i3*dof+3] += (f3*nx*(D11*f3*nx + D16*f3*ny) + f3*ny*(D16*f3*nx + D66*f3*ny))/As
        k0[i3*dof+3, i3*dof+4] += (f3*nx*(D16*f3*nx + D66*f3*ny) + f3*ny*(D12*f3*nx + D26*f3*ny))/As
        k0[i3*dof+4, i1*dof+0] += (f1*nx*(B12*f3*ny + B16*f3*nx) + f1*ny*(B26*f3*ny + B66*f3*nx))/As
        k0[i3*dof+4, i1*dof+1] += (f1*nx*(B26*f3*ny + B66*f3*nx) + f1*ny*(B22*f3*ny + B26*f3*nx))/As
        k0[i3*dof+4, i1*dof+3] += (f1*nx*(D12*f3*ny + D16*f3*nx) + f1*ny*(D26*f3*ny + D66*f3*nx))/As
        k0[i3*dof+4, i1*dof+4] += (f1*nx*(D26*f3*ny + D66*f3*nx) + f1*ny*(D22*f3*ny + D26*f3*nx))/As
        k0[i3*dof+4, i2*dof+0] += (f2*nx*(B12*f3*ny + B16*f3*nx) + f2*ny*(B26*f3*ny + B66*f3*nx))/As
        k0[i3*dof+4, i2*dof+1] += (f2*nx*(B26*f3*ny + B66*f3*nx) + f2*ny*(B22*f3*ny + B26*f3*nx))/As
        k0[i3*dof+4, i2*dof+3] += (f2*nx*(D12*f3*ny + D16*f3*nx) + f2*ny*(D26*f3*ny + D66*f3*nx))/As
        k0[i3*dof+4, i2*dof+4] += (f2*nx*(D26*f3*ny + D66*f3*nx) + f2*ny*(D22*f3*ny + D26*f3*nx))/As
        k0[i3*dof+4, i3*dof+0] += (f3*nx*(B12*f3*ny + B16*f3*nx) + f3*ny*(B26*f3*ny + B66*f3*nx))/As
        k0[i3*dof+4, i3*dof+1] += (f3*nx*(B26*f3*ny + B66*f3*nx) + f3*ny*(B22*f3*ny + B26*f3*nx))/As
        k0[i3*dof+4, i3*dof+3] += (f3*nx*(D12*f3*ny + D16*f3*nx) + f3*ny*(D26*f3*ny + D66*f3*nx))/As
        k0[i3*dof+4, i3*dof+4] += (f3*nx*(D26*f3*ny + D66*f3*nx) + f3*ny*(D22*f3*ny + D26*f3*nx))/As

# force vector
fext = np.zeros(n*dof, dtype=np.float64)
fext[nodes[0].index*dof + 0] = 50000000.
fext[nodes[3].index*dof + 0] = 100000000.
fext[nodes[6].index*dof + 0] = 50000000.

# boundary conditions
for i in [2, 5, 8]:
    k0[nodes[i].index*dof+0, :] = 0
    k0[:, nodes[i].index*dof+0] = 0
    k0[nodes[i].index*dof+1, :] = 0
    k0[:, nodes[i].index*dof+1] = 0

sparse = True
if sparse:
    k0 = coo_matrix(k0)
    print(repr(k0))
    assert is_symmetric(k0)
    u = solve(k0, fext)
else:
    ind = k0.sum(axis=0) == 0
    k0 = k0[~ind, :][:, ~ind]
    fext = fext[~ind]
    u = np.linalg.solve(k0, fext)

for nindi, node in ind2node.items():
    for i in range(3):
        try:
            node.pos[i] += u[nindi*dof+i]
        except:
            pass

xcord = [node.pos[0] for node in nodes]
ycord = [node.pos[1] for node in nodes]
plt.scatter(xcord, ycord)

for edge in edges:
    plt.plot([edge.n1.pos[0], edge.n2.pos[0]],
             [edge.n1.pos[1], edge.n2.pos[1]], '--r', lw=0.5, mfc=None)

for i in [0, 2, 4, 7, 8, 9, -1]:
    print()
    ipt = edges[i].ipts[0]
    print(ipt.pos, ipt.nx, ipt.ny)
    ipt = edges[i].ipts[1]
    print(ipt.pos, ipt.nx, ipt.ny)
    if len(edges[i].ipts) > 2:
        ipt = edges[i].ipts[2]
        print(ipt.pos, ipt.nx, ipt.ny)
    if len(edges[i].ipts) > 3:
        ipt = edges[i].ipts[3]
        print(ipt.pos, ipt.nx, ipt.ny)

plt.show()
