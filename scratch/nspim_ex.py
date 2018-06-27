from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

class Tria(object):
    def __init__(self, n1, n2, n3):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.edges = set()
        self.nodes = set([n1, n2, n3])

    def getMid(self):
        return 1/3*(self.n1 + self.n2 + self.n3)

class Edge(object):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.nodes = set([n1, n2])
        self.trias = set()

    def getMid(self):
        return 0.5*(self.n1 + self.n2)

class Node(object):
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.edges = set()
        self.trias = set()
        self.cell = None

    def __add__(self, node):
        if isinstance(node, Node):
            return Node(*(self.pos + node.pos))
        else:
            cte = node
            return Node(*(self.pos + cte))

    def __sub__(self, node):
        if isinstance(node, Node):
            return Node(*(self.pos - node.pos))
        else:
            cte = node
            return Node(*(self.pos - cte))

    def __rmul__(self, node):
        if isinstance(node, Node):
            return Node(*(self.pos * node.pos))
        else:
            cte = node
            return Node(*(self.pos * cte))
    def __lmul__(self, node):
        return self.__rmul__(node)

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


edges = [
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
        ]

for edge in edges:
    edge.n1.edges.add(edge)
    edge.n2.edges.add(edge)
    plt.plot([edge.n1.pos[0], edge.n2.pos[0]],
             [edge.n1.pos[1], edge.n2.pos[1]], '--r', mfc=None)
    for tria in trias:
        if len(edge.nodes & tria.nodes) == 2:
            tria.edges.add(edge)
            edge.trias.add(tria)
            for node in tria.nodes:
                node.trias.add(tria)

colors = cycle(['g', 'k', 'y', 'b'])

for node in nodes[...]:
    color = next(colors)

    edge0 = None
    boundary = False
    for edge in node.edges:
        if len(edge.trias) == 1: # boundary edge
            edge0 = edge # starting from a boundary edge
            boundary = True
    if edge0 is None:
        edge0 = list(node.edges)[0] # starting from any edge
        boundary = False

    edgesseq = [edge0]
    pickededges = set([edge0])

    for edge1 in cycle(node.edges):
        if edge1 in pickededges:
            continue
        if len(edge0.trias & edge1.trias) == 1:
            edgesseq.append(edge1)
            pickededges.add(edge1)
            edge0 = edge1

        #print(len(pickededges), len(node.edges), boundary)
        if len(pickededges) == len(node.edges):
            break

    cell = []
    for i in range(len(edgesseq) - 1):
        edge1 = edgesseq[i]
        edge2 = edgesseq[i+1]
        commontria = (edge1.trias & edge2.trias).pop()
        cg = commontria.getMid()
        if i == 0:
            cell.append(edge1.getMid())
        cell.append(cg)
        cell.append(edge2.getMid())
    if boundary:
        cell.append(node)
        cell.append(edgesseq[0].getMid())
    else:
        commontria = (edgesseq[-1].trias & edgesseq[0].trias).pop()
        cg = commontria.getMid()
        cell.append(cg)
        cell.append(edgesseq[0].getMid())

    node.cell = cell
    print(len(cell))

    xcoord = [pt.pos[0] for pt in cell]
    ycoord = [pt.pos[1] for pt in cell]
    plt.plot(xcoord, ycoord, '-' + color, mfc=None)


xcord = [node.pos[0] for node in nodes]
ycord = [node.pos[1] for node in nodes]
plt.scatter(xcord, ycord)
plt.gca().set_aspect('equal')

plt.savefig('plot_equally_shared_smoothing_domain.png', bbox_inches='tight')
plt.show()

K = 0
for node in nodes:
    for edge in node.cell:
        pass
        #compute Kcell = sum(Kedge)  :  strain smoothing


# integration edge: 2 nodes (only cells at boundary)
# integration

# Lagrangian interpolation
# - at any cell, the evaluation of the integrand will be a function of three
#   nodes
# - points at triagle edges a function of 2 nodes
# - points at equally shared smoothing domain internal edges a function of 3
# nodes
# - an internal cell will have 13 integration points at least, one for each
# edge

# - there should be a way to quickly find the nodes belonging to any
# integration point

# example of integrand
# (du/dx)^T * E (du/dx)
# for a given integration point find the interpolation of u

# u = u1 * c1 + u2 * c2 + u3 * c3
# ci could be function of the area ratios for ESSD edges and a function of
# the line ratios for ESSD edges coinciding with triangle edges

#


# constitutive stiffness matrix

# u = u1*a1 + u2*a2 + u3*a3
# v = v1*a1 + v2*a2 + v3*a3
# w = w1*a1 + w2*a2 + w3*a3



# u = unodal * area_ratios(integration point position, nodal coordinates)



# generated matrices are expected to be sparse since there will be a maximum of
# six integration points connecting two degrees of freedom

# plate equation
# exx = ux
# eyy = vy
# gxy = uy + vx
# kxx = -wxx
# kyy = -wyy
# kxy = -2wxy


# figure out how the integration along x and y will be transformed to
# integration along the edges... one should properly consider material
# anisotropies during this integration procedure





