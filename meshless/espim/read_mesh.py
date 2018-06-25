from collections import defaultdict

import numpy as np
from pyNastran.bdf.bdf import read_bdf, BDF, CTRIA3, GRID

from ..logger import msg


class Edge(object):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.nodes = [n1, n2]
        self.trias = []
        self.sdomain = []
        self.ipts = []
        self.Ac = None
        self.othernode1 = None
        self.othernode2 = None

    def __str__(self):
        return 'Edge (%s, %s)' % (self.n1.nid, self.n2.nid)

    def __repr__(self):
        return self.__str__()

    def getMid(self):
        try:
            return 0.5*(self.n1 + self.n2)
        except:
            return 0.5*(self.n1.xyz + self.n2.xyz)


def getMid(tria):
    return tria.get_node_positions().mean(axis=0)


def read_mesh(filepath, silent=True):
    msg('Reading mesh...', silent=silent)
    mesh = read_bdf(filepath, debug=False)
    nodes = []
    for node in mesh.nodes.values():
        node.trias = set()
        node.edges = set()
        node.index = set()
        node.prop = None
        nodes.append(node)

    trias = []
    for elem in mesh.elements.values():
        if isinstance(elem, CTRIA3):
            elem.edges = []
            elem.prop = None
            trias.append(elem)
        else:
            raise NotImplementedError('Element type %s not supported' %
                    type(elem))
    edges = {}
    edges_ids = defaultdict(list)
    for tria in trias:
        for edge_id in tria.get_edge_ids():
            edges_ids[edge_id].append(tria)
    for (n1, n2), e_trias in edges_ids.items():
        edge = Edge(mesh.nodes[n1], mesh.nodes[n2])
        edge.trias = e_trias
        edges[(n1, n2)] = edge
    for edge in edges.values():
        for tria in edge.trias:
            tria.edges.append(edge)
    for tria in trias:
        for node in tria.nodes:
            node.trias.add(tria)
        for edge in tria.edges:
            node.edges.add(edge)
    mesh.edges = edges
    msg('finished!', silent=silent)
    return mesh


def read_delaunay(points, tri, silent=True):
    msg('Reading Delaunay ouptut...', silent=silent)
    mesh = BDF()
    nodes = []
    nid = 0
    for pt in points:
        if len(pt) == 2:
            pt = np.array([pt[0], pt[1], 0.])
        nid += 1
        node = GRID(nid, 0, pt)
        node.trias = set()
        node.edges = set()
        node.index = set()
        node.prop = None
        nodes.append(node)
        mesh.nodes[nid] = node
    eid = 0
    trias = []
    for i1, i2, i3 in tri.simplices:
        n1 = nodes[i1]
        n2 = nodes[i2]
        n3 = nodes[i3]
        eid += 1
        tria = CTRIA3(eid, 0, (n1.nid, n2.nid, n3.nid))
        tria.nodes = [n1, n2, n3]
        tria.nodes_ref = tria.nodes
        tria.edges = []
        tria.prop = None
        trias.append(tria)
        mesh.elements[eid] = tria
    edges = {}
    edges_ids = defaultdict(list)
    for tria in trias:
        for edge_id in tria.get_edge_ids():
            edges_ids[edge_id].append(tria)
    for (n1, n2), e_trias in edges_ids.items():
        edge = Edge(mesh.nodes[n1], mesh.nodes[n2])
        edge.trias = e_trias
        edges[(n1, n2)] = edge
    for edge in edges.values():
        for tria in edge.trias:
            tria.edges.append(edge)
    for tria in trias:
        for node in tria.nodes:
            node.trias.add(tria)
        for edge in tria.edges:
            node.edges.add(edge)
    mesh.edges = edges
    msg('finished!', silent=silent)
    return mesh

