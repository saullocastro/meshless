from collections import defaultdict

from pyNastran.bdf.bdf import read_bdf, CTRIA3

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
        return 0.5*(self.n1 + self.n2)


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
