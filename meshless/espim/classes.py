class IntegrationPoint(object):
    """Entity used to carry data related to each integration point
    """
    def __init__(self, tria, n1, n2, n3, f1, f2, f3, nx, ny, nz, le):
        self.xyz = f1*n1.xyz + f2*n2.xyz + f3*n3.xyz
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


class Edge(object):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.nodes = [n1, n2]
        self.node_ids = [n1.nid, n2.nid]
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


