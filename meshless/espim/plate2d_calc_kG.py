from __future__ import absolute_import, division

import numpy as np

from ..logger import msg
from .plate2d import area_of_polygon
from .read_mesh import getMid


def calc_kG(d, mesh, prop_from_node, silent=True):
    msg('Calculating KG...', silent=silent)
    dof = 5
    n = d.shape[0] // dof

    #TODO allocate less memory here...
    kG = np.zeros((n*dof, n*dof), dtype=np.float64)

    for edge in mesh.edges.values():
        tria1 = edge.trias[0]
        Ac = edge.Ac
        ipts = edge.ipts
        mid1 = getMid(tria1)
        tmp = np.array([mid1, edge.n2.xyz, edge.n1.xyz])
        Ac1 = area_of_polygon(tmp[:, 0], tmp[:, 1])
        if len(edge.trias) == 1:
            tria2 = None
        elif len(edge.trias) == 2:
            tria2 = edge.trias[1]
            mid2 = getMid(tria2)
            tmp = np.array([mid2, edge.n1.xyz, edge.n2.xyz])
            Ac2 = area_of_polygon(tmp[:, 0], tmp[:, 1])
        else:
            raise RuntimeError('Found %d trias for edge' % len(edge.trias))
        indices = set()
        for ipt in ipts:
            indices.add(ipt.n1.index)
            indices.add(ipt.n2.index)
            indices.add(ipt.n3.index)
        indices = sorted(list(indices))
        if len(ipts) == 3:
            indices.append(0) # fourth dummy index
        indexpos = dict([[ind, i] for i, ind in enumerate(indices)])
        i1, i2, i3, i4 = indices
        f1 = np.array([0, 0, 0, 0], dtype=float)
        f2 = np.array([0, 0, 0, 0], dtype=float)
        f3 = np.array([0, 0, 0, 0], dtype=float)
        f4 = np.array([0, 0, 0, 0], dtype=float)

        nx1 = ipts[0].nx
        ny1 = ipts[0].ny
        le1 = ipts[0].le
        f1[indexpos[ipts[0].n1.index]] = ipts[0].f1
        f1[indexpos[ipts[0].n2.index]] = ipts[0].f2
        f1[indexpos[ipts[0].n3.index]] = ipts[0].f3

        nx2 = ipts[1].nx
        ny2 = ipts[1].ny
        le2 = ipts[1].le
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

        if prop_from_node:
            pn1 = edge.n1.prop
            pn2 = edge.n2.prop
            po1 = edge.othernode1.prop
            if tria2 is None:
                A = 4/9*pn1.A + 4/9*pn2.A + 1/9*po1.A
                B = 4/9*pn1.B + 4/9*pn2.B + 1/9*po1.B
            else:
                po2 = edge.othernode2.prop
                A = 5/12*pn1.A + 5/12*pn2.A + 1/12*po1.A + 1/12*po2.A
                B = 5/12*pn1.B + 5/12*pn2.B + 1/12*po1.B + 1/12*po2.B
        else:
            prop1 = tria1.prop
            if tria2 is None:
                A = prop1.A
                B = prop1.B
            else:
                prop2 = tria2.prop
                A = (Ac1*prop1.A + Ac2*prop2.A)/Ac
                B = (Ac1*prop1.B + Ac2*prop2.B)/Ac

        d1 = d[i1*dof: i1*dof+5]
        d2 = d[i2*dof: i2*dof+5]
        d3 = d[i3*dof: i3*dof+5]
        d4 = d[i4*dof: i4*dof+5]
        # d1... are [4, 4] [4, 5]

        dc = np.dot(np.array([f1, f2, f3, f4]), np.array([d1, d2, d3, d4]))
        # dc is [4, 5]

        u_c = dc[:, 0]
        v_c = dc[:, 1]
        phix_c = dc[:, 3]
        phiy_c = dc[:, 4]

        les = np.array([le1, le2, le3, le4])
        nx = np.array([nx1, nx2, nx3, nx4])
        ny = np.array([ny1, ny2, ny3, ny4])

        em = np.zeros(3)
        em[0] = 1/Ac*(les*nx*u_c).sum()
        em[1] = 1/Ac*(les*ny*v_c).sum()
        em[2] = 1/Ac*(les*(ny*u_c + nx*v_c)).sum()

        eb = np.zeros(3)
        eb[0] = 1/Ac*(les*nx*phix_c).sum()
        eb[1] = 1/Ac*(les*ny*phiy_c).sum()
        eb[2] = 1/Ac*(les*(ny*phix_c + nx*phiy_c)).sum()

        Nxx, Nyy, Nxy = np.dot(A, em) + np.dot(B, eb)

        #TODO calculate only upper triangle
        kG[i1*dof+2, i1*dof+2] += Ac*((Nxx*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nxy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (Nxy*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nyy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
        kG[i1*dof+2, i2*dof+2] += Ac*((Nxx*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nxy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (Nxy*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nyy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
        kG[i1*dof+2, i3*dof+2] += Ac*((Nxx*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nxy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (Nxy*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nyy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
        kG[i1*dof+2, i4*dof+2] += Ac*((Nxx*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nxy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (Nxy*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + Nyy*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
        kG[i2*dof+2, i1*dof+2] += Ac*((Nxx*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nxy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (Nxy*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nyy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
        kG[i2*dof+2, i2*dof+2] += Ac*((Nxx*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nxy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (Nxy*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nyy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
        kG[i2*dof+2, i3*dof+2] += Ac*((Nxx*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nxy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (Nxy*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nyy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
        kG[i2*dof+2, i4*dof+2] += Ac*((Nxx*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nxy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (Nxy*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + Nyy*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
        kG[i3*dof+2, i1*dof+2] += Ac*((Nxx*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nxy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (Nxy*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nyy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
        kG[i3*dof+2, i2*dof+2] += Ac*((Nxx*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nxy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (Nxy*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nyy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
        kG[i3*dof+2, i3*dof+2] += Ac*((Nxx*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nxy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (Nxy*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nyy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
        kG[i3*dof+2, i4*dof+2] += Ac*((Nxx*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nxy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (Nxy*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + Nyy*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)
        kG[i4*dof+2, i1*dof+2] += Ac*((Nxx*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nxy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*nx1 + f21*le2*nx2 + f31*le3*nx3 + f41*le4*nx4)/Ac + (Nxy*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nyy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f11*le1*ny1 + f21*le2*ny2 + f31*le3*ny3 + f41*le4*ny4)/Ac)
        kG[i4*dof+2, i2*dof+2] += Ac*((Nxx*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nxy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*nx1 + f22*le2*nx2 + f32*le3*nx3 + f42*le4*nx4)/Ac + (Nxy*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nyy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f12*le1*ny1 + f22*le2*ny2 + f32*le3*ny3 + f42*le4*ny4)/Ac)
        kG[i4*dof+2, i3*dof+2] += Ac*((Nxx*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nxy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*nx1 + f23*le2*nx2 + f33*le3*nx3 + f43*le4*nx4)/Ac + (Nxy*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nyy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f13*le1*ny1 + f23*le2*ny2 + f33*le3*ny3 + f43*le4*ny4)/Ac)
        kG[i4*dof+2, i4*dof+2] += Ac*((Nxx*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nxy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + (Nxy*(f14*le1*nx1 + f24*le2*nx2 + f34*le3*nx3 + f44*le4*nx4)/Ac + Nyy*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)*(f14*le1*ny1 + f24*le2*ny2 + f34*le3*ny3 + f44*le4*ny4)/Ac)

    msg('finished!', silent=silent)
    return kG

