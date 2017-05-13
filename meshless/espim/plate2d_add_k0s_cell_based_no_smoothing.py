from __future__ import absolute_import, division

import numpy as np

from ..logger import msg
from ..constants import ZGLOBAL


def add_k0s(k0, trias, prop_from_node, silent=True):
    msg('Adding K0s to K0...', silent=silent)
    dof = 5
    for tria in trias:
        # n1 -> n2 -> n3 -> n1

        n1, n2, n3 = tria.nodes

        if np.dot(np.cross((n2.xyz - n1.xyz), (n3.xyz - n1.xyz)), ZGLOBAL) < 0:
            n1, n2, n3 = n2, n1, n3

        Ac = tria.get_area()

        x1, y1, z1 = n1.xyz
        x2, y2, z2 = n2.xyz
        x3, y3, z3 = n3.xyz
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = x3 - x1
        d1 = y3 - y1

        k = 5/6
        if prop_from_node:
            pn1 = n1.prop
            pn2 = n2.prop
            pn3 = n3.prop
            G = k * (1/3*pn1.E*pn1.t**2 + 1/3*pn2.E*pn2.t**2 + 1/3*pn3.E*pn3.t**2)
        else:
            G = k * tria.prop.E * tria.prop.t**2
        G44 = G[0, 0]
        G45 = G[0, 1]
        G55 = G[1, 1]

        i1 = n1.index
        i2 = n2.index
        i3 = n3.index

        k0[i1*dof+2, i1*dof+2] += (-a1 + c1)*(G45*(b1/2 - d1/2) + G55*(-a1/2 + c1/2))/(2*Ac) + (b1 - d1)*(G44*(b1/2 - d1/2) + G45*(-a1/2 + c1/2))/(2*Ac)
        k0[i1*dof+2, i1*dof+3] += G44*(b1/2 - d1/2)/2 + G45*(-a1/2 + c1/2)/2
        k0[i1*dof+2, i1*dof+4] += G45*(b1/2 - d1/2)/2 + G55*(-a1/2 + c1/2)/2
        k0[i1*dof+2, i2*dof+2] += -c1*(G45*(b1/2 - d1/2) + G55*(-a1/2 + c1/2))/(2*Ac) + d1*(G44*(b1/2 - d1/2) + G45*(-a1/2 + c1/2))/(2*Ac)
        k0[i1*dof+2, i2*dof+3] += -a1*c1*(G45*(b1/2 - d1/2) + G55*(-a1/2 + c1/2))/(4*Ac) + a1*d1*(G44*(b1/2 - d1/2) + G45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+2, i2*dof+4] += -b1*c1*(G45*(b1/2 - d1/2) + G55*(-a1/2 + c1/2))/(4*Ac) + b1*d1*(G44*(b1/2 - d1/2) + G45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+2, i3*dof+2] += a1*(G45*(b1/2 - d1/2) + G55*(-a1/2 + c1/2))/(2*Ac) - b1*(G44*(b1/2 - d1/2) + G45*(-a1/2 + c1/2))/(2*Ac)
        k0[i1*dof+2, i3*dof+3] += a1*c1*(G45*(b1/2 - d1/2) + G55*(-a1/2 + c1/2))/(4*Ac) - b1*c1*(G44*(b1/2 - d1/2) + G45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+2, i3*dof+4] += a1*d1*(G45*(b1/2 - d1/2) + G55*(-a1/2 + c1/2))/(4*Ac) - b1*d1*(G44*(b1/2 - d1/2) + G45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+3, i1*dof+2] += G44*(b1 - d1)/4 + G45*(-a1 + c1)/4
        k0[i1*dof+3, i1*dof+3] += Ac*G44/4
        k0[i1*dof+3, i1*dof+4] += Ac*G45/4
        k0[i1*dof+3, i2*dof+2] += G44*d1/4 - G45*c1/4
        k0[i1*dof+3, i2*dof+3] += G44*a1*d1/8 - G45*a1*c1/8
        k0[i1*dof+3, i2*dof+4] += G44*b1*d1/8 - G45*b1*c1/8
        k0[i1*dof+3, i3*dof+2] += -G44*b1/4 + G45*a1/4
        k0[i1*dof+3, i3*dof+3] += -G44*b1*c1/8 + G45*a1*c1/8
        k0[i1*dof+3, i3*dof+4] += -G44*b1*d1/8 + G45*a1*d1/8
        k0[i1*dof+4, i1*dof+2] += G45*(b1 - d1)/4 + G55*(-a1 + c1)/4
        k0[i1*dof+4, i1*dof+3] += Ac*G45/4
        k0[i1*dof+4, i1*dof+4] += Ac*G55/4
        k0[i1*dof+4, i2*dof+2] += G45*d1/4 - G55*c1/4
        k0[i1*dof+4, i2*dof+3] += G45*a1*d1/8 - G55*a1*c1/8
        k0[i1*dof+4, i2*dof+4] += G45*b1*d1/8 - G55*b1*c1/8
        k0[i1*dof+4, i3*dof+2] += -G45*b1/4 + G55*a1/4
        k0[i1*dof+4, i3*dof+3] += -G45*b1*c1/8 + G55*a1*c1/8
        k0[i1*dof+4, i3*dof+4] += -G45*b1*d1/8 + G55*a1*d1/8
        k0[i2*dof+2, i1*dof+2] += (-a1 + c1)*(G45*d1/2 - G55*c1/2)/(2*Ac) + (b1 - d1)*(G44*d1/2 - G45*c1/2)/(2*Ac)
        k0[i2*dof+2, i1*dof+3] += G44*d1/4 - G45*c1/4
        k0[i2*dof+2, i1*dof+4] += G45*d1/4 - G55*c1/4
        k0[i2*dof+2, i2*dof+2] += -c1*(G45*d1/2 - G55*c1/2)/(2*Ac) + d1*(G44*d1/2 - G45*c1/2)/(2*Ac)
        k0[i2*dof+2, i2*dof+3] += -a1*c1*(G45*d1/2 - G55*c1/2)/(4*Ac) + a1*d1*(G44*d1/2 - G45*c1/2)/(4*Ac)
        k0[i2*dof+2, i2*dof+4] += -b1*c1*(G45*d1/2 - G55*c1/2)/(4*Ac) + b1*d1*(G44*d1/2 - G45*c1/2)/(4*Ac)
        k0[i2*dof+2, i3*dof+2] += a1*(G45*d1/2 - G55*c1/2)/(2*Ac) - b1*(G44*d1/2 - G45*c1/2)/(2*Ac)
        k0[i2*dof+2, i3*dof+3] += a1*c1*(G45*d1/2 - G55*c1/2)/(4*Ac) - b1*c1*(G44*d1/2 - G45*c1/2)/(4*Ac)
        k0[i2*dof+2, i3*dof+4] += a1*d1*(G45*d1/2 - G55*c1/2)/(4*Ac) - b1*d1*(G44*d1/2 - G45*c1/2)/(4*Ac)
        k0[i2*dof+3, i1*dof+2] += (-a1 + c1)*(G45*a1*d1/4 - G55*a1*c1/4)/(2*Ac) + (b1 - d1)*(G44*a1*d1/4 - G45*a1*c1/4)/(2*Ac)
        k0[i2*dof+3, i1*dof+3] += G44*a1*d1/8 - G45*a1*c1/8
        k0[i2*dof+3, i1*dof+4] += G45*a1*d1/8 - G55*a1*c1/8
        k0[i2*dof+3, i2*dof+2] += -c1*(G45*a1*d1/4 - G55*a1*c1/4)/(2*Ac) + d1*(G44*a1*d1/4 - G45*a1*c1/4)/(2*Ac)
        k0[i2*dof+3, i2*dof+3] += -a1*c1*(G45*a1*d1/4 - G55*a1*c1/4)/(4*Ac) + a1*d1*(G44*a1*d1/4 - G45*a1*c1/4)/(4*Ac)
        k0[i2*dof+3, i2*dof+4] += -b1*c1*(G45*a1*d1/4 - G55*a1*c1/4)/(4*Ac) + b1*d1*(G44*a1*d1/4 - G45*a1*c1/4)/(4*Ac)
        k0[i2*dof+3, i3*dof+2] += a1*(G45*a1*d1/4 - G55*a1*c1/4)/(2*Ac) - b1*(G44*a1*d1/4 - G45*a1*c1/4)/(2*Ac)
        k0[i2*dof+3, i3*dof+3] += a1*c1*(G45*a1*d1/4 - G55*a1*c1/4)/(4*Ac) - b1*c1*(G44*a1*d1/4 - G45*a1*c1/4)/(4*Ac)
        k0[i2*dof+3, i3*dof+4] += a1*d1*(G45*a1*d1/4 - G55*a1*c1/4)/(4*Ac) - b1*d1*(G44*a1*d1/4 - G45*a1*c1/4)/(4*Ac)
        k0[i2*dof+4, i1*dof+2] += (-a1 + c1)*(G45*b1*d1/4 - G55*b1*c1/4)/(2*Ac) + (b1 - d1)*(G44*b1*d1/4 - G45*b1*c1/4)/(2*Ac)
        k0[i2*dof+4, i1*dof+3] += G44*b1*d1/8 - G45*b1*c1/8
        k0[i2*dof+4, i1*dof+4] += G45*b1*d1/8 - G55*b1*c1/8
        k0[i2*dof+4, i2*dof+2] += -c1*(G45*b1*d1/4 - G55*b1*c1/4)/(2*Ac) + d1*(G44*b1*d1/4 - G45*b1*c1/4)/(2*Ac)
        k0[i2*dof+4, i2*dof+3] += -a1*c1*(G45*b1*d1/4 - G55*b1*c1/4)/(4*Ac) + a1*d1*(G44*b1*d1/4 - G45*b1*c1/4)/(4*Ac)
        k0[i2*dof+4, i2*dof+4] += -b1*c1*(G45*b1*d1/4 - G55*b1*c1/4)/(4*Ac) + b1*d1*(G44*b1*d1/4 - G45*b1*c1/4)/(4*Ac)
        k0[i2*dof+4, i3*dof+2] += a1*(G45*b1*d1/4 - G55*b1*c1/4)/(2*Ac) - b1*(G44*b1*d1/4 - G45*b1*c1/4)/(2*Ac)
        k0[i2*dof+4, i3*dof+3] += a1*c1*(G45*b1*d1/4 - G55*b1*c1/4)/(4*Ac) - b1*c1*(G44*b1*d1/4 - G45*b1*c1/4)/(4*Ac)
        k0[i2*dof+4, i3*dof+4] += a1*d1*(G45*b1*d1/4 - G55*b1*c1/4)/(4*Ac) - b1*d1*(G44*b1*d1/4 - G45*b1*c1/4)/(4*Ac)
        k0[i3*dof+2, i1*dof+2] += (-a1 + c1)*(-G45*b1/2 + G55*a1/2)/(2*Ac) + (b1 - d1)*(-G44*b1/2 + G45*a1/2)/(2*Ac)
        k0[i3*dof+2, i1*dof+3] += -G44*b1/4 + G45*a1/4
        k0[i3*dof+2, i1*dof+4] += -G45*b1/4 + G55*a1/4
        k0[i3*dof+2, i2*dof+2] += -c1*(-G45*b1/2 + G55*a1/2)/(2*Ac) + d1*(-G44*b1/2 + G45*a1/2)/(2*Ac)
        k0[i3*dof+2, i2*dof+3] += -a1*c1*(-G45*b1/2 + G55*a1/2)/(4*Ac) + a1*d1*(-G44*b1/2 + G45*a1/2)/(4*Ac)
        k0[i3*dof+2, i2*dof+4] += -b1*c1*(-G45*b1/2 + G55*a1/2)/(4*Ac) + b1*d1*(-G44*b1/2 + G45*a1/2)/(4*Ac)
        k0[i3*dof+2, i3*dof+2] += a1*(-G45*b1/2 + G55*a1/2)/(2*Ac) - b1*(-G44*b1/2 + G45*a1/2)/(2*Ac)
        k0[i3*dof+2, i3*dof+3] += a1*c1*(-G45*b1/2 + G55*a1/2)/(4*Ac) - b1*c1*(-G44*b1/2 + G45*a1/2)/(4*Ac)
        k0[i3*dof+2, i3*dof+4] += a1*d1*(-G45*b1/2 + G55*a1/2)/(4*Ac) - b1*d1*(-G44*b1/2 + G45*a1/2)/(4*Ac)
        k0[i3*dof+3, i1*dof+2] += (-a1 + c1)*(-G45*b1*c1/4 + G55*a1*c1/4)/(2*Ac) + (b1 - d1)*(-G44*b1*c1/4 + G45*a1*c1/4)/(2*Ac)
        k0[i3*dof+3, i1*dof+3] += -G44*b1*c1/8 + G45*a1*c1/8
        k0[i3*dof+3, i1*dof+4] += -G45*b1*c1/8 + G55*a1*c1/8
        k0[i3*dof+3, i2*dof+2] += -c1*(-G45*b1*c1/4 + G55*a1*c1/4)/(2*Ac) + d1*(-G44*b1*c1/4 + G45*a1*c1/4)/(2*Ac)
        k0[i3*dof+3, i2*dof+3] += -a1*c1*(-G45*b1*c1/4 + G55*a1*c1/4)/(4*Ac) + a1*d1*(-G44*b1*c1/4 + G45*a1*c1/4)/(4*Ac)
        k0[i3*dof+3, i2*dof+4] += -b1*c1*(-G45*b1*c1/4 + G55*a1*c1/4)/(4*Ac) + b1*d1*(-G44*b1*c1/4 + G45*a1*c1/4)/(4*Ac)
        k0[i3*dof+3, i3*dof+2] += a1*(-G45*b1*c1/4 + G55*a1*c1/4)/(2*Ac) - b1*(-G44*b1*c1/4 + G45*a1*c1/4)/(2*Ac)
        k0[i3*dof+3, i3*dof+3] += a1*c1*(-G45*b1*c1/4 + G55*a1*c1/4)/(4*Ac) - b1*c1*(-G44*b1*c1/4 + G45*a1*c1/4)/(4*Ac)
        k0[i3*dof+3, i3*dof+4] += a1*d1*(-G45*b1*c1/4 + G55*a1*c1/4)/(4*Ac) - b1*d1*(-G44*b1*c1/4 + G45*a1*c1/4)/(4*Ac)
        k0[i3*dof+4, i1*dof+2] += (-a1 + c1)*(-G45*b1*d1/4 + G55*a1*d1/4)/(2*Ac) + (b1 - d1)*(-G44*b1*d1/4 + G45*a1*d1/4)/(2*Ac)
        k0[i3*dof+4, i1*dof+3] += -G44*b1*d1/8 + G45*a1*d1/8
        k0[i3*dof+4, i1*dof+4] += -G45*b1*d1/8 + G55*a1*d1/8
        k0[i3*dof+4, i2*dof+2] += -c1*(-G45*b1*d1/4 + G55*a1*d1/4)/(2*Ac) + d1*(-G44*b1*d1/4 + G45*a1*d1/4)/(2*Ac)
        k0[i3*dof+4, i2*dof+3] += -a1*c1*(-G45*b1*d1/4 + G55*a1*d1/4)/(4*Ac) + a1*d1*(-G44*b1*d1/4 + G45*a1*d1/4)/(4*Ac)
        k0[i3*dof+4, i2*dof+4] += -b1*c1*(-G45*b1*d1/4 + G55*a1*d1/4)/(4*Ac) + b1*d1*(-G44*b1*d1/4 + G45*a1*d1/4)/(4*Ac)
        k0[i3*dof+4, i3*dof+2] += a1*(-G45*b1*d1/4 + G55*a1*d1/4)/(2*Ac) - b1*(-G44*b1*d1/4 + G45*a1*d1/4)/(2*Ac)
        k0[i3*dof+4, i3*dof+3] += a1*c1*(-G45*b1*d1/4 + G55*a1*d1/4)/(4*Ac) - b1*c1*(-G44*b1*d1/4 + G45*a1*d1/4)/(4*Ac)
        k0[i3*dof+4, i3*dof+4] += a1*d1*(-G45*b1*d1/4 + G55*a1*d1/4)/(4*Ac) - b1*d1*(-G44*b1*d1/4 + G45*a1*d1/4)/(4*Ac)

    msg('finished!', silent=silent)
    return k0
