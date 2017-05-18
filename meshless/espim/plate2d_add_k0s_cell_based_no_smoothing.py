from __future__ import absolute_import, division

import numpy as np

from ..logger import msg
from ..constants import ZGLOBAL


def add_k0s(k0, mesh, prop_from_node, silent=True):
    msg('Adding K0s to K0...', silent=silent)
    dof = 5
    for tria in mesh.elements.values():
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
            E = k * (1/3*pn1.E + 1/3*pn2.E + 1/3*pn3.E)
        else:
            E = k * tria.prop.E
        E44 = E[0, 0]
        E45 = E[0, 1]
        E55 = E[1, 1]

        i1 = n1.index
        i2 = n2.index
        i3 = n3.index

        k0[i1*dof+2, i1*dof+2] += (-a1 + c1)*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(2*Ac) + (b1 - d1)*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(2*Ac)
        k0[i1*dof+2, i1*dof+3] += E44*(b1/2 - d1/2)/2 + E45*(-a1/2 + c1/2)/2
        k0[i1*dof+2, i1*dof+4] += E45*(b1/2 - d1/2)/2 + E55*(-a1/2 + c1/2)/2
        k0[i1*dof+2, i2*dof+2] += -c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(2*Ac) + d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(2*Ac)
        k0[i1*dof+2, i2*dof+3] += -a1*c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) + a1*d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+2, i2*dof+4] += -b1*c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) + b1*d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+2, i3*dof+2] += a1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(2*Ac) - b1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(2*Ac)
        k0[i1*dof+2, i3*dof+3] += a1*c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) - b1*c1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+2, i3*dof+4] += a1*d1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) - b1*d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        k0[i1*dof+3, i1*dof+2] += E44*(b1 - d1)/4 + E45*(-a1 + c1)/4
        k0[i1*dof+3, i1*dof+3] += Ac*E44/4
        k0[i1*dof+3, i1*dof+4] += Ac*E45/4
        k0[i1*dof+3, i2*dof+2] += E44*d1/4 - E45*c1/4
        k0[i1*dof+3, i2*dof+3] += E44*a1*d1/8 - E45*a1*c1/8
        k0[i1*dof+3, i2*dof+4] += E44*b1*d1/8 - E45*b1*c1/8
        k0[i1*dof+3, i3*dof+2] += -E44*b1/4 + E45*a1/4
        k0[i1*dof+3, i3*dof+3] += -E44*b1*c1/8 + E45*a1*c1/8
        k0[i1*dof+3, i3*dof+4] += -E44*b1*d1/8 + E45*a1*d1/8
        k0[i1*dof+4, i1*dof+2] += E45*(b1 - d1)/4 + E55*(-a1 + c1)/4
        k0[i1*dof+4, i1*dof+3] += Ac*E45/4
        k0[i1*dof+4, i1*dof+4] += Ac*E55/4
        k0[i1*dof+4, i2*dof+2] += E45*d1/4 - E55*c1/4
        k0[i1*dof+4, i2*dof+3] += E45*a1*d1/8 - E55*a1*c1/8
        k0[i1*dof+4, i2*dof+4] += E45*b1*d1/8 - E55*b1*c1/8
        k0[i1*dof+4, i3*dof+2] += -E45*b1/4 + E55*a1/4
        k0[i1*dof+4, i3*dof+3] += -E45*b1*c1/8 + E55*a1*c1/8
        k0[i1*dof+4, i3*dof+4] += -E45*b1*d1/8 + E55*a1*d1/8
        k0[i2*dof+2, i1*dof+2] += (-a1 + c1)*(E45*d1/2 - E55*c1/2)/(2*Ac) + (b1 - d1)*(E44*d1/2 - E45*c1/2)/(2*Ac)
        k0[i2*dof+2, i1*dof+3] += E44*d1/4 - E45*c1/4
        k0[i2*dof+2, i1*dof+4] += E45*d1/4 - E55*c1/4
        k0[i2*dof+2, i2*dof+2] += -c1*(E45*d1/2 - E55*c1/2)/(2*Ac) + d1*(E44*d1/2 - E45*c1/2)/(2*Ac)
        k0[i2*dof+2, i2*dof+3] += -a1*c1*(E45*d1/2 - E55*c1/2)/(4*Ac) + a1*d1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        k0[i2*dof+2, i2*dof+4] += -b1*c1*(E45*d1/2 - E55*c1/2)/(4*Ac) + b1*d1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        k0[i2*dof+2, i3*dof+2] += a1*(E45*d1/2 - E55*c1/2)/(2*Ac) - b1*(E44*d1/2 - E45*c1/2)/(2*Ac)
        k0[i2*dof+2, i3*dof+3] += a1*c1*(E45*d1/2 - E55*c1/2)/(4*Ac) - b1*c1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        k0[i2*dof+2, i3*dof+4] += a1*d1*(E45*d1/2 - E55*c1/2)/(4*Ac) - b1*d1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        k0[i2*dof+3, i1*dof+2] += (-a1 + c1)*(E45*a1*d1/4 - E55*a1*c1/4)/(2*Ac) + (b1 - d1)*(E44*a1*d1/4 - E45*a1*c1/4)/(2*Ac)
        k0[i2*dof+3, i1*dof+3] += E44*a1*d1/8 - E45*a1*c1/8
        k0[i2*dof+3, i1*dof+4] += E45*a1*d1/8 - E55*a1*c1/8
        k0[i2*dof+3, i2*dof+2] += -c1*(E45*a1*d1/4 - E55*a1*c1/4)/(2*Ac) + d1*(E44*a1*d1/4 - E45*a1*c1/4)/(2*Ac)
        k0[i2*dof+3, i2*dof+3] += -a1*c1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) + a1*d1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        k0[i2*dof+3, i2*dof+4] += -b1*c1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) + b1*d1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        k0[i2*dof+3, i3*dof+2] += a1*(E45*a1*d1/4 - E55*a1*c1/4)/(2*Ac) - b1*(E44*a1*d1/4 - E45*a1*c1/4)/(2*Ac)
        k0[i2*dof+3, i3*dof+3] += a1*c1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) - b1*c1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        k0[i2*dof+3, i3*dof+4] += a1*d1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) - b1*d1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        k0[i2*dof+4, i1*dof+2] += (-a1 + c1)*(E45*b1*d1/4 - E55*b1*c1/4)/(2*Ac) + (b1 - d1)*(E44*b1*d1/4 - E45*b1*c1/4)/(2*Ac)
        k0[i2*dof+4, i1*dof+3] += E44*b1*d1/8 - E45*b1*c1/8
        k0[i2*dof+4, i1*dof+4] += E45*b1*d1/8 - E55*b1*c1/8
        k0[i2*dof+4, i2*dof+2] += -c1*(E45*b1*d1/4 - E55*b1*c1/4)/(2*Ac) + d1*(E44*b1*d1/4 - E45*b1*c1/4)/(2*Ac)
        k0[i2*dof+4, i2*dof+3] += -a1*c1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) + a1*d1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        k0[i2*dof+4, i2*dof+4] += -b1*c1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) + b1*d1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        k0[i2*dof+4, i3*dof+2] += a1*(E45*b1*d1/4 - E55*b1*c1/4)/(2*Ac) - b1*(E44*b1*d1/4 - E45*b1*c1/4)/(2*Ac)
        k0[i2*dof+4, i3*dof+3] += a1*c1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) - b1*c1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        k0[i2*dof+4, i3*dof+4] += a1*d1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) - b1*d1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        k0[i3*dof+2, i1*dof+2] += (-a1 + c1)*(-E45*b1/2 + E55*a1/2)/(2*Ac) + (b1 - d1)*(-E44*b1/2 + E45*a1/2)/(2*Ac)
        k0[i3*dof+2, i1*dof+3] += -E44*b1/4 + E45*a1/4
        k0[i3*dof+2, i1*dof+4] += -E45*b1/4 + E55*a1/4
        k0[i3*dof+2, i2*dof+2] += -c1*(-E45*b1/2 + E55*a1/2)/(2*Ac) + d1*(-E44*b1/2 + E45*a1/2)/(2*Ac)
        k0[i3*dof+2, i2*dof+3] += -a1*c1*(-E45*b1/2 + E55*a1/2)/(4*Ac) + a1*d1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        k0[i3*dof+2, i2*dof+4] += -b1*c1*(-E45*b1/2 + E55*a1/2)/(4*Ac) + b1*d1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        k0[i3*dof+2, i3*dof+2] += a1*(-E45*b1/2 + E55*a1/2)/(2*Ac) - b1*(-E44*b1/2 + E45*a1/2)/(2*Ac)
        k0[i3*dof+2, i3*dof+3] += a1*c1*(-E45*b1/2 + E55*a1/2)/(4*Ac) - b1*c1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        k0[i3*dof+2, i3*dof+4] += a1*d1*(-E45*b1/2 + E55*a1/2)/(4*Ac) - b1*d1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        k0[i3*dof+3, i1*dof+2] += (-a1 + c1)*(-E45*b1*c1/4 + E55*a1*c1/4)/(2*Ac) + (b1 - d1)*(-E44*b1*c1/4 + E45*a1*c1/4)/(2*Ac)
        k0[i3*dof+3, i1*dof+3] += -E44*b1*c1/8 + E45*a1*c1/8
        k0[i3*dof+3, i1*dof+4] += -E45*b1*c1/8 + E55*a1*c1/8
        k0[i3*dof+3, i2*dof+2] += -c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(2*Ac) + d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(2*Ac)
        k0[i3*dof+3, i2*dof+3] += -a1*c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) + a1*d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        k0[i3*dof+3, i2*dof+4] += -b1*c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) + b1*d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        k0[i3*dof+3, i3*dof+2] += a1*(-E45*b1*c1/4 + E55*a1*c1/4)/(2*Ac) - b1*(-E44*b1*c1/4 + E45*a1*c1/4)/(2*Ac)
        k0[i3*dof+3, i3*dof+3] += a1*c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) - b1*c1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        k0[i3*dof+3, i3*dof+4] += a1*d1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) - b1*d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        k0[i3*dof+4, i1*dof+2] += (-a1 + c1)*(-E45*b1*d1/4 + E55*a1*d1/4)/(2*Ac) + (b1 - d1)*(-E44*b1*d1/4 + E45*a1*d1/4)/(2*Ac)
        k0[i3*dof+4, i1*dof+3] += -E44*b1*d1/8 + E45*a1*d1/8
        k0[i3*dof+4, i1*dof+4] += -E45*b1*d1/8 + E55*a1*d1/8
        k0[i3*dof+4, i2*dof+2] += -c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(2*Ac) + d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(2*Ac)
        k0[i3*dof+4, i2*dof+3] += -a1*c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) + a1*d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)
        k0[i3*dof+4, i2*dof+4] += -b1*c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) + b1*d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)
        k0[i3*dof+4, i3*dof+2] += a1*(-E45*b1*d1/4 + E55*a1*d1/4)/(2*Ac) - b1*(-E44*b1*d1/4 + E45*a1*d1/4)/(2*Ac)
        k0[i3*dof+4, i3*dof+3] += a1*c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) - b1*c1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)
        k0[i3*dof+4, i3*dof+4] += a1*d1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) - b1*d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)

    msg('finished!', silent=silent)
    return k0
