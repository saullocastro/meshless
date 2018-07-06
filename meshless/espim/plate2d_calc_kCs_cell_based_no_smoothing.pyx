#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import absolute_import, division

from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np

from ..logger import msg
from ..constants import ZGLOBAL

ctypedef np.int64_t cINT
ctypedef np.double_t cDOUBLE


def calc_kCs(mesh, prop_from_node, alpha, silent=True):
    cdef int i1, i2, i3, i4, dof, c
    cdef double Ac, E44, E45, E55
    cdef double a1, b1, c1, d1
    cdef np.ndarray[cINT, ndim=1] kCsr, kCsc
    cdef np.ndarray[cDOUBLE, ndim=1] kCsv

    msg('Calculating KCs...', silent=silent)
    dof = 5

    size = len(mesh.nodes) * dof
    alloc = len(mesh.elements) * 81
    kCsr = np.zeros(alloc, dtype=np.int64)
    kCsc = np.zeros(alloc, dtype=np.int64)
    kCsv = np.zeros(alloc, dtype=np.float64)
    c = -1
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

        if prop_from_node:
            pn1 = n1.prop
            pn2 = n2.prop
            pn3 = n3.prop
            k13 = 1./3*pn1.scf_k13 + 1./3*pn2.scf_k13 + 1./3*pn3.scf_k13
            k23 = 1./3*pn1.scf_k23 + 1./3*pn2.scf_k23 + 1./3*pn3.scf_k23
            E = 1./3*pn1.E + 1./3*pn2.E + 1./3*pn3.E
            h = 1./3*pn1.h + 1./3*pn2.h + 1./3*pn3.h
        else:
            k13 = tria.prop.scf_k13
            k23 = tria.prop.scf_k23
            E = tria.prop.E
            h = tria.prop.h

        E44 = k13 * E[0, 0]
        E45 = min(k13, k23) * E[0, 1]
        E55 = k23 * E[1, 1]

        maxl = max([np.sum((e.n1.xyz - e.n2.xyz)**2)**0.5 for e in tria.edges])
        E44 = h**2 / (h**2 + alpha*maxl**2) * E44
        E45 = h**2 / (h**2 + alpha*maxl**2) * E45
        E55 = h**2 / (h**2 + alpha*maxl**2) * E55

        i1 = n1.index
        i2 = n2.index
        i3 = n3.index

        # kCs_sparse
        # kCs_sparse_num=81
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i1*dof+2
        kCsv[c] += (-a1 + c1)*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(2*Ac) + (b1 - d1)*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(2*Ac)
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i1*dof+3
        kCsv[c] += E44*(b1/2 - d1/2)/2 + E45*(-a1/2 + c1/2)/2
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i1*dof+4
        kCsv[c] += E45*(b1/2 - d1/2)/2 + E55*(-a1/2 + c1/2)/2
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i2*dof+2
        kCsv[c] += -c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(2*Ac) + d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(2*Ac)
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i2*dof+3
        kCsv[c] += -a1*c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) + a1*d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i2*dof+4
        kCsv[c] += -b1*c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) + b1*d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i3*dof+2
        kCsv[c] += a1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(2*Ac) - b1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(2*Ac)
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i3*dof+3
        kCsv[c] += a1*c1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) - b1*c1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        c += 1
        kCsr[c] = i1*dof+2
        kCsc[c] = i3*dof+4
        kCsv[c] += a1*d1*(E45*(b1/2 - d1/2) + E55*(-a1/2 + c1/2))/(4*Ac) - b1*d1*(E44*(b1/2 - d1/2) + E45*(-a1/2 + c1/2))/(4*Ac)
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i1*dof+2
        kCsv[c] += E44*(b1 - d1)/4 + E45*(-a1 + c1)/4
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i1*dof+3
        kCsv[c] += Ac*E44/4
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i1*dof+4
        kCsv[c] += Ac*E45/4
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i2*dof+2
        kCsv[c] += E44*d1/4 - E45*c1/4
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i2*dof+3
        kCsv[c] += E44*a1*d1/8 - E45*a1*c1/8
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i2*dof+4
        kCsv[c] += E44*b1*d1/8 - E45*b1*c1/8
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i3*dof+2
        kCsv[c] += -E44*b1/4 + E45*a1/4
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i3*dof+3
        kCsv[c] += -E44*b1*c1/8 + E45*a1*c1/8
        c += 1
        kCsr[c] = i1*dof+3
        kCsc[c] = i3*dof+4
        kCsv[c] += -E44*b1*d1/8 + E45*a1*d1/8
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i1*dof+2
        kCsv[c] += E45*(b1 - d1)/4 + E55*(-a1 + c1)/4
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i1*dof+3
        kCsv[c] += Ac*E45/4
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i1*dof+4
        kCsv[c] += Ac*E55/4
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i2*dof+2
        kCsv[c] += E45*d1/4 - E55*c1/4
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i2*dof+3
        kCsv[c] += E45*a1*d1/8 - E55*a1*c1/8
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i2*dof+4
        kCsv[c] += E45*b1*d1/8 - E55*b1*c1/8
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i3*dof+2
        kCsv[c] += -E45*b1/4 + E55*a1/4
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i3*dof+3
        kCsv[c] += -E45*b1*c1/8 + E55*a1*c1/8
        c += 1
        kCsr[c] = i1*dof+4
        kCsc[c] = i3*dof+4
        kCsv[c] += -E45*b1*d1/8 + E55*a1*d1/8
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i1*dof+2
        kCsv[c] += (-a1 + c1)*(E45*d1/2 - E55*c1/2)/(2*Ac) + (b1 - d1)*(E44*d1/2 - E45*c1/2)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i1*dof+3
        kCsv[c] += E44*d1/4 - E45*c1/4
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i1*dof+4
        kCsv[c] += E45*d1/4 - E55*c1/4
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i2*dof+2
        kCsv[c] += -c1*(E45*d1/2 - E55*c1/2)/(2*Ac) + d1*(E44*d1/2 - E45*c1/2)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i2*dof+3
        kCsv[c] += -a1*c1*(E45*d1/2 - E55*c1/2)/(4*Ac) + a1*d1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i2*dof+4
        kCsv[c] += -b1*c1*(E45*d1/2 - E55*c1/2)/(4*Ac) + b1*d1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i3*dof+2
        kCsv[c] += a1*(E45*d1/2 - E55*c1/2)/(2*Ac) - b1*(E44*d1/2 - E45*c1/2)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i3*dof+3
        kCsv[c] += a1*c1*(E45*d1/2 - E55*c1/2)/(4*Ac) - b1*c1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+2
        kCsc[c] = i3*dof+4
        kCsv[c] += a1*d1*(E45*d1/2 - E55*c1/2)/(4*Ac) - b1*d1*(E44*d1/2 - E45*c1/2)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i1*dof+2
        kCsv[c] += (-a1 + c1)*(E45*a1*d1/4 - E55*a1*c1/4)/(2*Ac) + (b1 - d1)*(E44*a1*d1/4 - E45*a1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i1*dof+3
        kCsv[c] += E44*a1*d1/8 - E45*a1*c1/8
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i1*dof+4
        kCsv[c] += E45*a1*d1/8 - E55*a1*c1/8
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i2*dof+2
        kCsv[c] += -c1*(E45*a1*d1/4 - E55*a1*c1/4)/(2*Ac) + d1*(E44*a1*d1/4 - E45*a1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i2*dof+3
        kCsv[c] += -a1*c1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) + a1*d1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i2*dof+4
        kCsv[c] += -b1*c1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) + b1*d1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i3*dof+2
        kCsv[c] += a1*(E45*a1*d1/4 - E55*a1*c1/4)/(2*Ac) - b1*(E44*a1*d1/4 - E45*a1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i3*dof+3
        kCsv[c] += a1*c1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) - b1*c1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+3
        kCsc[c] = i3*dof+4
        kCsv[c] += a1*d1*(E45*a1*d1/4 - E55*a1*c1/4)/(4*Ac) - b1*d1*(E44*a1*d1/4 - E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i1*dof+2
        kCsv[c] += (-a1 + c1)*(E45*b1*d1/4 - E55*b1*c1/4)/(2*Ac) + (b1 - d1)*(E44*b1*d1/4 - E45*b1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i1*dof+3
        kCsv[c] += E44*b1*d1/8 - E45*b1*c1/8
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i1*dof+4
        kCsv[c] += E45*b1*d1/8 - E55*b1*c1/8
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i2*dof+2
        kCsv[c] += -c1*(E45*b1*d1/4 - E55*b1*c1/4)/(2*Ac) + d1*(E44*b1*d1/4 - E45*b1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i2*dof+3
        kCsv[c] += -a1*c1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) + a1*d1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i2*dof+4
        kCsv[c] += -b1*c1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) + b1*d1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i3*dof+2
        kCsv[c] += a1*(E45*b1*d1/4 - E55*b1*c1/4)/(2*Ac) - b1*(E44*b1*d1/4 - E45*b1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i3*dof+3
        kCsv[c] += a1*c1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) - b1*c1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i2*dof+4
        kCsc[c] = i3*dof+4
        kCsv[c] += a1*d1*(E45*b1*d1/4 - E55*b1*c1/4)/(4*Ac) - b1*d1*(E44*b1*d1/4 - E45*b1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i1*dof+2
        kCsv[c] += (-a1 + c1)*(-E45*b1/2 + E55*a1/2)/(2*Ac) + (b1 - d1)*(-E44*b1/2 + E45*a1/2)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i1*dof+3
        kCsv[c] += -E44*b1/4 + E45*a1/4
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i1*dof+4
        kCsv[c] += -E45*b1/4 + E55*a1/4
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i2*dof+2
        kCsv[c] += -c1*(-E45*b1/2 + E55*a1/2)/(2*Ac) + d1*(-E44*b1/2 + E45*a1/2)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i2*dof+3
        kCsv[c] += -a1*c1*(-E45*b1/2 + E55*a1/2)/(4*Ac) + a1*d1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i2*dof+4
        kCsv[c] += -b1*c1*(-E45*b1/2 + E55*a1/2)/(4*Ac) + b1*d1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i3*dof+2
        kCsv[c] += a1*(-E45*b1/2 + E55*a1/2)/(2*Ac) - b1*(-E44*b1/2 + E45*a1/2)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i3*dof+3
        kCsv[c] += a1*c1*(-E45*b1/2 + E55*a1/2)/(4*Ac) - b1*c1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+2
        kCsc[c] = i3*dof+4
        kCsv[c] += a1*d1*(-E45*b1/2 + E55*a1/2)/(4*Ac) - b1*d1*(-E44*b1/2 + E45*a1/2)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i1*dof+2
        kCsv[c] += (-a1 + c1)*(-E45*b1*c1/4 + E55*a1*c1/4)/(2*Ac) + (b1 - d1)*(-E44*b1*c1/4 + E45*a1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i1*dof+3
        kCsv[c] += -E44*b1*c1/8 + E45*a1*c1/8
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i1*dof+4
        kCsv[c] += -E45*b1*c1/8 + E55*a1*c1/8
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i2*dof+2
        kCsv[c] += -c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(2*Ac) + d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i2*dof+3
        kCsv[c] += -a1*c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) + a1*d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i2*dof+4
        kCsv[c] += -b1*c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) + b1*d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i3*dof+2
        kCsv[c] += a1*(-E45*b1*c1/4 + E55*a1*c1/4)/(2*Ac) - b1*(-E44*b1*c1/4 + E45*a1*c1/4)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i3*dof+3
        kCsv[c] += a1*c1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) - b1*c1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+3
        kCsc[c] = i3*dof+4
        kCsv[c] += a1*d1*(-E45*b1*c1/4 + E55*a1*c1/4)/(4*Ac) - b1*d1*(-E44*b1*c1/4 + E45*a1*c1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i1*dof+2
        kCsv[c] += (-a1 + c1)*(-E45*b1*d1/4 + E55*a1*d1/4)/(2*Ac) + (b1 - d1)*(-E44*b1*d1/4 + E45*a1*d1/4)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i1*dof+3
        kCsv[c] += -E44*b1*d1/8 + E45*a1*d1/8
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i1*dof+4
        kCsv[c] += -E45*b1*d1/8 + E55*a1*d1/8
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i2*dof+2
        kCsv[c] += -c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(2*Ac) + d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i2*dof+3
        kCsv[c] += -a1*c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) + a1*d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i2*dof+4
        kCsv[c] += -b1*c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) + b1*d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i3*dof+2
        kCsv[c] += a1*(-E45*b1*d1/4 + E55*a1*d1/4)/(2*Ac) - b1*(-E44*b1*d1/4 + E45*a1*d1/4)/(2*Ac)
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i3*dof+3
        kCsv[c] += a1*c1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) - b1*c1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)
        c += 1
        kCsr[c] = i3*dof+4
        kCsc[c] = i3*dof+4
        kCsv[c] += a1*d1*(-E45*b1*d1/4 + E55*a1*d1/4)/(4*Ac) - b1*d1*(-E44*b1*d1/4 + E45*a1*d1/4)/(4*Ac)

    msg('finished!', silent=silent)

    kCs = coo_matrix((kCsv, (kCsr, kCsc)), shape=(size, size)).tocsr()

    return kCs
