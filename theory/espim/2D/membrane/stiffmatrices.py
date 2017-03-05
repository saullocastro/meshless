import sympy
from sympy import Matrix

from pim.sympytools import print_as_sparse, print_as_array, print_as_full

sympy.var('u, v, w, phix, phiy')
sympy.var('nx1, ny1')
sympy.var('nx2, ny2')
sympy.var('nx3, ny3')
sympy.var('nx4, ny4')
sympy.var('f11, f12, f13, f14')
sympy.var('f21, f22, f23, f24')
sympy.var('f31, f32, f33, f34')
sympy.var('f41, f42, f43, f44')
sympy.var('A11, A12, A16, A22, A26, A66')
sympy.var('le1, le2, le3, le4, Ac')

# approximation for linear interpolation within a tria
# u = u1*f1 + u2*f2 + u3*f3
# v = v1*f1 + v2*f2 + v3*f3
# w = w1*f1 + w2*f2 + w3*f3
# phix = phix1*f1 + phix2*f2 + phix3*f3
# phiy = phiy1*f1 + phiy2*f2 + phiy3*f3

# in matrix form
# u =    [f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0, 0, 0, 0] * [u1, v1, w1, phix1, phiy1, u2, v2, ... , u5, v5, w5, phix5, phiy5]
# v =    [0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0, 0, 0] *   ||
# w =    [0, 0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0, 0] *   ||
# phix = [0, 0, 0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0] *   ||
# phiy = [0, 0, 0, 0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3] *   ||

# u = f1 + x*f2 + y+f3
# u = [1, x, y].T*[f1, f2, f3]
# a = (inv(P) * unodes)
# u = [1, x, y] * inv(P) * unodes

# s = 0
# sympy.var('x1, y1, x2, y2, x3, y3')
# sympy.var('p11, p12, p13, p21, p22, p23, p31, p32, p33')
# P = Matrix([[1, x1, y1],
            # [1, x2, y2],
            # [1, x3, y3]])
# Pinv = Matrix([[p11, p12, p13],
               # [p21, p22, p23],
               # [p31, p32, p33]])
# s = [-1, +1]
# s = -1 : x = xa, y = xa
# s = +1 : x = xb, y = xb
# sympy.var('x, y')
# sympy.var('xa, ya, xb, yb')
# x = (xa*(1-s) + xb*(s+1))/2
# y = (ya*(1-s) + yb*(s+1))/2
# f1, f2, f3 = Matrix([[1, x, y]]) * Pinv

su1 =    Matrix([[f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0, 0, 0, f14, 0, 0, 0, 0]])
sv1 =    Matrix([[0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0, 0, 0, f14, 0, 0, 0]])
sw1 =    Matrix([[0, 0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0, 0, 0, f14, 0, 0]])
sphix1 = Matrix([[0, 0, 0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0, 0, 0, f14, 0]])
sphiy1 = Matrix([[0, 0, 0, 0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0, 0, 0, f14]])

su2 =    Matrix([[f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0, 0, 0, f24, 0, 0, 0, 0]])
sv2 =    Matrix([[0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0, 0, 0, f24, 0, 0, 0]])
sw2 =    Matrix([[0, 0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0, 0, 0, f24, 0, 0]])
sphix2 = Matrix([[0, 0, 0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0, 0, 0, f24, 0]])
sphiy2 = Matrix([[0, 0, 0, 0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0, 0, 0, f24]])

su3 =    Matrix([[f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0, 0, 0, f34, 0, 0, 0, 0]])
sv3 =    Matrix([[0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0, 0, 0, f34, 0, 0, 0]])
sw3 =    Matrix([[0, 0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0, 0, 0, f34, 0, 0]])
sphix3 = Matrix([[0, 0, 0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0, 0, 0, f34, 0]])
sphiy3 = Matrix([[0, 0, 0, 0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0, 0, 0, f34]])

su4 =    Matrix([[f41, 0, 0, 0, 0, f42, 0, 0, 0, 0, f43, 0, 0, 0, 0, f44, 0, 0, 0, 0]])
sv4 =    Matrix([[0, f41, 0, 0, 0, 0, f42, 0, 0, 0, 0, f43, 0, 0, 0, 0, f44, 0, 0, 0]])
sw4 =    Matrix([[0, 0, f41, 0, 0, 0, 0, f42, 0, 0, 0, 0, f43, 0, 0, 0, 0, f44, 0, 0]])
sphix4 = Matrix([[0, 0, 0, f41, 0, 0, 0, 0, f42, 0, 0, 0, 0, f43, 0, 0, 0, 0, f44, 0]])
sphiy4 = Matrix([[0, 0, 0, 0, f41, 0, 0, 0, 0, f42, 0, 0, 0, 0, f43, 0, 0, 0, 0, f44]])

A = Matrix([
    [A11, A12, A16, 0, 0, 0, 0, 0],
    [A12, A22, A26, 0, 0, 0, 0, 0],
    [A16, A26, A66, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    ])

# strains
# exx = u,x              # membrane
# eyy = v,y              # membrane
# gxy = u,y + v,x        # membrane
# gxz = w,x + phix       # membrane transverse shear
# gyz = w,y + phiy       # membrane transverse shear
# kxx = phix,x           # bending
# kyy = phiy,y           # bending
# kxy = phix,y + phiy,x  # bending

# for strain smoothing it will be, after applying divergence theorem
# cexx = u              # membrane
# ceyy = v              # membrane
# cgxy = u + v          # membrane
# ckxx = phix           # bending
# ckyy = phiy           # bending
# ckxy = phix + phiy    # bending

# transverse shear treated differently, by discrete shear gap (DSG)
# cgxz = w,x + phix       # membrane transverse shear
# cgyz = w,y + phiy       # membrane transverse shear

# exx eyy gxy kxx kyy kxy gxz gyz (8 strain components)
# dof = 5 for FSDT (u, v, w, phix, phiy)
# Bm, Bb matrices are 8 strain components x (N x dof)

# MATRIX FORM - membrane

ZERO = Matrix([[0]*su1.shape[1]])
Bm = 1/Ac * (
     le1*Matrix([nx1*su1,
             ny1*sv1,
             ny1*su1 + nx1*sv1,
             ZERO,
             ZERO,
             ZERO,
             ZERO,
             ZERO])
   + le2*Matrix([nx2*su2,
             ny2*sv2,
             ny2*su2 + nx2*sv2,
             ZERO,
             ZERO,
             ZERO,
             ZERO,
             ZERO])
   + le3*Matrix([nx3*su3,
             ny3*sv3,
             ny3*su3 + nx3*sv3,
             ZERO,
             ZERO,
             ZERO,
             ZERO,
             ZERO])
   + le4*Matrix([nx4*su4,
             ny4*sv4,
             ny4*su4 + nx4*sv4,
             ZERO,
             ZERO,
             ZERO,
             ZERO,
             ZERO])
   )

K = Ac*Bm.transpose() * A * Bm

print_as_full(sympy.simplify(K), 'k0', dofpernode=5)


