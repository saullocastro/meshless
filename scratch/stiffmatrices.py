import sympy
from sympy import Matrix

from meshfree.sympytools import print_as_sparse, print_as_array, print_as_full

sympy.var('u, v, w, phix, phiy')
sympy.var('nx, ny')
sympy.var('f1, f2, f3')
sympy.var('A11, A12, A16, A22, A26, A66')
sympy.var('B11, B12, B16, B22, B26, B66')
sympy.var('D11, D12, D16, D22, D26, D66')
sympy.var('As')

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

su =    Matrix([[f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0, 0, 0, 0]])
sv =    Matrix([[0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0, 0, 0]])
sw =    Matrix([[0, 0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0, 0]])
sphix = Matrix([[0, 0, 0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3, 0]])
sphiy = Matrix([[0, 0, 0, 0, f1, 0, 0, 0, 0, f2, 0, 0, 0, 0, f3]])

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

B = Matrix([
    [0, 0, 0, B11, B12, B16, 0, 0],
    [0, 0, 0, B12, B22, B26, 0, 0],
    [0, 0, 0, B16, B26, B66, 0, 0],
    [B11, B12, B16, 0, 0, 0, 0, 0],
    [B12, B22, B26, 0, 0, 0, 0, 0],
    [B16, B26, B66, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    ])

D = Matrix([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, D11, D12, D16, 0, 0],
    [0, 0, 0, D12, D22, D26, 0, 0],
    [0, 0, 0, D16, D26, D66, 0, 0],
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

zero = Matrix([[0]*su.shape[1]])
Bm = Matrix([
      nx*su,
      ny*sv,
      ny*su + nx*sv,
      zero,
      zero,
      zero,
      zero,
      zero
     ])

# MATRIX FORM - bending
#      exx eyy gxy kxx kyy kxy gxz gyz
Bb = Matrix([
      zero,
      zero,
      zero,
      nx*sphix,
      ny*sphiy,
      -ny*sphix + nx*sphiy,
      zero,
      zero
     ])

K = 1/As*(Bm.transpose() * A * Bm
    + Bm.transpose() * B * Bb
    + Bb.transpose() * B * Bm
    + Bb.transpose() * D * Bb)

print_as_full(K, 'k0', dofpernode=5)


