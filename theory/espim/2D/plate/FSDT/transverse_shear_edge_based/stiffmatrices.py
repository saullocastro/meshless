import sympy
from sympy import Matrix

from meshless.dev.sympytools import mprint_as_dense

sympy.var('nx1, ny1')
sympy.var('nx2, ny2')
sympy.var('nx3, ny3')
sympy.var('nx4, ny4')
sympy.var('f11, f12, f13, f14')
sympy.var('f21, f22, f23, f24')
sympy.var('f31, f32, f33, f34')
sympy.var('f41, f42, f43, f44')
sympy.var('A11, A12, A16, A22, A26, A66')
sympy.var('B11, B12, B16, B22, B26, B66')
sympy.var('D11, D12, D16, D22, D26, D66')
sympy.var('E44, E45, E55')
sympy.var('le1, le2, le3, le4, Ac')

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

A = Matrix([[A11, A12, A16],
            [A12, A22, A26],
            [A16, A26, A66]])

B = Matrix([[B11, B12, B16],
            [B12, B22, B26],
            [B16, B26, B66]])

D = Matrix([[D11, D12, D16],
            [D12, D22, D26],
            [D16, D26, D66]])

E = Matrix([[E44, E45],
            [E45, E55]])


# membrane

Bm = 1/Ac * (
     le1*Matrix([nx1*su1,
                 ny1*sv1,
                 ny1*su1 + nx1*sv1])
   + le2*Matrix([nx2*su2,
                 ny2*sv2,
                 ny2*su2 + nx2*sv2])
   + le3*Matrix([nx3*su3,
                 ny3*sv3,
                 ny3*su3 + nx3*sv3])
   + le4*Matrix([nx4*su4,
                 ny4*sv4,
                 ny4*su4 + nx4*sv4])
   )

# bending

Bb = 1/Ac * (
     le1*Matrix([nx1*sphix1,
                 ny1*sphiy1,
                 ny1*sphix1 + nx1*sphiy1])
   + le2*Matrix([nx2*sphix2,
                 ny2*sphiy2,
                 ny2*sphix2 + nx2*sphiy2])
   + le3*Matrix([nx3*sphix3,
                 ny3*sphiy3,
                 ny3*sphix3 + nx3*sphiy3])
   + le4*Matrix([nx4*sphix4,
                 ny4*sphiy4,
                 ny4*sphix4 + nx4*sphiy4])
   )


K = Ac*(Bm.transpose() * A * Bm
      + Bm.transpose() * B * Bb
      + Bb.transpose() * B * Bm
      + Bb.transpose() * D * Bb)

mprint_as_dense(K, 'k0', dofpernode=5)

# transverse shear terms

sympy.var('a1, b1, c1, d1, Ac1')
sympy.var('a2, b2, c2, d2, Ac2')

# Tria1: mid1 -> node1 -> node2
# Tria2: node1 -> mid2 -> node2

         #mid 1
Tria1Mid1 = 1/(2*Ac1) * Matrix([
  [0, 0, b1-d1, Ac1,  0],
  [0, 0, c1-a1,  0, Ac1]])

         #node 1
Tria1N1 = 1/(2*Ac1) * Matrix([
  [0, 0,  d1,  a1*d1/2,  b1*d1/2],
  [0, 0, -c1, -a1*c1/2, -b1*c1/2]])

         #node 2
Tria1N2 = 1/(2*Ac1) * Matrix([
  [0, 0, -b1, -b1*c1/2, -b1*d1/2],
  [0, 0,  a1,  a1*c1/2,  a1*d1/2]])


         #node 1
Tria2N1 = 1/(2*Ac2) * Matrix([
  [0, 0, b2-d2, Ac2,  0],
  [0, 0, c2-a2,  0, Ac2]])

         #mid 2
Tria2Mid2 = 1/(2*Ac2) * Matrix([
  [0, 0,  d2,  a2*d2/2,  b2*d2/2],
  [0, 0, -c2, -a2*c2/2, -b2*c2/2]])

         #node 2
Tria2N2 = 1/(2*Ac2) * Matrix([
  [0, 0, -b2, -b2*c2/2, -b2*d2/2],
  [0, 0,  a2,  a2*c2/2,  a2*d2/2]])


ZERO = Tria1Mid1*0

                       #node 1               ,            node 2          ,    other 1     ,      other 2
BsTria1 = Matrix([Tria1N1.T + 1/3*Tria1Mid1.T, Tria1N2.T + 1/3*Tria1Mid1.T, 1/3*Tria1Mid1.T,      ZERO.T    ]).T
BsTria2 = Matrix([Tria2N1.T + 1/3*Tria2Mid2.T, Tria2N2.T + 1/3*Tria2Mid2.T,    ZERO.T      , 1/3*Tria2Mid2.T]).T

Bs = 1/Ac*(Ac1*BsTria1 + Ac2*BsTria2)

K = Ac*Bs.transpose()*E*Bs
mprint_as_dense(K, 'k0s_interior_edge', dofpernode=5)


         #mid 1
Tria1Mid1 = 1/(2*Ac) * Matrix([
  [0, 0, b1-d1, Ac,  0],
  [0, 0, c1-a1,  0, Ac]])

         #node 1
Tria1N1 = 1/(2*Ac) * Matrix([
  [0, 0,  d1,  a1*d1/2,  b1*d1/2],
  [0, 0, -c1, -a1*c1/2, -b1*c1/2]])

         #node 2
Tria1N2 = 1/(2*Ac) * Matrix([
  [0, 0, -b1, -b1*c1/2, -b1*d1/2],
  [0, 0,  a1,  a1*c1/2,  a1*d1/2]])

                       #node 1               ,             node 2         ,    other 1
BsTria1 = Matrix([Tria1N1.T + 1/3*Tria1Mid1.T, Tria1N2.T + 1/3*Tria1Mid1.T, 1/3*Tria1Mid1.T]).T

Bs = BsTria1

K = Ac*Bs.transpose()*E*Bs
mprint_as_dense(K, 'k0s_boundary_edge', dofpernode=5)
