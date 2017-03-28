# Implementation as per phungvan2013

import sympy
from sympy import Matrix

from pim.sympytools import print_as_sparse, print_as_array, print_as_full

sympy.var('u, v, w, phix, phiy')
sympy.var('f11, f12, f13')
sympy.var('f21, f22, f23')
sympy.var('f31, f32, f33')
sympy.var('A11, A12, A16, A22, A26, A66')
sympy.var('B11, B12, B16, B22, B26, B66')
sympy.var('D11, D12, D16, D22, D26, D66')
sympy.var('E44, E45, E55')
sympy.var('Ac, Ac1, Ac2, Ac3, Ac4')

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

su1 =    Matrix([[f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0, 0, 0]])
sv1 =    Matrix([[0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0, 0]])
sw1 =    Matrix([[0, 0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0, 0]])
sphix1 = Matrix([[0, 0, 0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13, 0]])
sphiy1 = Matrix([[0, 0, 0, 0, f11, 0, 0, 0, 0, f12, 0, 0, 0, 0, f13]])

su2 =    Matrix([[f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0, 0, 0]])
sv2 =    Matrix([[0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0, 0]])
sw2 =    Matrix([[0, 0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0, 0]])
sphix2 = Matrix([[0, 0, 0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23, 0]])
sphiy2 = Matrix([[0, 0, 0, 0, f21, 0, 0, 0, 0, f22, 0, 0, 0, 0, f23]])

su3 =    Matrix([[f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0, 0, 0]])
sv3 =    Matrix([[0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0, 0]])
sw3 =    Matrix([[0, 0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0, 0]])
sphix3 = Matrix([[0, 0, 0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33, 0]])
sphiy3 = Matrix([[0, 0, 0, 0, f31, 0, 0, 0, 0, f32, 0, 0, 0, 0, f33]])


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

sympy.var('a1, b1, d1, c1, Ac1')
sympy.var('a2, b2, d2, c2, Ac2')
sympy.var('a3, b3, d3, c3, Ac3')

# membrane terms

Bm1 = 1/(2*Ac1)*(Matrix([
         #node1            node 2                node 3
  [0, 0, 0, 0, 0,    c1,   0, 0, 0, 0,    -b1,   0, 0, 0, 0],
  [0, 0, 0, 0, 0,     0, -d1, 0, 0, 0,      0,  a1, 0, 0, 0],
  [0, 0, 0, 0, 0,   -d1,  c1, 0, 0, 0,     a1, -b1, 0, 0, 0]])

  +

  1/3*Matrix([[b1-c1,     0, 0, 0, 0,    b1-c1,     0, 0, 0, 0,    b1-c1,     0, 0, 0, 0],
              [    0, d1-a1, 0, 0, 0,        0, d1-a1, 0, 0, 0,        0, d1-a1, 0, 0, 0],
              [d1-a1, b1-c1, 0, 0, 0,    d1-a1, b1-c1, 0, 0, 0,    d1-a1, b1-c1, 0, 0, 0]])
      )

Bm2 = 1/(2*Ac2)*(Matrix([
         #node1                node 2                node 3
  [b2-c2,     0, 0, 0, 0,   0, 0, 0, 0, 0,    -b2,   0, 0, 0, 0],
  [    0, d2-a2, 0, 0, 0,   0, 0, 0, 0, 0,      0,  a2, 0, 0, 0],
  [d2-a2, b2-c2, 0, 0, 0,   0, 0, 0, 0, 0,     a2, -b2, 0, 0, 0]])

  +

  1/3*Matrix([[ c2,   0, 0, 0, 0,     c2,   0, 0, 0, 0,     c2,   0, 0, 0, 0],
              [  0, -d2, 0, 0, 0,      0, -d2, 0, 0, 0,      0, -d2, 0, 0, 0],
              [-d2,  c2, 0, 0, 0,    -d2,  c2, 0, 0, 0,    -d2,  c2, 0, 0, 0]])
      )

Bm3 = 1/(2*Ac3)*(Matrix([
         #node1                   node 2              node 3
  [b3-c3,     0, 0, 0, 0,    c3,   0, 0, 0, 0,    0, 0, 0, 0, 0],
  [    0, d3-a3, 0, 0, 0,     0, -d3, 0, 0, 0,    0, 0, 0, 0, 0],
  [d3-a3, b3-c3, 0, 0, 0,   -d3,  c3, 0, 0, 0,    0, 0, 0, 0, 0]])

  +

  1/3*Matrix([[-b3,   0, 0, 0, 0,    -b3,   0, 0, 0, 0,    -b3,   0, 0, 0, 0],
              [  0,  a3, 0, 0, 0,      0,  a3, 0, 0, 0,      0,  a3, 0, 0, 0],
              [ a3, -b3, 0, 0, 0,     a3, -b3, 0, 0, 0,     a3, -b3, 0, 0, 0]])
      )


Bm = 1/Ac*(Ac1*Bm1 + Ac2*Bm2 + Ac3*Bm3)


# bending terms

Bb1 = 1/(2*Ac1)*(Matrix([
         #node1          node 2                node 3
  [0, 0, 0, 0, 0,    0, 0, 0,  c1, -d1,      0, 0, 0, -b1,   0],
  [0, 0, 0, 0, 0,    0, 0, 0,   0,   0,      0, 0, 0,   0,  a1],
  [0, 0, 0, 0, 0,    0, 0, 0, -d1,  c1,      0, 0, 0,  a1, -b1]])

  +

  1/3*Matrix([[0, 0, 0, b1-c1,     0,    0, 0, 0, b1-c1,     0,    0, 0, 0, b1-c1,     0],
              [0, 0, 0,     0, d1-a1,    0, 0, 0,     0, d1-a1,    0, 0, 0,     0, d1-a1],
              [0, 0, 0, d1-a1, b1-c1,    0, 0, 0, d1-a1, b1-c1,    0, 0, 0, d1-a1, b1-c1]])
      )

Bb2 = 1/(2*Ac2)*(Matrix([
         #node1                  node 2                node 3
  [0, 0, 0, b2-c2,     0,    0, 0, 0, 0, 0,      0, 0, 0, -b2,   0],
  [0, 0, 0,     0, d2-a2,    0, 0, 0, 0, 0,      0, 0, 0,   0,  a2],
  [0, 0, 0, d2-a2, b2-c2,    0, 0, 0, 0, 0,      0, 0, 0,  a2, -b2]])

  +

  1/3*Matrix([[0, 0, 0,  c2, -d2,    0, 0, 0,  c2, -d2,    0, 0, 0,  c2, -d2],
              [0, 0, 0,   0,   0,    0, 0, 0,   0,   0,    0, 0, 0,   0,   0],
              [0, 0, 0, -d2,  c2,    0, 0, 0, -d2,  c2,    0, 0, 0, -d2,  c2]])
      )

Bb3 = 1/(2*Ac3)*(Matrix([
         #node1                  node 2                node 3
  [0, 0, 0, b3-c3,     0,    0, 0, 0,  c3, -d3,      0, 0, 0, 0, 0],
  [0, 0, 0,     0, d3-a3,    0, 0, 0,   0,   0,      0, 0, 0, 0, 0],
  [0, 0, 0, d3-a3, b3-c3,    0, 0, 0, -d3,  c3,      0, 0, 0, 0, 0]])

  +

  1/3*Matrix([[0, 0, 0, -b3,   0,    0, 0, 0, -b3,   0,    0, 0, 0, -b3,   0],
              [0, 0, 0,   0,  a3,    0, 0, 0,   0,  a3,    0, 0, 0,   0,  a3],
              [0, 0, 0,  a3, -b3,    0, 0, 0,  a3, -b3,    0, 0, 0,  a3, -b3]])
      )


Bb = 1/Ac*(Ac1*Bb1 + Ac2*Bb2 + Ac3*Bb3)



# transverse shear terms

Bs1 = 1/(2*Ac1)*(Matrix([
         #node1                       node 2                      node 3
  [0, 0, 0, 0, 0,   0, 0,  c1,  a1*c1/2,  b1*c1/2,    0, 0, -b1, -b1*d1/2, -b1*c1/2   ],
  [0, 0, 0, 0, 0,   0, 0, -d1, -a1*d1/2, -b1*d1/2,    0, 0,  a1,  a1*d1/2,  a1*c1/2   ]])

  +

  1/3*Matrix([[0, 0, b1-c1, Ac1,  0,    0, 0, b1-c1, Ac1,  0,    0, 0, b1-c1, Ac1,  0],
              [0, 0, d1-a1,  0, Ac1,    0, 0, d1-a1,  0, Ac1,    0, 0, d1-a1,  0, Ac1]])
      )

Bs2 = 1/(2*Ac2)*(Matrix([
         #node1                node 2                    node 3
  [0, 0, b2-c2, Ac2,  0,   0, 0, 0, 0, 0,    0, 0, -b2, -b2*d2/2, -b2*c2/2   ],
  [0, 0, d2-a2,  0, Ac2,   0, 0, 0, 0, 0,    0, 0,  a2,  a2*d2/2,  a2*c2/2   ]])

  +

  1/3*Matrix([[0, 0,  c2,  a2*c2/2,  b2*c2/2,    0, 0,  c2,  a2*c2/2,  b2*c2/2,    0, 0,  c2,  a2*c2/2,  b2*c2/2],
              [0, 0, -d2, -a2*d2/2, -b2*d2/2,    0, 0, -d2, -a2*d2/2, -b2*d2/2,    0, 0, -d2, -a2*d2/2, -b2*d2/2]])
      )

Bs3 = 1/(2*Ac3)*(Matrix([
         #node1                         node 2                      node 3
  [0, 0, b3-c3, Ac3,  0,       0, 0,  c3,  a3*c3/2,  b3*c3/2,    0, 0, 0, 0, 0  ],
  [0, 0, d3-a3,  0, Ac3,       0, 0, -d3, -a3*d3/2, -b3*d3/2,    0, 0, 0, 0, 0  ]])

  +

  1/3*Matrix([[0, 0, -b3, -b3*d3/2, -b3*c3/2,    0, 0, -b3, -b3*d3/2, -b3*c3/2,    0, 0, -b3, -b3*d3/2, -b3*c3/2],
              [0, 0,  a3,  a3*d3/2,  a3*c3/2,    0, 0,  a3,  a3*d3/2,  a3*c3/2,    0, 0,  a3,  a3*d3/2,  a3*c3/2]])
      )


Bs = 1/Ac*(Ac1*Bs1 + Ac2*Bs2 + Ac3*Bs3)

K = Ac*(  Bm.transpose()*A*Bm
        + Bm.transpose()*B*Bb
        + Bb.transpose()*B*Bm
        + Bb.transpose()*D*Bb
        + Bs.transpose()*E*Bs
        )

print_as_full(K, 'k0', dofpernode=5)









