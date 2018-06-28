import sympy

from meshless.dev.sympytools import (pow2mul, mprint_as_sparse, mprint_as_dense,
        mprint_as_array)

def test_expandpow():
    s = 'a** 3 + pow((b+2)*c+1, 2) + (a+b * c)**3'
    assert pow2mul(s) == '(a*a*a) + (((b+2)*c+1)*((b+2)*c+1)) + ((a+b*c)*(a+b*c)*(a+b*c))'

def test_print_as_sparse():
    xvar = sympy.var('xvar')
    yvar = sympy.var('yvar')
    m = sympy.Matrix([[1, 2*xvar + yvar**3]])

    res1 = """
# test_m
# test_m_num=2
c += 1
test_mr[c] = row+0
test_mc[c] = col+0
test_mv[c] += 1
c += 1
test_mr[c] = row+0
test_mc[c] = col+1
test_mv[c] += 2*xvar + (yvar*yvar*yvar)
"""
    res1 = res1.strip()
    mprint_as_sparse(m, 'test_m', print_file=False) == res1
    mprint_as_sparse(m, 'test_m', print_file=False, full_symmetric=True) == res1

    res2 = """
# test_m
# test_m_num=2
c += 1
test_mr[c] = row+0
test_mc[c] = col+0
test_mv[c] += 1
c += 1
test_mr[c] = row+0
test_mc[c] = col+1
test_mv[c] += (yvar*yvar*yvar) + 2*yvar
"""
    res2 = res2.strip()
    mprint_as_sparse(m, 'test_m', print_file=False, subs={xvar: yvar}) == res2

def test_print_as_full():
    xvar = sympy.var('xvar')
    yvar = sympy.var('yvar')
    res1 = '''
# m
# m_num=2
m[0, 0] += 1
m[0, 1] += 2*xvar + (yvar*yvar*yvar)
'''
    res1 = res1.strip()

    m = sympy.Matrix([[1, 2*xvar + yvar**3]])
    assert mprint_as_dense(m, 'm') == res1

    res2 = '''
# subs
# yvar = xvar
# m
# m_num=2
m[0, 0] += 1
m[0, 1] += (yvar*yvar*yvar) + 2*yvar
'''
    res2 = res2.strip()
    assert mprint_as_dense(m, 'm', subs={xvar: yvar}) == res2

def test_print_as_array():
    xvar = sympy.var('xvar')
    yvar = sympy.var('yvar')
    m = sympy.Matrix([[1, 2*xvar + yvar**3, + 2*xvar*(yvar+1)]])
    res1 = '''
# m
# m_num=3
m[pos+0] += 1
m[pos+1] += 2*xvar + (yvar*yvar*yvar)
m[pos+2] += 2*xvar*(yvar + 1)
    '''
    res1 = res1.strip()
    assert mprint_as_array(m, 'm') == res1

    res2 = '''
# cdefs
cdef double x0
# subs
x0 = 2*xvar
# m
# m_num=3
m[pos+0] += 1
m[pos+1] += x0 + (yvar*yvar*yvar)
m[pos+2] += x0*(yvar + 1)
'''
    res2 = res2.strip()
    assert mprint_as_array(m, 'm', use_cse=True) == res2

if __name__ == '__main__':
    test_expandpow()
    test_print_as_sparse()
    test_print_as_full()
    test_print_as_array()
