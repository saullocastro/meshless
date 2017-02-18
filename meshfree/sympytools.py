import re

import numpy as np
import sympy
from sympy import collect


def pow2mult(instr):
    """Power to multiplications

    Substitutes x**5 or pow(x, 5) by x*x*x*x*x

    Parameters
    ----------
    instr : str
        The input string

    Returns
    -------
    outstr : str
        The output string

    """
    old_new = []
    for p in re.findall(r'\w+\*\*\d+', instr):
        var, exp = p.split('**')
        new = '(' + '*'.join([var]*int(exp)) + ')'
        old_new.append([p, new])
    # putting longer patterns first to avoid wrong
    # substitutions
    old_new = sorted(old_new, key=lambda x: len(x[0]))[::-1]
    outstr = instr
    for old, new in old_new:
        outstr = outstr.replace(old, new)

    old_new = []
    for p in re.findall(r'pow\(\w+,\s*\w+\)', instr):
        var, exp = p.split('pow')[1].split('(')[1].split(')')[0].split(',')
        new = '(' + '*'.join([var]*int(exp)) + ')'
        old_new.append([p, new])
    for old, new in old_new:
        outstr = outstr.replace(old, new)

    return outstr


def print_as_sparse(m, mname, sufix=None, subs=None, header=None,
        print_file=True, collect_for=None, pow_by_mul=True,
        full_symmetric=False):
    if sufix is None:
        left = right = '1'
        namesufix = '{0}'.format(mname)
    else:
        left, right = sufix
        namesufix = '{0}_{1}'.format(mname, sufix)
    filename = 'print_{0}.txt'.format(namesufix)
    ls = []
    if header:
        ls.append(header)
    ls.append('# {0}'.format(namesufix))

    if full_symmetric:
        num = len([None for (i, j), v in np.ndenumerate(m) if (v and i <= j)])
    else:
        num = len([i for i in list(m) if i])

    ls.append('# {0}_num={1}'.format(namesufix, num))

    for (i, j), v in np.ndenumerate(m):
        if v:
            if full_symmetric and i > j:
                continue

            if subs:
                v = v.subs(subs)
            ls.append('c += 1')

            if left == '0' or full_symmetric:
                ls.append('{mname}r[c] = {i}'.format(mname=mname, i=i))
            else:
                ls.append('{mname}r[c] = row+{i}'.format(mname=mname, i=i))

            if right == '0' or full_symmetric:
                ls.append('{mname}c[c] = {j}'.format(mname=mname, j=j))
            else:
                ls.append('{mname}c[c] = col+{j}'.format(mname=mname, j=j))

            if collect_for is not None:
                v = collect(v, collect_for, evaluate=False)
                ls.append('{mname}v[c] +='.format(mname=mname))
                for k, expr in v.items():
                    ls.append('#   collected for {k}'.format(k=k))
                    ls.append('    {expr}'.format(expr=k*expr))
            else:
                if pow_by_mul:
                    v = pow2mult(str(v))
                ls.append('{mname}v[c] += {v}'.format(mname=mname, v=v))

    string = '\n'.join(ls)

    if subs:
        items = sorted(subs.items(), key= lambda x: str(x[1]))
        items = [(k, v) for k, v in items if str(v) in string]
        if items:
            ls_header = []
            ls_header.append('subs\n')
            for k, v in items:
                ls_header.append('{0} = {1}'.format(v, k))
            ls_header.append('\n')

            string = '\n'.join(ls_header + ls)

    if print_file:
        with open(filename, 'w') as f:
            f.write(string)
    return string


def print_as_full(m, mname, subs=None, header=None, print_file=True,
        collect_for=None, pow_by_mul=True, dofpernode=None):
    namesufix = '{0}'.format(mname)
    filename = 'print_{0}.txt'.format(namesufix)
    ls = []
    if header:
        ls.append(header)
    ls.append('# {0}'.format(namesufix))

    num = len([None for (i, j), v in np.ndenumerate(m) if v])

    ls.append('# {0}_num={1}'.format(namesufix, num))

    for (i, j), v in np.ndenumerate(m):
        if dofpernode:
            nindi = (i // dofpernode) + 1
            nindj = (j // dofpernode) + 1
            i = i % dofpernode
            j = j % dofpernode
        if v:
            if subs:
                v = v.subs(subs)
            if collect_for is not None:
                v = collect(v, collect_for, evaluate=False)
                if dofpernode:
                    ls.append('{mname}[i{nindi}*dof+{i}, i{nindj}*dof+{j}] +='.format(
                        mname=mname, i=i, j=j, nindi=nindi, nindj=nindj))
                else:
                    ls.append('{mname}[{i}, {j}] +='.format(mname=mname, i=i, j=j))
                for k, expr in v.items():
                    ls.append('#   collected for {k}'.format(k=k))
                    ls.append('    {expr}'.format(expr=k*expr))
            else:
                if pow_by_mul:
                    v = pow2mult(str(v))
                if dofpernode:
                    ls.append('{mname}[i{nindi}*dof+{i}, i{nindj}*dof+{j}] += {v}'.format(
                        mname=mname, i=i, j=j, v=v, nindi=nindi, nindj=nindj))
                else:
                    ls.append('{mname}[{i}, {j}] += {v}'.format(mname=mname,
                        i=i, j=j, v=v))

    string = '\n'.join(ls)

    if subs:
        items = sorted(subs.items(), key= lambda x: str(x[1]))
        items = [(k, v) for k, v in items if str(v) in string]
        if items:
            ls_header = []
            ls_header.append('subs\n')
            for k, v in items:
                ls_header.append('{0} = {1}'.format(v, k))
            ls_header.append('\n')

            string = '\n'.join(ls_header + ls)

    if print_file:
        with open(filename, 'w') as f:
            f.write(string)
    return string


def print_as_array(m, mname, sufix=None, use_cse=False, header=None,
        print_file=True, collect_for=None, pow_by_mul=True, order='C',
        op='+='):
    ls = []
    if use_cse:
        subs, m_list = sympy.cse(m)
        for i, v in enumerate(m_list):
            m[i] = v
    if sufix is None:
        namesufix = '{0}'.format(mname)
    else:
        namesufix = '{0}_{1}'.format(mname, sufix)
    filename = 'print_{0}.txt'.format(namesufix)
    if header:
        ls.append(header)
    if use_cse:
        ls.append('# cdefs')
        num = 9
        for i, sub in enumerate(subs[::num]):
            ls.append('cdef double ' + ', '.join(
                        map(str, [j[0] for j in subs[num*i:num*(i+1)]])))
        ls.append('# subs')
        for sub in subs:
            ls.append('{0} = {1}'.format(*sub))
    ls.append('# {0}'.format(namesufix))
    num = len([i for i in list(m) if i])
    ls.append('# {0}_num={1}'.format(namesufix, num))
    if order == 'C':
        miter = enumerate(np.ravel(m))
    elif order == 'F':
        miter = enumerate(np.ravel(m.T))
    miter = list(miter)
    for i, v in miter:
        if v:
            if collect_for is not None:
                v = collect(v, collect_for, evaluate=False)
                ls.append('{0}[pos+{1}] +='.format(mname, i))
                for k, expr in v.items():
                    ls.append('#   collected for {k}'.format(k=k))
                    ls.append('    {expr}'.format(expr=k*expr))
            else:
                if pow_by_mul:
                    v = str(v)
                    for p in re.findall(r'\w+\*\*\d+', v):
                        var, exp = p.split('**')
                        v = v.replace(p, '(' + '*'.join([var]*int(exp)) + ')')
                ls.append('{0}[pos+{1}] {2} {3}'.format(mname, i, op, v))
    string = '\n'.join(ls)
    if print_file:
        with open(filename, 'w') as f:
            f.write(string)
    return string
