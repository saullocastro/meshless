from __future__ import absolute_import, division

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from .logger import msg, warn
from .sparse import remove_null_cols


def lb(K, KG, tol=0, sparse_solver=True, silent=False,
       num_eigvalues=25, num_eigvalues_print=5):
    """Linear Buckling Analysis

    It can also be used for more general eigenvalue analyzes if `K` is the
    tangent stiffness matrix of a given load state.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix. Should include all constant terms of the initial
        stress stiffness matrix, aerodynamic matrix and so forth when
        applicable.
    KG : sparse_matrix
        Initial stress stiffness matrix that multiplies the load multiplcator
        `\lambda` of the eigenvalue problem.
    tol : float, optional
        A float tolerance passsed to the eigenvalue solver.
    sparse_solver : bool, optional
        Tells if solver :func:`scipy.linalg.eigh` or
        :func:`scipy.sparse.linalg.eigsh` should be used.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    num_eigvalues : int, optional
        Number of calculated eigenvalues.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.

    Notes
    -----
    The extracted eigenvalues are stored in the ``eigvals`` parameter
    of the ``Panel`` object and the `i^{th}` eigenvector in the
    ``eigvecs[:, i-1]`` parameter.

    """
    msg('Running linear buckling analysis...', silent=silent)

    msg('Eigenvalue solver... ', level=2, silent=silent)

    k = min(num_eigvalues, KG.shape[0]-2)
    if sparse_solver:
        mode = 'cayley'
        try:
            msg('eigsh() solver...', level=3, silent=silent)
            eigvals, eigvecs = eigsh(A=KG, k=k,
                    which='SM', M=K, tol=tol, sigma=1., mode=mode)
            msg('finished!', level=3, silent=silent)
        except Exception as e:
            warn(str(e), level=4, silent=silent)
            msg('aborted!', level=3, silent=silent)
            sizebkp = KG.shape[0]
            K, KG, used_cols = remove_null_cols(K, KG, silent=silent)
            msg('eigsh() solver...', level=3, silent=silent)
            eigvals, peigvecs = eigsh(A=KG, k=k,
                    which='SM', M=K, tol=tol, sigma=1., mode=mode)
            msg('finished!', level=3, silent=silent)
            eigvecs = np.zeros((sizebkp, num_eigvalues),
                               dtype=peigvecs.dtype)
            eigvecs[used_cols, :] = peigvecs

    else:
        size = KG.shape[0]
        K, KG, used_cols = remove_null_cols(K, KG, silent=silent)
        K = K.toarray()
        KG = KG.toarray()
        msg('eigh() solver...', level=3, silent=silent)
        eigvals, peigvecs = eigh(a=KG, b=K)
        msg('finished!', level=3, silent=silent)
        eigvecs = np.zeros((size, num_eigvalues), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs[:, :num_eigvalues]

    check = eigvals!=0
    eigvals[check] = -1./eigvals[check]

    msg('finished!', level=2, silent=silent)

    msg('first {0} eigenvalues:'.format(num_eigvalues_print), level=1,
        silent=silent)

    for eig in eigvals[:num_eigvalues_print]:
        msg('{0}'.format(eig), level=2, silent=silent)

    return eigvals, eigvecs
