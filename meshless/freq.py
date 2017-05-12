from __future__ import absolute_import, division

import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

from .logger import msg
from .sparse import remove_null_cols


def freq(K, M, tol=0, sparse_solver=True, silent=False,
         sort=True, reduced_dof=False,
         num_eigvalues=25, num_eigvalues_print=5):
    """Frequency Analysis

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix. Should include initial stress stiffness matrix,
        aerodynamic matrix and so forth when applicable.
    M : sparse_matrix
        Mass matrix.
    tol : float, optional
        A tolerance value passed to ``scipy.sparse.linalg.eigs``.
    sparse_solver : bool, optional
        Tells if solver :func:`scipy.linalg.eig` or
        :func:`scipy.sparse.linalg.eigs` should be used.

        .. note:: It is recommended ``sparse_solver=False``, because it
                  was verified that the sparse solver becomes unstable
                  for some cases, though the sparse solver is faster.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    sort : bool, optional
        Sort the output eigenvalues and eigenmodes.
    reduced_dof : bool, optional
        Considers only the contributions of `v` and `w` to the stiffness
        matrix and accelerates the run. Only effective when
        ``sparse_solver=False``.
    num_eigvalues : int, optional
        Number of calculated eigenvalues.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.

    Returns
    -------
    The extracted eigenvalues are stored in the ``eigvals`` parameter and
    the `i^{th}` eigenvector in the ``eigvecs[:, i-1]`` parameter.

    """
    msg('Running frequency analysis...', silent=silent)

    msg('Eigenvalue solver... ', level=2, silent=silent)

    k = min(num_eigvalues, M.shape[0]-2)
    if sparse_solver:
        msg('eigs() solver...', level=3, silent=silent)
        sizebkp = M.shape[0]
        K, M, used_cols = remove_null_cols(K, M, silent=silent,
                level=3)
        #NOTE Looking for better performance with symmetric matrices, I tried
        #     using meshless.sparse.is_symmetric and eigsh, but it seems not
        #     to improve speed (I did not try passing only half of the sparse
        #     matrices to the solver)
        eigvals, peigvecs = eigs(A=K, k=k, which='LM', M=M, tol=tol,
                                 sigma=-1.)
        eigvecs = np.zeros((sizebkp, num_eigvalues), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs

        eigvals = np.sqrt(eigvals) # omega^2 to omega, in rad/s

    else:
        msg('eig() solver...', level=3, silent=silent)
        M = M.toarray()
        K = K.toarray()
        sizebkp = M.shape[0]
        col_sum = M.sum(axis=0)
        check = col_sum != 0
        used_cols = np.arange(M.shape[0])[check]
        M = M[:, check][check, :]
        K = K[:, check][check, :]

        if reduced_dof:
            i = np.arange(M.shape[0])
            take = np.column_stack((i[1::3], i[2::3])).flatten()
            M = M[:, take][take, :]
            K = K[:, take][take, :]
        #TODO did not try using eigh when input is symmetric to see if there
        #     will be speed improvements
        eigvals, peigvecs = eig(a=-M, b=K)
        eigvecs = np.zeros((sizebkp, K.shape[0]),
                           dtype=peigvecs.dtype)
        eigvecs[check, :] = peigvecs
        eigvals = np.sqrt(-1./eigvals) # -1/omega^2 to omega, in rad/s
        eigvals = eigvals

    msg('finished!', level=3, silent=silent)

    if sort:
        sort_ind = np.lexsort((np.round(eigvals.imag, 1),
                               np.round(eigvals.real, 1)))
        eigvals = eigvals[sort_ind]
        eigvecs = eigvecs[:, sort_ind]

        higher_zero = eigvals.real > 1e-6

        eigvals = eigvals[higher_zero]
        eigvecs = eigvecs[:, higher_zero]

    if not sparse_solver and reduced_dof:
        new_eigvecs = np.zeros((3*eigvecs.shape[0]//2, eigvecs.shape[1]),
                dtype=eigvecs.dtype)
        new_eigvecs[take, :] = eigvecs
        eigvecs = new_eigvecs


    msg('finished!', level=2, silent=silent)

    msg('first {0} eigenvalues:'.format(num_eigvalues_print), level=1,
        silent=silent)
    for eigval in eigvals[:num_eigvalues_print]:
        msg('{0} rad/s'.format(eigval), level=2, silent=silent)

    return eigvals, eigvecs
