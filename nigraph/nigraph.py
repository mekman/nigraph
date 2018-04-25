#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Commonly used utility functions."""

# mainly backports from future numpy here

from __future__ import absolute_import, division, print_function


def thresholding_abs(A, thr, smaller=True, copy=True):
    """thresholding of the adjacency matrix with an absolute value.

    Parameters
    ----------
    A : ndarray, shape(n, n) **or** shape(n_tps, n, n)
        Adjacency matrix of the graph. The matrix must be weighted with zero
        values along the main diagonal.
    thr : float
        Absolute threshold. Edges with weights `</>` to the given threshold
        value will be removed.
    smaller : boolean (default=True)
        Threshold values smaller than the given threshold. If ``smaller=False``
        values greater
        than are thresholded.
    copy : boolean
        Whether to copy the input or change the matrix in-place (default=True).

    Returns
    -------
    A_thr : ndarray, shape(n, n) **or** shape(n_tps, n, n)
        Thresholded adjacency matrix.

    Notes
    -----
    Supports also thresholding of **dynamic graphs**.

    See also
    --------
    thresholding_rel, thresholding_pval, thresholding_M, thresholding_max

    Examples
    --------
    >>> d = get_fmri_data()
    >>> A = adj_static(d, pval=False) # p-values are not required for thresholding with an given value
    >>> A_thr = thresholding_abs(A, .3)
    >>> print A_thr[A_thr > 0].min() # smallest nonzero edge weight after thresholding
    0.30012873644
    """

    # this function was tested against BCT and gives the same
    # results as the matlab function: 'threshold_absolute.m'

    # TODO np.clip is faster than inplace operation:
    # A = np.random.normal(size=10000*10000).reshape((10000,10000))
    # In [30]: %timeit np.clip(A,0,np.inf, A)
    # 1 loops, best of 3: 315 ms per loop
    #
    # In [31]: %timeit A[A < 0]=0
    # 1 loops, best of 3: 638 ms per loop

    def _thr(data, smaller=True):
        if smaller:
            data[data < thr] = 0.
        else:
            data[data > thr] = 0.
        return data

    if copy:
        data = A.copy()
        return _thr(data)
    else:
        return _thr(A)
