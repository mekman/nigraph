from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy.testing as npt
import numpy as np
import nigraph as nig


def test_get_random_graph():
    A = nig.get_random_graph(10, weighted=False, directed=False)
    npt.assert_equal(A.shape[0], 10)
    A = nig.get_random_graph(10, weighted=True, directed=True)
    npt.assert_equal(A.shape[0], 10)
    A = nig.get_random_graph(10, weighted=True, directed=False, fmt='ig')
    npt.assert_equal(A.get_adjacency().shape[0], 10)


def test_adj_static():
    ts = np.random.normal(size=(3, 100))
    A = nig.adj_static(ts, measure='corr')
    npt.assert_equal(A.shape[0], 3)
    A = nig.adj_static(ts, measure='cov')
    npt.assert_equal(A.shape[0], 3)
    A = nig.adj_static(ts, measure='pcorr')
    npt.assert_equal(A.shape[0], 3)
    A = nig.adj_static(ts, measure='rho')
    npt.assert_equal(A.shape[0], 3)
    A = nig.adj_static(ts, measure='coh')
    npt.assert_equal(A.shape[0], 3)
