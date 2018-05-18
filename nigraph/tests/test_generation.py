from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy.testing as npt
import numpy as np
import nigraph as nig


def test_get_random_graph():
    A = nig.get_random_graph(10, directed=True)
    npt.assert_equal(A.shape[0], 10)


def test_adj_static():
    ts = np.random.normal(size=(3, 100))
    A = nig.adj_static(ts, measure='corr')
    npt.assert_equal(A.shape[0], 3)
