from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy.testing as npt
import nigraph as nig


def test_degree():
    A = nig.get_random_graph(30, directed=False)
    k = nig.degree(A, weighted=False)
    npt.assert_equal(A.shape[0], k.shape[0])
