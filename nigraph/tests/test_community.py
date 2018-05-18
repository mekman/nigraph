from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy.testing as npt
import nigraph as nig


def test_louvain():
    A = nig.A = nig.get_random_graph(30, directed=False)
    n2c, extras = nig.louvain(A, weighted=False, return_tree=True)
    npt.assert_equal(A.shape[0], n2c.shape[0])
