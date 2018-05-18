from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy.testing as npt
import nigraph as nig


def test_get_random_graph():
    A = nig.A = nig.get_random_graph(10, directed=True)
    npt.assert_equal(A.shape[0], 10)
