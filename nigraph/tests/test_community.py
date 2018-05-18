from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy.testing as npt
import nigraph as nig


def test_louvain():
    A = nig.get_random_graph(30, directed=False)
    n2c, extras = nig.louvain(A, weighted=False, return_tree=False)
    npt.assert_equal(A.shape[0], n2c.shape[0])
    n2c, extras = nig.louvain(A, weighted=False, return_tree=True)
    npt.assert_equal(A.shape[0], n2c.shape[1])


def test_label_propagation():
    A = nig.get_random_graph(30, directed=False)
    n2c, extras = nig.label_propagation(A)
    npt.assert_equal(A.shape[0], n2c.shape[0])


def test_k_shell():
    A = nig.get_random_graph(30, directed=False)
    n2c, extras = nig.n2c, _ = nig.k_shell(A, only_core_periphery=False)
    npt.assert_equal(A.shape[0], n2c.shape[0])
    n2c, extras = nig.n2c, _ = nig.k_shell(A, only_core_periphery=True)
    npt.assert_equal(A.shape[0], n2c.shape[0])
