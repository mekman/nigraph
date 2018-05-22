from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy.testing as npt
import numpy as np
import nigraph as nig


def test_degree():
    A = nig.get_random_graph(30, directed=False)
    k = nig.degree(A, directed=False, ignore_self_loops=False)
    npt.assert_equal(A.shape[0], k.shape[0])


def test_betweenness_centrality():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.betweenness_centrality(A, weighted=False)
    npt.assert_equal(A.shape[0], m.shape[0])

    A = nig.get_random_graph(30, weighted=True)
    m = nig.betweenness_centrality(A, weighted=True)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_local_characteristic_path_length():
    # A = nig.get_random_graph(30, weighted=False)
    A = np.zeros((3, 3))
    A[:, 0] = 1
    m = nig.local_characteristic_path_length(A, weighted=False)
    npt.assert_equal(A.shape[0], m.shape[0])

    # A = nig.get_random_graph(30, weighted=True)
    m = nig.local_characteristic_path_length(A, weighted=True)
    npt.assert_equal(A.shape[0], m.shape[0])


# def test_avg_shortest_path_length():
#     A = nig.get_random_graph(30, weighted=False)
#     m = nig.avg_shortest_path_length(A, weighted=False)
#     npt.assert_equal(1, len(m))
#
#     A = nig.get_random_graph(30, weighted=True)
#     m = nig.avg_shortest_path_length(A, weighted=True)
#     npt.assert_equal(1, len(m))


def test_nodal_average_shortest_path_length():
    # A = nig.get_random_graph(30, weighted=False)
    A = np.zeros((3, 3))
    A[:, 0] = 1
    m = nig.nodal_average_shortest_path_length(A, weighted=False)
    npt.assert_equal(A.shape[0], m.shape[0])

    A = nig.get_random_graph(30, weighted=True)
    m = nig.nodal_average_shortest_path_length(A, weighted=True)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_within_module_degree_z_score():
    A = nig.get_random_graph(30, weighted=False)
    n2c, extras = nig.louvain(A, weighted=False, return_tree=False)
    m = nig.within_module_degree_z_score(A, partition=n2c)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_participation_coefficient():
    A = nig.get_random_graph(30, weighted=False)
    n2c, extras = nig.louvain(A, weighted=False, return_tree=False)
    m = nig.participation_coefficient(A, partition=n2c)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_absorption():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.absorption(A)
    npt.assert_equal(A.shape, m.shape)


def test_driftness():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.driftness(A)
    npt.assert_equal(A.shape, m.shape)


def test_number_k_max():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.number_k_max(A)
    npt.assert_equal(1, np.asarray(m).size)


def test_wiring_costs():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.wiring_costs(A)
    npt.assert_equal(1, np.asarray(m).size)


def test_efficiency_global():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.efficiency_global(A, weighted=False)
    npt.assert_equal(1, np.asarray(m).size)

    A = nig.get_random_graph(30, weighted=True)
    m = nig.efficiency_global(A, weighted=True)
    npt.assert_equal(1, np.asarray(m).size)


def test_efficiency_local():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.efficiency_local(A, weighted=False)
    npt.assert_equal(1, np.asarray(m).size)

    A = nig.get_random_graph(30, weighted=True)
    m = nig.efficiency_local(A, weighted=True)
    npt.assert_equal(1, np.asarray(m).size)


def test_efficiency_nodal():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.efficiency_nodal(A)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_eigenvector_centrality():
    A = nig.get_random_graph(30)
    m = nig.eigenvector_centrality(A)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_pagerank():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.pagerank(A)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_resolvent_centrality():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.resolvent_centrality(A)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_subgraph_centrality():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.subgraph_centrality(A)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_spread_of_infection():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.spread_of_infection(A)
    npt.assert_equal(A.shape[0], m.shape[0])


def test_google_matrix():
    A = nig.get_random_graph(30, weighted=False)
    m = nig.google_matrix(A)
    npt.assert_equal(A.shape, m.shape)


# def test_controllability():
#     A = nig.get_random_graph(30, directed=True)
#     m = nig.controllability(A)
#     npt.assert_equal(A.shape[0], m.shape[0])


# 'node_importance', 'size_giant_component', 'synchronizability',
# 'adjacency_spectrum',
# 'algebraic_connectivity',
# 'small_world_scalar',
# 'small_world_scalar_faster',
# '', 'rich_club_coefficients',
# 'rich_club_coefficient', 'vulnerability',
# 'diversity_coefficient', '',
