#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Algorithms for the detection of community structure in network graphs.

Algortithm Overview
====================

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 14 14 14


   * - Method name
     - Parameter
     - Weighted
     - Directed
     - overlapping
     - Tree

   * - `spectral`
     - n_partitions
     - yes
     - --
     - --
     - --

   * - `edge_betweenness`
     - --
     - yes
     - yes
     - --
     - yes

   * - `label_propagation`
     - --
     - yes
     - --
     - --
     - --

   * - `walktrap`
     - steps
     - yes
     - --
     - --
     - yes

   * - `optimal_modularity`
     - --
     - yes
     - --
     - --
     - --

   * - `leading_eigenvector`
     - n_partitions
     - --
     - --
     - --
     - --

   * - `fast_greedy`
     - --
     - yes
     - --
     - --
     - yes

   * - `louvain`
     - --
     - yes
     - --
     - --
     - yes

   * - `k_shell`
     - --
     - --
     - --
     - --
     - --

   * - `s_core`
     - s (strength)
     - yes
     - --
     - --
     - --

   * - `simulated_annealing`
     - n_spins, n_iter
     - yes
     - --
     - --
     - --

   * - `affinity_propagation`
     - p
     - yes
     - --
     - --
     - --

   * - `link_communities`
     - --
     - yes
     - --
     - yes
     - --

   * - `link_communities_cpp`
     - thr
     - --
     - --
     - yes
     - --

   * - `expectation_maximization`
     - n_partitions
     - ?
     - --
     - --
     - --

   * - `expectation_maximization_overlapping`
     - n_partitions, thr
     - ?
     - --
     - yes
     - --

   * - `moses`
     - --
     - --
     - --
     - yes
     - --

   * - `fastcommunity`
     - --
     - ?
     - ?
     - --
     - --

   * - `edge_clustering`
     - g, level
     - --
     - --
     - --
     - --

   * - `cfinder`
     - many
     - --
     - --
     - yes
     - --

   * - `greedy_clique_expansion`
     - s, eta, alpha, phi
     - ?
     - --
     - yes
     - --

   * - `mixture_model`
     - n_partitions
     - --
     - yes
     - --
     - --

   * - `markov_cluster`
     - inflation
     - ?
     - --
     - --
     - --

   * - `infomap`
     - n_attempts
     - yes
     - yes
     - --
     - --

   * - `hierarchical_random_graphs`
     - max_fit, max_consensus, t, method
     - --
     - --
     - --
     - yes

   * - `rich_club`
     - alpha
     - --
     - --
     - --
     - --

   * - `k_means`
     - n_partitions
     - yes
     - --
     - --
     - --

   * - `local_search_optimization`
     - n_partitions (optional)
     - yes
     - --
     - --
     - --
"""

import numpy as np
import igraph as ig
from .utilities import graph_type, convert_to_graph


def label_propagation(A, weighted=False):
    """Community structure based on label propagation (LP)

    Finds the community structure of the graph according to the label
    propagation method of [1]_.

    Initially, each node is assigned a different label. After that, each vertex
    chooses the dominant label in its neighbourhood in each iteration. Ties are
    broken randomly and the order in which the vertices are updated is
    randomized before every iteration. The algorithm ends when vertices reach a
    consensus.

    Note that since ties are broken randomly, there is no guarantee that the
    algorithm returns the same community structure after each run. In fact,
    they frequently differ. See the paper of Raghavan et al. on how to come up
    with an aggregated community structure.

    Parameters
    ----------
    A : ndarray, shape(n, n) | ig.Graph
        Adjacency matrix of the graph
    weighted : boolean
        The adjacency matrix is weighted

    Returns
    -------
    n2c : ndarray, shape(n, )
        Community structure. A mapping from node IDs [0..n-1] to community
        IDs [0..nc-1].

    extras : dict

        * ``Q``: Newman modularity.

    References
    ----------
    .. [1] Raghavan, U., Albert, R., & Kumara, S. (2007). Near linear time
           algorithm to detect community structures in large-scale networks.
           Physical Review E, 76(3). doi:10.1103/PhysRevE.76.036106
    .. [2] This implementation is based on `igraph
           <http://igraph.sourceforge.net/>`_

    Examples
    --------
    >>> from nigraph import *
    >>> n2c, extras = label_propagation(karate_club())
    >>> print extras['Q']  #doctest: +SKIP
    0.132807363577
    """
    if graph_type(A) == 'np':
        A = convert_to_graph(A, weighted=weighted, fmt='ig')

    weights = None
    if weighted:
        weights = 'weight'

    extras = {}
    c = A.community_label_propagation(weights=weights, initial=None,
                                      fixed=None)
    n2c = np.asarray(c.membership)
    q = c.modularity

    extras['Q'] = q
    return (n2c, extras)


def louvain(A, weighted=False, return_tree=False):
    """Community structure based on Louvain (LV)

    This is a bottom-up algorithm: initially every vertex belongs to a
    separate community, and vertices are moved between communities iteratively
    in a way that maximizes the vertices' local contribution to the overall
    modularity score. When a consensus is reached (i.e. no single move would
    increase the modularity score), every community in the original graph is
    shrank to a single vertex (while keeping the total weight of the adjacent
    edges) and the process continues on the next level. The algorithm stops
    when it is not possible to increase the modularity any more after shrinking
    the communities to vertices.

    This algorithm runs almost in linear time on sparse graphs.

    Parameters
    ----------
    A : ndarray, shape(n, n) | ig.Graph | edge_list
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted.
    return_tree : boolean
        Return the hierarchical community tree (default=False).

    Returns
    -------
    n2c : ndarray, shape(n, )
        Community structure. A mapping from node IDs [0..n-1] to community
        IDs [0..nc-1].

    extras : dict

        * ``Q``: Newman modularity. If ``return_tree=True`` this is the
          modularity at each hierarchy level.
        * ``tree``: If ``return_tree=True`` the whole community tree is
          returned, shape(n_level, n).

    Notes
    -----
    :func:`~local_search_optimization` implements the Louvain heuristic, but
    allows in addition to specify the number of partitions.

    See Also
    --------
    nt.community.metrics.static.find_cut: Optimal cut of a non-overlapping
        hierarchical community dendrogram
    local_search_optimization: Community structure based on Local Search
        Optimization

    References
    ----------
    .. [1] Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E.
           (2008). Fast unfolding of communities in large networks. Journal of
           Statistical Mechanics: Theory and Experiment, 2008(10), P10008.
           doi:10.1088/1742-5468/2008/10/P10008
    .. [2] This implementation is based on `igraph
           <http://igraph.sourceforge.net/>`_

    Examples
    --------
    >>> from nigraph import *
    >>> A = karate_club()
    >>> n2c, extras = louvain(A, weighted=False, return_tree=True)
    >>> print extras.keys()
    ['Q', 'tree']
    >>> print extras['Q'] # modularity at all levels  #doctest: +SKIP
    [ 0.36135766  0.41880342]
    """
    if graph_type(A) == 'np':
        A = convert_to_graph(A, weighted=weighted, fmt='ig')
    elif graph_type(A) == 'edge_list':
        if weighted:
            raise NotImplementedError
        A = ig.Graph(A, directed=False)

    weights = None
    if weighted:
        weights = 'weight'

    n_nodes = A.vcount()

    extras = {}
    if return_tree:
        cc = A.community_multilevel(weights=weights, return_levels=return_tree)
        n_level = len(cc)  # hierarchy levels

        tree = np.empty((n_level, n_nodes))
        q = np.zeros(n_level)
        for i in range(n_level):
            tree[i, :] = np.asarray(cc[i].membership)
            q[i] = cc[i].modularity  # is already calculated

        extras['tree'] = tree
        idx = np.where(q == q.max())[0]  # select max modualrity
        n2c = tree[idx, :]

    else:
        c = A.community_multilevel(weights=weights, return_levels=False)
        n2c = np.asarray(c.membership)
        q = c.modularity

    extras['Q'] = q
    return (n2c, extras)


def k_shell(A, only_core_periphery=False):
    """Community structure based on k-core decomposition (KS)

    Finds the coreness (shell index) of the vertices of the network.

    The M{k}-core of a graph is a maximal subgraph in which each vertex has at
    least degree k. (Degree here means the degree in the subgraph). The
    coreness of a vertex is M{k} if it is a member of the M{k}-core but not a
    member of the M{k+1}-core.

    This supports unweighted, undirected networks.

    Parameters
    ----------
    A : ndarray, shape(n, n) | ig.Graph
        Adjacency matrix of the graph.
    only_core_periphery : boolean
        If True this returns the core-periphery structure with core (1) and
        periphery (0) (default=False).

    Returns
    -------
    n2c : ndarray, shape(n, )
        Community structure. A mapping from node IDs [0..n-1] to community
        IDs [0..nc-1].

    extras : dict

        * ``nk``: number of k-core nodes

    Notes
    -----
    A core-periphery structure of a network is a division of the nodes into a
    densely connected core and a sparsely connected periphery. The core in a
    network is not merely densely connected but also tends to be “central” to
    the network in terms of short paths through the network.

    Several results underscore the importance of considering core-periphery
    structure in addition to community structure. Nodes with a high degree
    (‘hubs’) can pose a problem for classic community detection, as they often
    are connected to nodes in many parts of a network and can thus have strong
    ties to several diﬀerent communities.

    References
    ----------
    .. [1] Carmi, S., Havlin, S., Kirkpatrick, S., Shavitt, Y., & Shir, E.
           (2007). A model of Internet topology using k-shell decomposition.
           Proceedings of the National Academy of Sciences of the United States
           of America, 104(27), 11150–11154. doi:10.1073/pnas.0701175104
    .. [2] Hagmann, P., Cammoun, L., Gigandet, X., Meuli, R., Honey, C. J.,
           Wedeen, V. J., & Sporns, O. (2008). Mapping the structural core of
           human cerebral cortex. PLoS Biology, 6(7), e159.
           doi:10.1371/journal.pbio.0060159
    .. [3] The unweighted implementation is based on `igraph
           <http://igraph.sourceforge.net/>`_

    See Also
    --------
    s_core: weighted k-shell decomposition

    Examples
    --------
    >>> from nigraph import *
    >>> A = get_graph(fmt='np')
    >>> n2c, _ = k_shell(A, only_core_periphery=False)
    >>> print n2c  #doctest: +SKIP
    [1 1 1 1 2 1 1 1 1 0 1 1 0 1 2 3 3 2 2 1 2 1 1 0 2 1 2 2 2 3 3]

    >>> print k_shell(A, only_core_periphery=True) # core-periphery division
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
    """

    if graph_type(A) == 'np':
        A = convert_to_graph(A, fmt='ig')
    n2c = np.asarray(A.coreness())

    if only_core_periphery:
        idx = np.where(n2c == np.max(n2c))[0]
        n2c[:] = 0
        n2c[idx] = 1

    n_core_nodes = n2c[n2c == n2c.max()].size
    return (n2c, {'nk': n_core_nodes})
