#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Metrics for static graphs.

**Centrality metrics**

.. autosummary::
   :nosignatures:

    degree
    betweenness_centrality
    eigenvector_centrality
    subgraph_centrality
    resolvent_centrality
    pagerank
    efficiency_nodal

**Local graph properties**

.. autosummary::
   :nosignatures:

    dist_matrix_topological
    local_characteristic_path_length
    nodal_average_shortest_path_length
    absorption
    driftness
    node_importance
    adjacency_spectrum
    rich_club_coefficients
    rich_club_coefficient
    vulnerability
    google_matrix
    spread_of_infection

**Global graph properties**

.. autosummary::
   :nosignatures:

    efficiency_global
    efficiency_local
    avg_shortest_path_length
    wiring_costs
    synchronizability
    algebraic_connectivity
    small_world_scalar
    small_world_scalar_faster
    controllability

Graph metrics that take the **community structure** of a graph into account.

.. autosummary::
   :nosignatures:

   within_module_degree_z_score
   participation_coefficient
   module_centrality
   diversity_coefficient
   number_k_max
   size_giant_component

"""
import networkx as nx
import scipy as scp
import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import time
from numpy.random import binomial
from copy import deepcopy

import scipy.sparse.csgraph as skl_graph
from scipy.sparse.csgraph import connected_components as cs_graph_components

from ..utilities import convert_to_graph, inverse_adj, subgraph, \
    remove_nodes, make_undirected
from ..utilities import tril_indices, triu_indices
from ..utilities import remove_self_loops, graph_type, number_edges, \
    number_nodes, is_directed, laplacian_matrix

from ..utilities import num_communities, is_overlapping, unique_communities

advanced = False
if advanced:
    # import igraph as ig
    import graph_tool.all as gt

# TODO this module needs a re-ordering of the functions
__all__ = ['degree', 'betweenness_centrality', 'dist_matrix_topological',
           'avg_shortest_path_length', 'absorption', 'driftness',
           'local_characteristic_path_length',
           'nodal_average_shortest_path_length',
           'within_module_degree_z_score', 'participation_coefficient',
           'number_k_max', 'wiring_costs',
           'node_importance', 'size_giant_component', 'synchronizability',
           'adjacency_spectrum',
           'algebraic_connectivity', 'efficiency_global', 'efficiency_local',
           'efficiency_nodal', 'small_world_scalar',
           'small_world_scalar_faster',
           'controllability', 'rich_club_coefficients',
           'rich_club_coefficient', 'vulnerability',
           'diversity_coefficient', 'eigenvector_centrality',
           'subgraph_centrality', 'resolvent_centrality', 'google_matrix',
           'pagerank', 'spread_of_infection']


def degree(A, directed=False, ignore_self_loops=False):
    r"""degree centrality (DC) of the nodes in the graph

    The degree is the *number of neighbors* of of a node i. For weighted
    networks it is the *sum of edge weights* for a given node (connectivity
    strength).

    .. math::

        k_i = \sum_{j \in N} a_{ij}

    where :math:`a_{ij}` is the connection status between node `i` and node
    `j`.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph
    directed : boolean
        The adjacency matrix is directed (default=False).
    ignore_self_loops : boolean
        Ignore self-loops for the degree calculation (default=False). In case
        the main diagonal is already zero, this should be set to ``False`` in
        order to speed-up the computation.

    Returns
    -------
    k : ndarray, shape(n, )
        local degree

    k_in : ndarray, shape(n, )
        If ``directed=True``: local in-degree (defined as the lower triangle of
        the adjacency matrix)

    k_out : ndarray, shape(n, )
        If ``directed=True``: local out-degree (defined as the upper triangle
        of the adjacency matrix)

    Notes
    -----
    Returns the connectivity strength (weighted degree) if the adjacency matrix
    is weighted.

    Examples
    --------
    >>> from nigraph import get_graph, degree
    >>> # unweighted degree
    >>> A = get_graph(weighted=False)
    >>> k = degree(A)
    >>> print k
    array([ 3., 2.,  2.,  2.,  4.,  2.,  2.,  2.,  2.,  1.,  3.,  2.,  1.,
            3., 4.,  4.,  4.,  3.,  4.,  2.,  4.,  2.,  2.,  1.,  4.,  2.,
            3., 4.,  3.,  4.,  4.])

    >>> # weighted degree (strength)
    >>> A = get_graph(weighted=True)
    >>> k = degree(A)
    >>> print k
    array([ 2.3408977 ,  1.55037578,  1.79052192,  1.60754308,  2.694035  ,
            1.73456824,  1.83475922,  1.62522424,  1.62522424,  1.        ,
            2.19096524,  1.61817237,  1.        ,  2.07327917,  2.89348897,
            2.9024747 ,  3.08608532,  2.15749671,  2.58347417,  1.73456824,
            2.99150174,  1.68341924,  1.68341924,  1.        ,  2.99956307,
            1.61141989,  2.41093933,  3.01251548,  2.35483143,  3.13909815,
            3.00507946])

    >>> # directed degree
    >>> A = get_graph(directed=True)
    >>> k_in, k_out = degree(A, directed=True)
    >>> print k_in
    array([ 3.,  3.,  1.,  1.,  1.,  0.,  2.,  2.,  2.,  1.,  2.,  3.,  0.,
            1.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  1.,  0.,  0.,  0.])
    >>> print k_out
    array([ 2.,  0.,  0.,  1.,  2.,  1.,  1.,  1.,  0.,  0.,  2.,  0.,  0.,
            1.,  2.,  3.,  2.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  3.,  0.,
            1.,  0.,  0.,  1.,  0.])
    """

    n_nodes = A.shape[0]
    if ignore_self_loops:
        # this creates an expensive copy of the matrix
        A = remove_self_loops(A, copy=True)

    if directed:
        T = A.copy()
        idx = triu_indices(n_nodes, 0)
        T[idx] = 0
        k_in = np.sum(T, axis=0)

        T = A.copy()
        idx = tril_indices(n_nodes, 0)
        T[idx] = 0
        k_out = np.sum(T, axis=1)
        return k_in, k_out

    else:
        return np.sum(A, axis=1)


def betweenness_centrality(A, weighted=False, directed=False, norm=True):
    r"""betweenness centrality (BC) for nodes in the graph

    Betweenness centrality of a node :math:`v` is the sum of the fraction of
    all-pairs shortest paths that pass through :math:`v`:

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the number of
    shortest :math:`(s, t)`-paths, and :math:`\sigma(s, t|v)` is the number of
    those paths passing through some  node :math:`v` other than :math:`s, t`.
    If :math:`s = t`, :math:`\sigma(s, t) = 1`, and if :math:`v \in {s, t}`,
    :math:`\sigma(s, t|v) = 0` [2]_.

    Parameters
    ----------
    A : ndarray, shape(n, n) **or** nx.Graph **or** ig.Grah **or** gt.Graph
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted. #not impl yet
    directed : boolean
        The adjacency matrix is directed.
    norm : boolean
        If True the betweenness values are normalized by :math:`1/(n-1)(n-2)`
        where :math:`n` is the number of nodes in A.

    Returns
    -------
    v : ndarray, shape(n, )
       betweenness centrality for each node.

    Notes
    -----
    See [3]_ for a comparison of different centrality metrics in brain
    networks.

    References
    ----------
    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
           Journal of Mathematical Sociology 25(2):163-177, 2001.
           http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness Centrality
           and their Generic Computation. Social Networks 30(2):136-145, 2008.
           http://www.inf.uni-konstanz.de/algo/publications/b-vspbc-08.pdf
    .. [3] Zuo, X.-N., Ehmke, R., Mennes, M., Imperati, D., Castellanos, F. X.,
           Sporns, O., & Milham, M. P. (2011). Network Centrality in the Human
           Functional Connectome. Cerebral cortex (New York, NY : 1991).
           doi:10.1093/cercor/bhr269

    Examples
    --------
    >>> from nigraph import ahn_graph, betweenness_centrality
    >>> A = ahn_graph()
    >>> betweenness_centrality(A)
    array([ 0., 0., 0., 0.75, 0., 0., 0.42857143, 0., 0.])
    >>> betweenness_centrality(A, norm=False)
    array([ 0., 0., 0., 21., 0., 0., 12., 0., 0.])
    >>> # comparison with degree
    >>> degree(nx.to_numpy_matrix(A))
    array([ 3.,  3.,  3.,  6.,  2.,  2.,  3.,  2.,  2.])
    """

    # XXX think about how to handle this
    # if auto_inv:
    #     # inverse of ADJ required for some metrics
    #     # different ways on how to do that:
    #     # weight=1/v, weight=max(v)-v, or weight=exp(-v)
    #     A = inverse_adj(A, method='inv', norm=True)

    if graph_type(A) is 'np':
        A = convert_to_graph(A, weighted=weighted, directed=directed, fmt='ig')

    if graph_type(A) is 'ig':

        weights = None
        if weighted:
            weights = 'weight'  # = g.edge_attributes()
        M = np.asarray(A.betweenness(vertices=None, directed=directed,
                       cutoff=None, weights=weights, nobigint=True))

        norm = (A.vcount() - 1) * (A.vcount() - 2)
        M /= float(norm)
        if not directed:
            M *= 2

    elif graph_type(A) is 'nx':

        nx_version = nx.__version__
        if not nx_version == '1.4':
            if weighted:
                weighted = 'weight'
            else:
                weighted = None

        if nx_version == '1.4':
            M = nx.algorithms.betweenness_centrality(A, normalized=norm,
                                                     weighted_edges=weighted)
        else:
            M = nx.algorithms.betweenness_centrality(A, normalized=norm,
                                                     weight=weighted)

        M = np.asarray(M.values(), dtype=np.float64)

    elif graph_type(A) is 'gt':

        weights = None
        if weighted:
            weights = A.edge_properties["weight"]

        vb, eb = gt.betweenness(A, vprop=None, eprop=None, weight=weights,
                                norm=norm)
        M = vb.a
    return M


def dist_matrix_topological(A, weighted=False, directed=False, norm=False,
                            auto_inv=True):
    """topological distance matrix

    This is the *topological* distance Matrix, containing the shortest-path
    from each node to all other nodes in the network.

    Supports weighted, unweighted, directed and undirected graphs.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted (default=False).
    directed : boolean
        The adjacency matrix is directed (default=False).
    norm : boolean
        Normalize the distance matrix (default=False).
    auto_inv : boolean
        Auto inverse the adjacency matrix if the network is weighted
        (default=True).

    Returns
    -------
    D : ndarray, shape(n, n)
        Topological distance matrix.

    Notes
    -----
    Depending on the network sparsity this implementation uses either the
    Floyd-Warshall algorithm (O[N^3]) or the Dijkstra's algorithm
    ([(k+log(N))N^2]) with Fibonacci stacks.

    See also
    --------
    nt.roi.utils.dist_matrix_spatial: *spatial* distance matrix

    Examples
    --------
    >>> from nigraph import get_random_graph, dist_matrix_spatial
    >>> A = get_random_graph()
    >>> print A.shape
    (30, 30)
    >>> D = dist_matrix_topological(A)
    >>> print D.shape
    (30, 30)
    """

    n_nodes = A.shape[0]
    if weighted:
        if auto_inv:
            A = inverse_adj(A, method='inv')

    D = skl_graph.graph_shortest_path(sp.lil_matrix(A), directed=directed)
    if norm:
        D /= (n_nodes * (n_nodes - 1))
    return D


def avg_shortest_path_length(A, weighted=False, directed=False, auto_inv=True):
    """average shortest path length of a graph

    The average shortest path length, also called *characteristic path length*,
    is:

    .. math::

        a = \sum_{s,t \in V} \\frac{d(s,t)}{n(n-1)}


    where `V` is the set of nodes in `A`, `d(s, t)` is the shortest path from
    `s` to `t`, and `n` is the number of nodes in `A`.

    Supports weighted, unweighted, directed and undirected graphs.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted (default=False).
    directed : boolean
        The adjacency matrix is directed (default=False).
    auto_inv : boolean
        Auto inverse the adjacency matrix if the network is weighted
        (default=True).

    Returns
    -------
    v : float
        Average shortest path length.

    Notes
    -----
    About 5-100 times faster than ``nx.average_shortest_path_length()``

    ::

        >>> %timeit avg_shortest_path_length(A, weighted=False)
        100 loops, best of 3: 4.95 ms per loop
        >>> %timeit nx.average_shortest_path_length(nx.Graph(A),
                                                    weight='weight')
        1 loops, best of 3: 284 ms per loop

    Examples
    --------
    >>> from nigraph import karate_club, avg_shortest_path_length
    >>> A = karate_club()
    >>> v = avg_shortest_path_length(A)
    """

    # NB this does _not_ compute the shortest path within each connected
    # component
    ADJ = deepcopy(A)

    n_nodes = ADJ.shape[0]
    if weighted:
        if auto_inv:
            ADJ = inverse_adj(ADJ, method='inv')

    M = sp.lil_matrix(ADJ)
    d = skl_graph.graph_shortest_path(M, directed=False).sum()
    d /= (n_nodes * (n_nodes - 1))
    return d


def local_characteristic_path_length(A, weighted=False, auto_inv=True):
    r"""local characteristic path length

    The local characteristic path length is a measure of functional integration
    and is defined as:

    .. math::

        L_{i} = \frac{1}{n-1} \sum_{j \in N} d_{ij}

    where :math:`d_{ij}` is the shortest path length (distance), between nodes
    `i` and `j`.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted (default=False).
    auto_inv : boolean
        Auto inverse the adjacency matrix if the network is weighted
        (default=True).

    Returns
    -------
    v : ndarray, shape(n, )
        Local characteristic path length for each node.

    Notes
    -----
    The characteristic path length is the most commonly used measure of
    functional integration. However, the characteristic path length is
    primarily influenced by long paths, while the global efficiency is
    primarily influenced by short paths. Some authors have argued that this may
    make the global efficiency a superior measure of integration [2]_ [3]_.

    See Also
    --------
    avg_shortest_path_length: the characteristic path length
    efficiency_nodal: nodal efficiency

    References
    ----------
    .. [1] van den Heuvel, M. P., Stam, C. J., Kahn, R. S., & Hulshoff Pol, H.
           E. (2009). Efficiency of functional brain networks and intellectual
           performance. Journal of Neuroscience, 29(23), 7619–7624.
           doi:10.1523/JNEUROSCI.1443-09.2009
    .. [2] Achard, S., & Bullmore, E. T. (2007). Efficiency and cost of
           economical brain functional networks. PLoS computational biology,
           3(2), e17. doi:10.1371/journal.pcbi.0030017
    .. [3] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain
           connectivity: uses and interpretations. NeuroImage, 52(3),
           1059–1069. doi:10.1016/j.neuroimage.2009.10.003

    Examples
    --------
    >>> from nigraph import karate_club, local_characteristic_path_length
    >>> A = karate_club()
    >>> L = local_characteristic_path_length(A)
    """

    # NB this implementation assumes the graph is connecetd
    # skl is about 35% faster than the graph_tool implementation!
    if weighted:
        if auto_inv:
            A = deepcopy(A)
            A = inverse_adj(A, method='inv')

    # dist matrix
    local_path_length = skl_graph.graph_shortest_path(sp.lil_matrix(A),
                                                      directed=False)
    return np.nan_to_num(local_path_length.sum(-1) / (local_path_length != 0).sum(1))


def absorption(A, n_walks=10):
    r"""returns the absorption matrix

    This uses a random walk to measure diffusion processes in networks. The
    entry :math:`AB_{i,j}` of the absorption matrix AB denotes the average
    number of steps an agent takes to be absorbed at node j departing from node
    i.

    This metric is different from other network metrics based on the shortest
    path distance between nodes.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    n_walks : integer
        Number of random walks.

    Returns
    -------
    AB : ndarray, shape(n, n)
        Absorption matrix, averaged over ``n_walks``.

    Notes
    -----
    The absorption matrix is not necessarily symmetric.

    References
    ----------
    .. [1] da Fontoura Costa, L., Batista, J. L. B., & Ascoli, G. A. (2011).
           Communication structure of cortical networks. Frontiers in
           computational neuroscience, 5, 6. doi:10.3389/fncom.2011.00006

    Examples
    --------
    >>> A = nx.to_numpy_matrix(nx.cycle_graph(5))
    >>> AB = absorption(np.asarray(A))
    """

    # TODO the random walk is implemented in a way that allows to stay at a
    # node or walk "back"... check paper again whether that should be allowed

    # TODO this implementation of a random walker, that takes the edge weights
    # into account
    # https://github.com/Midnighter/Random-Walker/blob/master/walkers.py

    n_nodes = A.shape[0]

    res = np.zeros((n_walks, n_nodes, n_nodes))
    for n in range(n_walks):
        for i in range(n_nodes):
            for j in range(n_nodes):
                steps = 0
                if not i == j:
                    # neig = A[i,:]
                    neig = np.where(A[i, :] != 0)[0]

                    # simulate random walk
                    not_j = True
                    while not_j:
                        node = np.random.permutation(neig)
                        if node[0] == j:
                            not_j = False

                        # neig = A[node,:]
                        neig = np.where(A[node[0], :] != 0)[0]
                        steps += 1

                res[n, i, j] = steps

    return res.mean(axis=0)


def driftness(A, AB=None, D=None):
    r"""returns driftness matrix

    Driftness is the relation of the absorption between any two nodes
    :math:`AB_{i,j}` divided by the respective shortest path distance
    :math:`D_{i,j}`.

    .. math::

        W_{i,j} = \frac{AB_{i,j}}{D_{i,j}} for AB_{i,j} \neq 0

    For :math:`AB_{i,j} = 0` the driftness is defined to be 0. A value of 1
    means that the shortest path distance and the driftness are identical.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    AB : ndarray, shape(n, n)
        Adjacency matrix of the graph (default=None).
    D : ndarray, shape(n, n)
        Distance matrix of the graph (default=None).

    Returns
    -------
    W : ndarray, shape(n, n)
        Driftness matrix of the graph.

    Notes
    -----
    The graph needs to be connected.

    See Also
    --------
    absorption: absorption matrix

    References
    ----------
    .. [1] da Fontoura Costa, L., Batista, J. L. B., & Ascoli, G. A. (2011).
           Communication structure of cortical networks. Frontiers in
           computational neuroscience, 5, 6. doi:10.3389/fncom.2011.00006

    Examples
    --------
    >>> A = nx.to_numpy_matrix(nx.cycle_graph(5))
    >>> W = driftness(np.asarray(A))
    """

    if AB is None:
        AB = absorption(A)

    if D is None:
        D = dist_matrix_topological(A, weighted=False)

    idx = D.nonzero()
    W = np.zeros((AB.shape[0], AB.shape[0]))
    W[idx] = AB[idx] / D[idx]
    return W


def nodal_average_shortest_path_length(A, weighted=False, auto_inv=True):
    """average shortest path lenth from node i to all other nodes in the network

    #TODO: untested: see closeness centralit it's almost the same!!!

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph
    weighted : boolean
        The adjacency matrix is weighted
    auto_inv : boolean
        Auto inverse the adjacency matrix if the network is weighted
        (default=True).

    Returns
    -------
    value : ndarray, shape(n, )
        Nodal average shortest path length.

    Examples
    --------
    >>> from nigraph import karate_club, nodal_average_shortest_path_length
    >>> A = karate_club()
    >>> pl = nodal_average_shortest_path_length(A)
    >>> k = degree(A)
    >>> #plt.scatter(k, pl, s=50,c='r',alpha=0.5) # compare to degree
    """

    if weighted:
        if auto_inv:
            A = inverse_adj(A, method='inv')

    G = nx.Graph(A)

    if weighted:
        path_length = nx.single_source_dijkstra_path_length
    else:
        path_length = nx.single_source_shortest_path_length

    n_nodes = len(G)
    avg = np.zeros(n_nodes)
    c = 0
    for node in G:
        length = list(path_length(G, node).values())
        avg[c] = sum(length)
        c += 1

    return avg / (n_nodes * (n_nodes - 1))


def within_module_degree_z_score(A, partition=None):
    r"""within-module degree z-score for each node

    The within-module degree z-score measures how well-connected
    node i is to other nodes in the same module.

    .. math::

        z_i =  \frac{k_i-\bar{k}_{s_i}}{\sigma k_{s_i}}

    where :math:`k_{i}` is the number of links of node i to other nodes in its
    module :math:`s_i`, :math:`k_{s_i}` is the average degree over all nodes in
    :math:`s_i`, and :math:`\sigma k_{s_i}` is the standard deviation of the
    node degree in :math:`s_i`.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    partition : ndarray, shape(n, )
        Partition vector of the graph into non-overlapping modules.

    Returns
    -------
    v : ndarray, shape(n, )
        Within-module degree z-score for each node.

    Notes
    -----
    Supports weighted and unweighted graphs, depending on the input matrix
    ``A``.

    See Also
    --------
    participation_coefficient: participation coefficient
    nt.community.metrics.static.zp_parameter_space_stats: classification of
        nodes based on the zP-parameter space

    References
    ----------
    .. [1] Guimerà & Amaral (2005). Functional cartography of complex metabolic
           networks. Nature, 433(7028), 895–900. doi:10.1038/nature03288

    Examples
    --------
    >>> A = nig.karate_club()
    >>> n2c, _ = nig.leading_eigenvector(A)  # find a graph partition
    >>> z = nig.within_module_degree_z_score(A, partition=n2c)
    """

    # TODO if A is weighted the metric will always return the weighted version
    # assert partition != None, 'No graph partition specified'
    n_nodes = A.shape[0]
    res = np.zeros(n_nodes)
    for m in np.unique(partition):
        # select subgraph containing only nodes in the current module
        idx = np.where(partition == m)[0]
        SG = subgraph(A, idx, idy=None)
        ki_mi = np.sum(SG, axis=0)  # degree k in subgraph
        res[idx] = np.nan_to_num((ki_mi - ki_mi.mean()) / ki_mi.std())

    return res


def participation_coefficient(A, partition=None, theta=None):
    """participation coefficient for each node

    The participation coefficient measures how well a node is connected to
    other modules. It is close to :math:`P_{i}=1` if its links are uniformly
    distributed among all the modules and :math:`P_{i}=0` if all its links are
    within its own module.

    .. math::

        P_i = 1 - \sum (\\frac{k_{is}}{k_{i}})^2

    where :math:`k_{is}` is the number of links of node :math:`i` to nodes in
    module :math:`s`, and :math:`k_{i}` is the total degree of node :math:`i`.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    partition : ndarray, shape(n, )
        Partition vector of the graph into modules.
    theta : ndarray, shape(n, nc)
        In case of overlapping modules, the likelihood of node ``n`` belonging
        to module ``nc`` (default=None).

    Returns
    -------
    p : ndarray, shape(n, )
        Participation coefficient p.

    Notes
    -----
    Supports weighted and unweighted graphs, depending on the input matrix
    ``A``.

    See Also
    --------
    within_module_degree_z_score: within-module degree z-score
    diversity_coefficient: diversity coefficient *h*
    nt.community.metrics.static.zp_parameter_space_stats: classification of
        nodes based on the zP-parameter space

    References
    ----------
    .. [1] Guimerà & Amaral (2005). Functional cartography of complex metabolic
           networks. Nature, 433(7028), 895–900. doi:10.1038/nature03288

    Examples
    --------
    >>> from nigraph import karate_club, leading_eigenvector, participation_coefficient
    >>> A = karate_club()
    >>> n2c, _ = leading_eigenvector(A) # find a graph partition
    >>> p = participation_coefficient(A, partition=n2c)

    There is also some *experimental* support for overlapping communities

    >>> from nigraph import expectation_maximization_overlapping
    >>> A = karate_club()
    >>> n2c, extras = expectation_maximization_overlapping(A, n_partitions=2)
    >>> theta = extras['theta_matrix']
    >>> p = participation_coefficient(A, n2c, theta)
    """

    # assert partition != None, 'No graph partition specified'
    n_nodes = A.shape[0]

    res = np.zeros(n_nodes)
    k_i = np.sum(A, axis=-1)  # degree k of each node i

    if is_overlapping(partition):
        # overlapping communities; scaling might be off! UNTESTED
        ar = np.arange(n_nodes)
        n_partitions = num_communities(partition)
        unique_partitions = list(unique_communities(partition))

        # L1 norm of theta
        t = 1. / np.sum(np.abs(theta), -1)
        theta = theta * t[:, np.newaxis]

        for m in range(n_partitions):
            # select subgraph containing only nodes in the current module
            idx = []
            for i in range(n_nodes):
                for j in partition[i]:
                    if j == unique_partitions[m]:
                        idx.append(ar[i])

            SG = subgraph(A, idx, idy=None)
            # degree k in subgraph
            ki_mi = np.sum(SG, axis=-1)
            res[idx] += np.nan_to_num(ki_mi / k_i[idx]) ** 2 * theta[idx, m]

    else:
        # non-overlapping communities
        for m in np.unique(partition):
            # select subgraph containing only nodes in the current module
            idx = np.where(partition == m)[0]
            SG = subgraph(A, idx, idy=None)
            ki_mi = np.sum(SG, axis=-1)  # degree k in subgraph
            res[partition == m] = np.nan_to_num(ki_mi / k_i[idx]) ** 2

    return np.ones(n_nodes) - res


def number_k_max(A, perc=False):
    """number of nodes in the core (k_max) after k-shell decomposition

    Parameters
    ----------
    A : ndarray, shape(n, n) *or* ig.Graph *or* nx.Graph
        Adjacency matrix of the graph
    perc : boolean
        return the number in percent of the total number of nodes

    Returns
    -------
    value : integer
        Number of nodes in k_max after k-shell decomposition

    Notes
    -----
    This metric is sometimes also called k-core centrality (BCT toolbox)

    See Also
    --------
    nt.community.detection.k_shell: k-shell decomposition

    References
    ----------
    .. [1] Hagmann, P., Cammoun, L., Gigandet, X., Meuli, R., Honey, C. J.,
           Wedeen, V. J., & Sporns, O. (2008). Mapping the structural core of
           human cerebral cortex. PLoS Biology, 6(7), e159.
           doi:10.1371/journal.pbio.0060159

    Examples
    --------
    >>> from nigraph import get_graph, number_k_max
    >>> A = get_graph()
    >>> print number_k_max(A, perc=False)
    4
    >>> print number_k_max(A, perc=True)
    12.9032258065
    """

    if graph_type(A) is 'nx':
        M = np.asarray(nx.algorithms.find_cores(A).values())

    elif graph_type(A) == 'ig':
        M = np.asarray(A.coreness())

    elif graph_type(A) == 'np':
        G = convert_to_graph(A, fmt='ig')
        M = np.asarray(G.coreness())

    # count nodes with corresponding k-core value
    n_k_max = len(np.where(M == np.max(M))[0])

    if perc:
        n_k_max /= float(M.size)
        n_k_max *= 100
    return n_k_max


def wiring_costs(A, fmt='np'):
    r"""wiring cost of a weighted or unweighted network

    The wiring-costs are defined as:

    .. math::

        K = \frac{\#Edges}{\#PossibleEdges}

    Parameters
    ----------
    A : ndarray, shape(n, n) or nx.Graph
        Adjacency matrix of the graph. The diagonal is assumed to be zero.
    fmt : string ['nx'|'np']
        Input graph format.

    Returns
    -------
    value : float
        Network wiring costs.

    Notes
    -----
    In [2]_ wiring costs are defined as the sum of the product of connection
    weights and physical lengths.

    References
    ----------
    .. [1] Bullmore & Sporns (2012). The economy of brain network organization.
           Nature Reviews Neuroscience, 13(5), 336–349. doi:10.1038/nrn3214
    .. [2] Wang, Q., Sporns, O., & Burkhalter, A. (2012). Network analysis of
           corticocortical connections reveals ventral and dorsal processing
           streams in mouse visual cortex. Journal of Neuroscience, 32(13),
           4386–4399. doi:10.1523/JNEUROSCI.6063-11.2012

    Examples
    --------
    >>> from nigraph import get_graph, wiring_costs, thresholding_prop
    >>> A = get_graph(weighted=True)
    >>> print 'costs: %.2f' %(wiring_costs(A))
    costs: 0.06
    >>> A_thr = thresholding_prop(A, 0.04) # thresholding the adjacency matrix
    >>> print 'costs: %.2f' %(wiring_costs(A_thr))
    costs: 0.04
    """

    # based on input either nx or pure nt implementation
    m = 1.0
    if graph_type(A) is 'np':
        n_edges = number_edges(A)
        n_nodes = number_nodes(A)
        if not is_directed(A):
            m = 0.5
    else:
        n_edges = A.number_of_edges()
        n_nodes = len(A)
        if not nx.is_directed(A):
            m = 0.5

    # - n_nodes: assuming that the diagonal is zero!
    max_edges = (n_nodes ** 2 - n_nodes) * m
    costs = n_edges / max_edges
    return costs


def node_importance(A, weighted=False, metric='efficiency_global'):
    """node importance for each node

    Node importance is calculated by removing a node `i` and measuring the
    impact on the overall network [1]_. The impact is calculated for a certain
    network metric (e.g. ``efficiency_global()``).

    XXX metric is currently hardcoded

    Parameters
    ----------
    A : ndarray, shape(n, n) **or** nx.Graph
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted.
    metric : string
        Global network metric to investigate (default=efficiency_global).

    Returns
    -------
    value : ndarray, shape(n, )
        Local node importance.

    Notes
    -----
    The metric is also called vulnerability [2]_.

    See Also
    --------
    nt.graph.attack.edge_importance: Same function for network edges

    References
    ----------
    .. [1] Achard et al. (2006). A resilient, low-frequency, small-world human
           brain functional network with highly connected association cortical
           hubs. Journal of Neuroscience, 26(1), 63–72.
           doi:10.1523/JNEUROSCI.3874-05.2006
    .. [2] Gong et al. Cereb Cortex (2009) vol. 19 (3) pp. 524-36

    Examples
    --------
    >>> from nigraph import karate_club, node_importance
    >>> A = karate_club()
    >>> i = node_importance(A, weighted=False)
    >>> print 'Most important node:', np.where(i == i[i>0].min())[0]
    Most important node: [7]
    """

    n_nodes = A.shape[0]
    res = np.zeros(n_nodes)

    # orig value for the whole network
    M0 = efficiency_global(A, weighted, auto_inv=True)
    # M0 = metrics_global(A, weighted=weighted, metric=metric)
    for i in xrange(n_nodes):
        A_ = remove_nodes(A, i)
        # impact of removing that node
        M1 = efficiency_global(A_, weighted, auto_inv=True)
        # M1 = metrics_global(A_, weighted=weighted, metric=metric)
        res[i] = M0 - M1

    return res


def size_giant_component(A, directed=False, perc=True):
    """size of the giant connected component (GCC) of the graph

    For weighted and unweighted graphs.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    directed : boolean
        The adjacency matrix is directed.
    perc : boolean
        Size of the GCC in percent of the total network size (default=True).

    Returns
    -------
    value : float
        Size in % (if ``perc=True``) of the largest connected component.

    Examples
    --------
    >>> from nigraph import get_graph, size_giant_component
    >>> A = get_graph()
    >>> print size_giant_component(A, perc=True)
    48.3870967742
    >>> print size_giant_component(A, perc=False)
    15.
    """

    n_nodes = A.shape[0]

    # n_comp, label = cs_graph_components(A)
    n_comp, label = cs_graph_components(A, directed)
    # -2 indicates nodes with 0 degree
    # not anymore with recent sklearn > 0.13, but not an issue for the GCC
    size_gcc = np.bincount(label[label != -2]).max()  # size largest component

    if perc:
        size_gcc = size_gcc * 100. / n_nodes  # in %

    return float(size_gcc)


def synchronizability(A, weighted=False):
    r"""synchronizability of the graph

    .. math::

        S = \frac{\lambda_1}{\lambda_N}

    where :math:`\lambda_1` is the second smallest eigenvalue of the Laplacian
    matrix of A that is different from zero (also called algebraic connectivity)
    and :math:`\lambda_N` is the largest eigenvalue.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted.

    Returns
    -------
    S : float
        Synchronizability.

    Notes
    -----
    Heterogeneity in the degree distribution of complex networks often reduces
    the average distance between nodes but, paradoxically, may suppress
    synchronization in networks [5]_. A *synchronizability matrix* is introduced
    in [5]_ that can be used to identidy relevant edges for maximal
    synchronizability. See also [4]_. Synchronizability was used to characterise
    MEG brain networks in [1]_.

    See Also
    --------
    algebraic_connectivity: second-smallest eigenvalue of the Laplacian matrix

    References
    ----------
    .. [1] Bassett et al. (2006). Adaptive reconfiguration of fractal
           small-world human brain functional networks. Proceedings of the
           National Academy of Sciences of the United States of America,
           103(51), 19518–19523. doi:10.1073/pnas.0606005103
    .. [2] http://en.wikipedia.org/wiki/Laplacian_matrix
    .. [3] Motter, A. E., Zhou, C., & Kurths, J. (2005). Network
           synchronization, diffusion, and the paradox of heterogeneity.
           Physical Review E, Statistical, Nonlinear, and Soft Matter Physics,
           71(1 Pt 2), 016116.
    .. [4] Barahona, M., & Pecora, L. M. (2002). Synchronization in small-world
           systems. Physical Review Letters, 89(5), 054101.
    .. [5] Lu, J., Yu, X., Chen, G., & Cheng, D. (2004). Characterizing the
           synchronizability of small-world dynamical networks. Circuits and
           Systems I: Regular Papers, IEEE Transactions on, 51(4), 787–796.

    Examples
    --------
    >>> from nigraph import karate_club, synchronizability
    >>> A = karate_club()
    >>> print synchronizability(A)
    0.0258329977742
    """

    # scipy.linalg.eigvals might have some performance advantages over numpy
    eigenvalues_laplacian = np.linalg.eigvals(laplacian_matrix(A))
    eigenvalues_laplacian_sort = np.sort(eigenvalues_laplacian)

    # e o is always zero; e 1 = algebraic connectivity
    return eigenvalues_laplacian_sort[1] / eigenvalues_laplacian_sort[-1]


def algebraic_connectivity(A):
    r"""algebraic connectivity of a graph

    Defined as the second-smallest eigenvalue of the Laplacian matrix
    (:math:`\lambda_1`).

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.

    Returns
    -------
    value : float
        Algebraic connectivity.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Algebraic_connectivity

    Examples
    --------
    >>> from nigraph import karate_club, algebraic_connectivity
    >>> A = karate_club()
    >>> print algebraic_connectivity(A)
    0.468525226701
    """

    # e o is always zero; e 1 = algebraic connectivity
    eigenvalues_laplacian = np.linalg.eigvals(laplacian_matrix(A))
    return np.sort(eigenvalues_laplacian)[1]


def adjacency_spectrum(A):
    """returns the eigenvalues of the graph

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph (weighted or unweighted).

    Returns
    -------
    res : ndarray, shape(n, )
        Eigenvalues.

    See Also
    --------
    synchronizability: Synchronizability based on eigenvalues

    Examples
    --------
    >>> from nigraph import karate_club, adjacency_spectrum
    >>> A = karate_club()
    >>> s = adjacency_spectrum(A)
    """
    return np.linalg.eigvals(A)


def efficiency_global(A, weighted=False, directed=False, auto_inv=True):
    r"""global efficiency of the graph

    This is a *global* network measure commonly used to quantify functional
    integration.

    .. math::

        E_{glob} = \frac{1}{N(N-1)} \sum_{i \neq j} E_{ij} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac {1}{d_{ij}}


    where :math:`d_{ij}` is the shortest path length between node :math:`i`
    and node :math:`j` in graph A.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted (default=False).
    weighted : boolean
        The adjacency matrix is directed (default=False).
    auto_inv : boolean
        Auto inverse the adjacency matrix if the network is weighted
        (default=True).

    Returns
    -------
    E_glob : float
        Global efficiency in range 0..1.

    See Also
    --------
    local_characteristic_path_length: related functional integration measure

    References
    ----------
    .. [1] Latora & Marchiori (2001). Efficient behavior of small-world
           networks. Physical review letters, 87(19), 198701.

    Examples
    --------
    >>> from nigraph import get_graph
    >>> A = get_graph()
    >>> print efficiency_global(A)
    0.114582693292

    >>> A = get_graph(weighted=True, fmt='np')
    >>> print efficiency_global(A, weighted=True)
    0.070806067901533126
    """

    # the implementation for weighted and unweighted is checked against the
    # matlab BCT version. NB, with the graph above the matlab implementation
    # take ~10 ms, this impl. 1.63 ms with the cat cortex matlab takes 200 ms,
    # this implementation 15.8 ms
    n_nodes = A.shape[0]

    if weighted:
        if auto_inv:
            A = inverse_adj(A, method='inv', norm=False)

    d = skl_graph.graph_shortest_path(sp.lil_matrix(A), directed=directed)
    d = d[d != 0] ** -1  # for some reasons d[d != 0] **=-1 takes longer to compute
    avg = d.sum()
    avg *= 1.0/(n_nodes*(n_nodes - 1))

    # 10% slower
    # d = skl_graph.graph_shortest_path(sp.lil_matrix(A), directed=False)
    # idx = d.nonzero()
    # d[idx] = 1./d[idx]
    # avg = d.sum() / (n_nodes*(n_nodes - 1))
    return avg


def efficiency_local(A, weighted=False, auto_inv=True):
    r"""local efficiency of a graph

    This is a *global* network measure. The local efficiency is the global
    efficiency computed on the subgraph of each node.

    .. math::

        E_{loc} = \frac{1}{N} \sum_{i \in N} E_{glob, i}

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted (default=False).
    auto_inv : boolean
        Auto inverse the adjacency matrix if the network is weighted
        (default=True).

    Returns
    --------
    E : float
        Local efficiency in the range 0..1.

    Notes
    -----
    The weighted local efficiency broadly parallels the weighted clustering
    coefficient of Onnela et al. (2005) and distinguishes the influence of
    different paths based on connection weights of the corresponding neighbors
    to the node in question. In other words, a path between two neighbors with
    strong connections to the node in question contributes more to the local
    efficiency than a path between two weakly connected neighbors. Note that the
    weighted variant of the local efficiency is hence not a strict
    generalization of the unweighted variant.

    See also
    --------
    efficiency_nodal: efficiency nodal for each node
    efficiency_global: efficiency global for the network

    References
    ----------
    .. [1] Latora & Marchiori (2001). Efficient behavior of small-world
           networks. Physical review letters, 87(19), 198701.

    Examples
    --------
    >>> from nigraph import karate_club, efficiency_local
    >>> A = karate_club()
    >>> print efficiency_local(A)
    0.64512651020003953
    """

    if weighted:
        if auto_inv:
            A = inverse_adj(A, method='inv')

    n_nodes = A.shape[0]

    E_LOC = np.zeros(n_nodes)
    ks = np.zeros(n_nodes)

    # if directed:
    #     # not impl yet
    #     k_in, k_out = degree(A, directed=directed)
    #     divisor = (k_in[i] + k_out[i]) * (k_in[i] + k_out[i] - 1) - 2 * np.sum(A[i,:]*A[:,i])
    #
    #     for i in range(n_nodes):
    #         # get neighbors of node i
    #         idx = A[i,:] != 0
    #         k = np.sum(idx)  # degree
    #
    #         if k >= 2:
    #             SG = A[np.ix_(idx, idx)] # subgraph
    #             # distance matrix
    #             D = skl_graph.graph_shortest_path(sp.lil_matrix(SG), directed=False)**-1
    #             sum_sg = np.sum(D[D!=np.inf])
    #             if sum_sg > 0:
    #                 if weighted:
    #                     E_LOC[i] = sum_sg ** (1 / 3.) / (k ** 2 - k)
    #                 else:
    #                     E_LOC[i] = sum_sg / (k ** 2 - k)

    # loop over nodes, parallel version would speed this up significantly
    for i in range(n_nodes):
        # get neighbors of node i
        idx = A[i, :] != 0
        k = np.sum(idx)  # degree
        ks[i] = k
        if k >= 2:
            SG = A[np.ix_(idx, idx)]  # subgraph
            # distance matrix
            D = skl_graph.graph_shortest_path(sp.lil_matrix(SG),
                                              directed=False) ** -1
            sum_sg = np.sum(D[D != np.inf])
            if sum_sg > 0:
                if weighted:
                    E_LOC[i] = sum_sg ** (1 / 3.) / (k ** 2 - k)
                else:
                    E_LOC[i] = sum_sg / (k ** 2 - k)

    # divisor = (k ** 2 - k)
    if weighted:
        E_LOC = np.sum(E_LOC) * (0.5)
    else:
        E_LOC = np.mean(E_LOC)

    return E_LOC


def efficiency_nodal(A, weighted=False, auto_inv=True):
    r"""nodal efficiency for each node in the graph

    .. math::

        E_i = \frac{1}{N-1} \sum_j \frac{1}{d_{ij}}

    where :math:`d_{ij}` is the shortest path length between node :math:`i` and
    node :math:`j`.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted (default=False).
    auto_inv : boolean
        Auto inverse the adjacency matrix if the network is weighted
        (default=True).

    Returns
    -------
    E : ndarray, shape(n, )
        The nodal efficiency for each node of the graph. Defined value range
        is {0,1}.

    See Also
    --------
    efficiency_local: efficiency local for the network
    efficiency_global: efficiency global for the network

    References
    ----------
    .. [1] Latora & Marchiori (2001). Efficient behavior of small-world
           networks. Physical review letters, 87(19), 198701.

    Examples
    --------
    >>> from nigraph import karate_club, efficiency_nodal
    >>> A = karate_club()
    >>> # local efficiency for each member of the karate club
    >>> print efficiency_nodal(A)
    [ 0.7020202   0.58080808  0.63636364  0.53535354  0.44444444  0.45959596
      0.45959596  0.49747475  0.56060606  0.47222222  0.44444444  0.40909091
      0.42424242  0.56060606  0.43030303  0.43030303  0.33636364  0.42929293
      0.43030303  0.53030303  0.43030303  0.42929293  0.43030303  0.48585859
      0.42171717  0.42171717  0.42272727  0.51262626  0.49747475  0.46565657
      0.51262626  0.58585859  0.63383838  0.70454545]
    """

    n_nodes = A.shape[0]

    if weighted:
        if auto_inv:
            A = inverse_adj(A, method='inv')

    G = sp.lil_matrix(A)
    # SP = g.shortest_paths(source=None, target=None, weights=None)
    # SP = np.asarray(SP) # the igraph way

    SP = skl_graph.graph_shortest_path(G, directed=False)
    (i, j) = SP.nonzero()
    idx = np.where(SP == 0)
    SP = SP ** -1
    SP[idx] = 0
    d = np.sum(SP, axis=0)
    d /= (n_nodes - 1)
    return d


def small_world_scalar(A, weighted=False, method='watts', n_iter=100):
    r"""small-world scalar of the graph

    .. math::

        S = \frac{\frac{C}{C_{rand}}}{\frac{L}{L_{rand}}}

    The small world scalar describes the ratio of the average clustering
    coefficient of a network (``C``) and the characteristic path length (``L``).
    The coefficients are normalized by a version obtained from a randomized
    network (``C_rand`` and ``L_rand``). The number of randomized networks is
    controlled by ``n_iter``.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted. #TODO not impl yet
    method : string ['watts'|'humphries']

        * 'watts': original small-world introduces by Watts & Strogatz.
        * 'humphries': small-world scalar by Humphries & Gurney.

    n_iter : integer
        Number of iterations/randomizations.

    Returns
    -------
    C_ratio : float
        Normalized clustering coefficient (C/C_rand).
    L_ratio : float
        Normalized characteristic path length (L/C_rand).
    S : float
        Small-world scalar.

    Notes
    -----
    A value of S >> 1 is regarded as "small world".

    References
    ----------
    .. [1] Watts & Strogatz (1998). Collective dynamics of “small-world”
           networks. Nature, 393(6684), 440–442. doi:10.1038/30918
    .. [2] Humphries & Gurney (2008). Network “small-world-ness”: a quantitative
           method for determining canonical network equivalence. PloS one, 3(4),
           e0002051. doi:10.1371/journal.pone.0002051

    Examples
    --------
    >>> from nigraph import karate_club, small_world_scalar
    >>> A = karate_club()
    >>> C_ratio, L_ratio, S = small_world_scalar(A)
    """

    # TODO: how do deal with unconnected graphs?
    # TODO: impl different methods

    # !! cave C can be calculated based on the cluster coef defined by watts & st
    # or by transitivity --> see paper

    fmt = graph_type(A)
    if fmt == 'np':
        G = nx.Graph(A)
    else:
        G = A.copy()

    # if not nx.is_connected(G):
    GCC = nx.connected_component_subgraphs(G)
    GG = GCC[0]  # giant connected component

    n = GG.number_of_nodes()
    m = GG.number_of_edges()  # XXX onlyhalf matrix
    N = n * (n - 1.) / 2

    if weighted:
        raise NotImplementedError

    else:
        C = nx.algorithms.average_clustering(GG)
        L = nx.algorithms.average_shortest_path_length(GG)  # weighted=weighted)

        # create a random graph with same characteristics; sensu Humphries 2008 a Erdős-Rényi graph
        # determine p
        # m ~ p*n*(n-1)/2
        p = m / N

        C_rands = np.zeros(n_iter)
        L_rands = np.zeros(n_iter)
        for i in range(n_iter):
            # create random graph #XXX better double edge swap?
            R = nx.random_graphs.fast_gnp_random_graph(n, p)
            # R = nx.generators.random_graphs.erdos_renyi_graph(n,p)
            # XXX look at
            # networkx.generators.random_graphs.erdos_renyi_graph

            RCC = nx.connected_component_subgraphs(R)
            RR = RCC[0]  # giant connected component

            C_rands[i] = nx.algorithms.average_clustering(RR)
            L_rands[i] = nx.average_shortest_path_length(RR)  # weighted=False)
            i += 1

        # average C_rands & L_rands
        C_rand = np.mean(C_rands)
        L_rand = np.mean(L_rands)

    C_ratio = 0.
    if C_rand > 0:
        C_ratio = C / float(C_rand)

    L_ratio = 0.
    if L > 0 and L_rand > 0:
        L_ratio = (L / float(L_rand))

    S = 0.
    if L_ratio > 0:
        S = C_ratio / L_ratio

    return C_ratio, L_ratio, S


def small_world_scalar_faster(A, weighted=False, n_iter=100):
    r"""small-world scalar of the graph (faster version)

    .. math::

        S = \frac{\frac{C}{C_{rand}}}{\frac{L}{L_{rand}}}

    The small world scalar describes the ratio of the average clustering
    coefficient of a network (``C``) and the characteristic path length (``L``).
    The coefficients are normalized by a version obtained from a randomized
    network (``C_rand`` and ``L_rand``). The number of randomized networks is
    controlled by ``n_iter``.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted (default=False). #TODO not impl yet
    n_iter : integer
        Number of iterations/randomizations (default=100).

    Returns
    -------
    C_ratio : float
        Normalized clustering coefficient (C/C_rand).
    L_ratio : float
        Normalized characteristic path length (L/C_rand).
    S : float
        Small-world scalar.

    Notes
    -----
    A value of S >> 1 is regarded as "small world".

    References
    ----------
    .. [1] Watts & Strogatz (1998). Collective dynamics of “small-world”
           networks. Nature, 393(6684), 440–442. doi:10.1038/30918
    .. [2] Humphries & Gurney (2008). Network “small-world-ness”: a quantitative
           method for determining canonical network equivalence. PloS one, 3(4),
           e0002051. doi:10.1371/journal.pone.0002051

    Examples
    --------
    >>> from nigraph import karate_club, small_world_scalar_faster
    >>> A = karate_club()
    >>> C_ratio, L_ratio, S = small_world_scalar_faster(A)
    """

    # verbose = 1
    directed = False

    if weighted:
        raise NotImplementedError('weighted graphs are not supported yet')

    G = convert_to_graph(A, weighted=weighted, directed=directed, fmt='ig')

    L = G.average_path_length(directed=False)
    C = G.transitivity_avglocal_undirected()

    C_rands = np.zeros(n_iter)
    L_rands = np.zeros(n_iter)
    for i in range(n_iter):
        # if verbose:
        #     print '#iter:', i+1
        R = G.copy()
        R.rewire_edges(prob=1)

        L_rands[i] = R.average_path_length(directed=False)
        C_rands[i] = R.transitivity_avglocal_undirected()

    # average C_rands & L_rands
    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)

    C_ratio = 0.
    if C_rand > 0:
        C_ratio = C / float(C_rand)

    L_ratio = 0.
    if L > 0 and L_rand > 0:
        L_ratio = (L / float(L_rand))

    S = 0.
    if L_ratio > 0:
        S = C_ratio / L_ratio

    return C_ratio, L_ratio, S


def controllability(A, directed=True, perc=True):
    r"""returns the controllability of a directed network

    Controllability is defined as the amount of driver nodes :math:`N_{D}` in
    a network [1]_. Networks with fewer driver nodes are easier to controll
    than networks with more driver nodes.
    Driver nodes are nodes whose control is sufficient to fully control the
    system’s dynamics, i.e. to move it from its initial state to some desired
    final state in the state space.

    The number of driver nodes is determined mainly by the network’s degree
    distribution. Surprisingly, driver nodes tend to avoid high-degree nodes
    (network hubs).

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    directed : boolean
        The adjacency matrix is directed.
    perc : boolean
        Return driver nodes in percent of the total amount of network nodes
        (default=True).

    Returns
    -------
    value : float
        Amount of driver nodes in the graph.

    Notes
    -----
    Controllability is only defined for directed graphs. However, if the graph
    is not directed the metric is still computed.

    In [1]_ it was shown, that the minimum number of driver nodes needed to
    maintain full control of the network is determined by the maximum matching
    in the network, that is, the maximum set of links that do not share start
    or end nodes.

    Expanding on this work [2]_ showed that the the number of nodes to gain
    controll over a network can be well approximated without knowledge of the
    full degree distribution, but rather by the fraction of 'souce' and 'sink'
    nodes (i.e. nodes with zero in-degree, or zero out-degree). The authors
    show further that networks can be characterized by their control profiles.

    References
    ----------
    .. [1] Liu, Y.-Y., Slotine, J.-J., & Barabási, A.-L. (2011).
           Controllability of complex networks. Nature, 473(7346), 167–173.
           doi:10.1038/nature10011
    .. [2] Ruths, J., & Ruths, D. (2014). Control Profiles of Complex Networks.
           Science (New York, NY), 343(6177), 1373–1376.
           doi:10.1126/science.1242063

    Examples
    --------

    Determine the percentage of nodes that need to be controlled in order to
    gain full 'controll' over the neuronal network of Caenorhabditis elegans.

    >>> from nigraph import celegans, controllability, number_nodes
    >>> A = celegans()
    >>> print number_nodes(A)
    297
    >>> # see Table 1 in Liu et al. (0.165)
    >>> print controllability(A, perc=True)
    16.4983164983
    """

    # NB: the same can also be calculated with the zen library:
    # requires recent cython for compilation
    # import zen
    # import networkx as nx
    # G = zen.nx.from_networkx(nx.DiGraph(A))
    # nc = zen.control.num_min_controls(G)
    # nd = nc/float(G.num_nodes)

    # XXX take only connected GRAPHS!!!; later calc nD for seperatly for all
    # connected components

    # XXX return also link categories, see Fig 4
    # - critical
    # - redundant
    # - ordinary (??? don't get that definition)
    # this can be achieved using the attack.edge_importance function

    # create augmented adj (=bipartite representation with outnodes and innodes
    # on different layers)
    n_nodes = A.shape[0]

    adj_aug = np.zeros((n_nodes * 2, n_nodes * 2))
    idx = np.where(A != 0)
    adj_aug[idx[0], idx[1] + n_nodes] = A[idx]
    ADJ_AUG = make_undirected(adj_aug)

    A = convert_to_graph(ADJ_AUG, directed=directed, fmt='gt')

    # calc matching
    res = gt.max_cardinality_matching(A)

    # the matching is given by
    matching = np.asarray(res[0].a)

    # we are interested in UNMATCHED vertices (= driver nodes)
    # undirected:
    # - not incident to edges in the matching
    # directed:
    # - starting edge (!= ending) in the matching
    # "A vertex is matched if it is an ending vertex
    # of an edge in the matching. Otherwise, it is unmatched."

    # Definition: For a digraph, an edge subset M is a matching if no two edges
    # in M share a common starting vertex or a common ending vertex. A vertex is
    # matched if it is an ending vertex of an edge in the matching. Otherwise,
    # it is unmatched.

    idx_matching = np.where(matching == 1)[0]

    # get edge_list
    adj = gt.adjacency(A).todense()
    # this is the augmented adj, we are interested in n of the orig adj
    n_nodes = adj.shape[0] / 2

    # XXX recode this part, gt must offer a function to obtain an edgelist!
    idx = tril_indices(n_nodes * 2, -1)
    adj_aug_ = adj
    adj_aug_[idx] = 0  # set lower triangle to zero
    edge_list = np.asarray(np.where(adj_aug_ != 0)).T
    dim = edge_list.shape
    edge_list = edge_list.reshape(dim[0], dim[2])

    # potential_driver_nodes = np.unique(edge_list[idx_matching,0]) # only start-nodes
    # non_d_nodes = np.unique(edge_list[idx_matching,1]) - n_nodes # these are ending nodes; - n_nodes, because node_i and node_i+n_nodes are the same node
    #
    # # check if potential_driver_nodes are endpoints in the augmented adjacency matrix
    # d_nodes=[]
    # for i in potential_driver_nodes:
    #   if i not in non_d_nodes:
    #       d_nodes.append(i)

    # driver nodes = all unmatched nodes (including those not in the matching!)
    # these are ending nodes; - n_nodes, because node_i and node_i+n_nodes are
    # the same node
    matched_nodes = np.unique(edge_list[idx_matching, 1]) - n_nodes
    all_nodes = np.arange(n_nodes)
    driver_nodes = np.setdiff1d(all_nodes, matched_nodes)

    len_driver_nodes = driver_nodes.size
    if len_driver_nodes == 0:
        # in that case, per definition; graph can be controlled by a single,
        # random node
        len_driver_nodes = 1

    if perc:
        len_driver_nodes /= float(n_nodes)
        len_driver_nodes *= 100.

    # return len_driver_nodes, driver_nodes
    return len_driver_nodes


def rich_club_coefficients(ADJ, n_per=0, n_swap=1000, fmt='np'):
    """return rich-club coefficients for each node of the unweighted, undirected
    network

    The rich-club coefficient [1]_ is the ratio, for every degree k, of the
    number of actual to the number of potential edges for nodes
    with degree greater than k:

    .. math::

        \\phi(k) = \\frac{2 Ek}{Nk(Nk-1)}

    where `Nk` is the number of nodes with degree larger than `k`, and `Ek`
    be the number of edges among those nodes.

    Parameters
    ----------
    ADJ : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    n_per : integer
        Number of permutations. The distribution of phi random(k) yields a null
        distribution of rich club coefficients obtained from random topologies.
        Using this null distribution, phi(k) was assigned a P value by computing
        the percentage of random values that were found to be more extreme than
        the observed rich club coefficient.
    n_swap : integer
        Number of randomizations for the double edge-swap.
    fmt : string ['np'|'nx']
        Input format of the graph matrix.

    Returns
    -------
    rc : ndarray, shape(n, )
        Rich-club coefficient for every node n. If n_per > 0 the coefficient is
        normalized by a random graph.

    r : dict
        A dictionary, keyed by degree, with rich club coefficient values. If
        n_per > 0 the coefficient is normalized by a random graph.

    p : ndarray, shape(n, )
        Statistical p-values for each node. (only if n_per > 0). The p-value is
        calculated with a Monte-Carlo simuation with `n_per` permutations.

    Notes
    -----
    The p-value of the Monte-Carlo Simulation is calculated as in [3]_ ("more
    extreme" and nor "more extreme or equal" as one would expect from normal
    MCS testing). However this makes a big difference in the results.

    References
    ----------
    .. [1] Julian J. McAuley, Luciano da Fontoura Costa, and Tibério S. Caetano,
           "The rich-club phenomenon across complex network hierarchies",
           Applied Physics Letters Vol 91 Issue 8, August 2007.
           http://arxiv.org/abs/physics/0701290
    .. [2] van den Heuvel, M. P., & Sporns, O. (2011). Rich-Club Organization
           of the Human Connectome. Journal of Neuroscience, 31(44),
           15775–15786. doi:10.1523/JNEUROSCI.3539-11.2011
    .. [3] van den Heuvel, M. P., Kahn, R. S., Goñi, J., & Sporns, O. (2012).
           High-cost, high-capacity backbone for global brain communication.
           Proceedings of the National Academy of Sciences.
           doi:10.1073/pnas.1203593109

    Examples
    --------
    >>> from nigraph import get_random_graph, rich_club_coefficients
    >>> A = get_random_graph(fmt='np')
    >>> rc, r = rich_club_coefficients(A, n_per=0)
    >>> rc[0] # un-normalized rich-club coefficient
    #TODO
    >>> rc, r, p = rich_club_coefficients(A, n_per=1000)
    >>> rc[0] # here the rich-club coefficient is normalized
    #TODO
    """

    if fmt == 'np':
        G = nx.Graph(ADJ)
    else:
        G = ADJ.copy()

    n_edges = G.number_of_edges()
    k = np.asarray(G.degree().values())
    r = nx.rich_club_coefficient(G, normalized=False)

    rc = np.zeros(len(G))
    if n_per > 0:
        p = np.ones(len(G))
        pval = dict([(x, 1) for x in r.keys()])
        r_norm = dict([(x, 0) for x in r.keys()])

        for i in range(n_per):
            R = G.copy()
            nx.double_edge_swap(R, n_swap*n_edges, max_tries=n_swap*n_edges*10)
            r_rand = nx.rich_club_coefficient(R, normalized=False)
            for j in r.keys():
                r_norm[j] += r_rand[j]
                # in van Heuvel: "more extreme", but should be more
                # extreme or equal?!
                if r_rand[j] > r[j]:
                    pval[j] += 1

        for j in r.keys():
            # print r_norm[j], pval[j]
            r_norm[j] /= float(n_per)
            pval[j] /= float(n_per) + 1.
            # print r_norm[j], pval[j]

        # trans to rc vector for each node
        for i in r.keys():
            idx = np.where(k == i)[0]
            rc[idx] = r_norm[i]
            p[idx] = pval[i]
            r[i] = r[i] / r_norm[i]
        return rc, r, p
    else:
        # trans to rc vector for each node
        for i in r.keys():
            idx = np.where(k == i)[0]
            rc[idx] = r[i]
        return rc, r


def rich_club_coefficient(A, weighted=False, k_level_max=None):
    r"""return rich-club coefficient (rcc) for each node of the undirected
    network

    For unweighted networks, the rcc is the ratio, for every degree :math:`k`,
    of the number of actual to the number of potential edges for nodes with
    degree greater than :math:`k` [1]_:

    .. math::

        \phi(k) = \frac{2 E_k}{N_k(N_k-1)}

    where :math:`N_k` is the number of nodes with degree larger than :math:`k`,
    and :math:`E_k` is the number of edges among those nodes.

    For weighted networks, the rcc is calculated by ranking edges by weight,
    resulting in a vector :math:`w^\mathrm{ranked}`. Then, for each degree
    :math:`k`, the subset of nodes with degree larger than :math:`k` was
    selected. The number of links between the resulting subgraph is denoted
    :math:`E_{>k}` and their collective weight, i.e. the sum of the weights of
    the :math:`E_{>k}` edges, is written :math:`W_{>k}`.The rcc is then
    computed as the ratio between :math:`W_{>k}` and the sum of the weights of
    the strongest :math:`E_{>k}` connections of the whole graph, given by the
    top :math:`E_{>k}` number of connections of the collection of ranked
    connections in :math:`w^\mathrm{ranked}` [2]_:

    .. math::

        \phi(k) = \frac{W_{>k}}{\sum_{l=1}^{E_{>k}} w_l^\mathrm{ranked}}

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : bool
        The adjacency matrix is weighted (default=False). The implementation
        supports only non-negative weights.
    k_level_max : int | None
        Maximum level at which the rich club coefficient is calculated.
        Defaults to ``k_level_max=max(k)`` if ``None`` (default=``None``).

    Returns
    -------
    rcc : ndarray, shape(n, )
        Rich-club coefficient for every node n. If norm > 0 the coefficient is
        normalized by a rewired graph with equal degree distribution. rcc for
        nodes with a degree higher than ``k_level_max`` is equal to -1.

    r : ndarray, shape(k_level_max, )
        The rich-club curve, i.e. the rich club coefficient for all degree level

    See Also
    --------
    nt.graph.metrics.warehouse.metric_normalization: Normalized rich-club
        coefficient

    References
    ----------
    .. [1] Julian J. McAuley, Luciano da Fontoura Costa, and Tibério S. Caetano,
           "The rich-club phenomenon across complex network hierarchies",
           Applied Physics Letters Vol 91 Issue 8, August 2007.
           http://arxiv.org/abs/physics/0701290
    .. [2] van den Heuvel, M. P., & Sporns, O. (2011). Rich-Club Organization
           of the Human Connectome. Journal of Neuroscience, 31(44),
           15775–15786. doi:10.1523/JNEUROSCI.3539-11.2011
    .. [3] van den Heuvel, M. P., Kahn, R. S., Goñi, J., & Sporns, O. (2012).
           High-cost, high-capacity backbone for global brain communication.
           Proceedings of the National Academy of Sciences.
           doi:10.1073/pnas.1203593109

    Examples
    --------
    >>> from nigraph import macaque_cortical, rich_club_coefficient, cat_cortex
    >>> A = macaque_cortical()
    >>> rcc, r = rich_club_coefficient(A) # un-normalized rich-club coefficient

    >>> W = cat_cortex()
    >>> rcc, r = rich_club_coefficient(W, weighted=True)
    """

    # unweighted version is tested against BCT implementation rich_club_bu.m
    # TODO merge with function "rich_club_coefficients"

    if weighted:
        k = np.sum(A > 0, axis=0)  # degree
        if k_level_max is None:
            k_level_max = k.max()
        else:
            k_level_max = min(k.max(), k_level_max)

        Rk = np.zeros(k_level_max)
        for k_level in range(int(k_level_max)):
            idx = np.where(k > k_level+1)[0]
            SG = subgraph(A, idx)
            Nk = SG.shape[0]  # number of nodes with degree >k
            Ek = np.sum(SG > 0)  # number of connections in subgraph
            if Ek == 0:
                Rk[k_level] = 0.
                continue
            Wk = np.sum(SG)  # collective weight of edges in subgraph

            # order edges by strength
            idx = np.tril_indices_from(A, k=-1)
            W_sorted = np.sort(A[idx])  # ascending order; quicker

            # sum of Ek strongest edges
            sum_W_strongest = np.sum(W_sorted[-Ek:])
            Rk[k_level] = Wk / float(sum_W_strongest)

    else:
        k = np.sum(A, axis=0, dtype=np.int)  # degree
        if k_level_max is None:
            k_level_max = k.max()
        else:
            k_level_max = min(k.max(), k_level_max)

        Rk = np.zeros(k_level_max)
        for i in range(int(k_level_max)):
            idx = np.where(k > i+1)[0]
            SG = subgraph(A, idx)
            Nk = SG.shape[0]  # number of nodes with degree >k
            Ek = np.sum(SG)   # number of connections in subgraph
            nominator = Nk * (Nk - 1.)
            if nominator > 0:
                Rk[i] = Ek / nominator  # unweighted rich-club coefficient

    # Rk for nodes with k > k_level_max get rcc = -1
    _Rk = -1 * np.ones(k.max())
    _Rk[:len(Rk)] = Rk
    rcc = _Rk[k-1]

    return rcc, Rk


def vulnerability(A, weighted=False, auto_inv=True):
    """local vulnerability

    Vulnerability of a node is the change in the sum of distances (average
    shortes path length [ASPL]) between all node pairs when excluding that node.
    Nodes with high "importance/vulnerability" should have a bigger impact on
    the resulting distance than other nodes.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph
    weighted : boolean
        The adjacency matrix is weighted

    Returns
    -------
    v : ndarray, shape(n, )

    Notes
    -----
    The metric is sometimes also called `closeness_vitality`. With the removal
    of high degree nodes, the ASPL should increase. However, if the graph
    becomes unconnected the ASPL might descrease. #TODO it might be more
    meaningful to calculate the average closeness (1/d) instead of the ASPL.

    See Also
    --------
    node_importance

    Examples
    --------
    >>> from nigraph import get_random_graph, vulnerability
    >>> A = get_random_graph()
    >>> v = vulnerability(A, weighted=False)
    >>> e = efficiency_nodal(A)
    >>> from scipy.stats import pearsonr
    >>> r = pearsonr(v,e) # test relationship with efficiency_nodal
    """

    if weighted:
        if auto_inv:
            A = inverse_adj(A, method='inv')

    dist = avg_shortest_path_length(A, weighted=weighted, directed=False,
                                    auto_inv=False)
    n_nodes = A.shape[0]
    v = np.zeros(n_nodes)
    for i in range(n_nodes):
        # remove node i and calculate the overall distance
        G = nx.Graph(A)
        G.remove_node(i)
        # TODO should check for connected-ness and calc metric only in GCC
        d_i = avg_shortest_path_length(nx.to_numpy_matrix(G),
                                       weighted=weighted,
                                       directed=False, auto_inv=False)
        v[i] = dist - d_i
    return v


# def powerlaw_exponent(A, weighted=False, xmin=None):
#     r"""fitting a power-law distribution to the degree distribution of the graph
#
#     Mathematically, a quantity `x` obeys a power law if it is drawn from a
#     probability distribution
#
#     .. math::
#
#         p(x) \propto x^{- \alpha}
#
#     where :math:`\alpha` is a constant parameter of the distribution known as
#     the exponent or scaling parameter.
#
#     Parameters
#     ----------
#     A : ndarray, shape(n, n)
#         Adjacency matrix of the graph.
#     weighted : boolean
#         The adjacency matrix is weighted.
#     xmin : integer
#         Lower bound for fitting the power law. If ``xmin=None``, the smallest
#         degree value is used. This argument makes it possible to fit only the
#         tail of the distribution. If the powerlaw package is installed, ``xmin``
#         is estimated using the method described in [1]_.
#
#     Returns
#     -------
#     alpha : float
#         Estimated power-law exponent.
#
#     xmin : float
#         Lower bound for the power law.
#
#     Notes
#     -----
#     In practice, few empirical phenomena obey power laws for all values of `x`.
#     More often the power law applies only for values greater than some minimum
#     xmin. In such cases we say that the tail of the distribution follows a
#     power law [1]_.
#
#     See also
#     --------
#     nt.stats.math.powerlaw_fit: more statistics as the Loglikelihood ratio
#                                 and comparisons to other distributions
#
#     References
#     ----------
#     .. [1] Clauset, A., Shalizi, C., & Newman, M. (2009). Power-law
#            distributions in empirical data. SIAM review, 51(4), 661–703.
#     .. [2] http://en.wikipedia.org/wiki/Power_law
#
#     Examples
#     --------
#     >>> from nigraph import powerlaw_simple_and_connected
#     >>> A = powerlaw_simple_and_connected(10000, alpha=2)
#     >>> print powerlaw_exponent(A)
#     >>> (2.0298753310495976, 6)
#
#     >>> #TODO viz example: plot degree distribution on log-log plot
#     >>> # use cumulative distribution?
#     >>> #plt.loglog(np.sort(k)[::-1], 'k', marker='o', ls='')
#     """
#
#     # does not seem to work so well
#     # A = powerlaw_simple_and_connected(10000, alpha=1.8)
#     # In [10]: power_law_fitting(A)
#     # Out[10]: (2.0118497849185402, None)
#
#     k = degree(A)
#
#     if (externals_info()['powerlaw'] != -1) and (np.__version__ >= '1.6'):
#         # requires np > 1.6
#         import powerlaw as pwl
#         # results = pwl.distribution_fit(k, distribution='all')
#         # here xmin can be estimated using the clauset method
#         if weighted:
#             discrete = False
#         else:
#             discrete = True
#
#         results = pwl.Fit(k, xmin=xmin, discrete=discrete)
#         alpha = results.power_law.alpha
#         xmin = results.xmin
#         # R, p = results.loglikelihood_ratio('power_law', 'exponential')
#     else:
#         if weighted:
#             method = 'continuous'
#         else:
#             method = 'discrete_approx'
#
#         alpha = ig.power_law_fit(k, xmin=xmin, method=method)
#         xmin = k.min()
#     return alpha, xmin


def module_centrality(A, weighted=False, module=None, start_points=None,
                      end_points=None, auto_inv=True, impl='nx'):
    r"""centrality of a module/community

    The amount of shortest path of any two non-module nodes through the core.

    .. math::

        C = \sum_{s,t\in P} \frac{n^M_{st}}{g^P_{st}}

    where :math:`M` and :math:`P` are distinct modules of the graph.
    :math:`n^M_{st}` is the number of shortest path from :math:`s` to
    :math:`t \in P` that pass through :math:`M` and :math:`g^P_{st}` is the
    number of shortest paths from :math:`s` to :math:`t \in P`.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted.
    module : ndarray, shape(m, )
        Node ids of the module M. If ``module=None`` the core, derived from
        k-core decomposition is chosen.
    start_points : ndarray, shape(s, )
        Specifiy nodes from which the shortest path through the module are
        counted. If ``start_points=None`` all nodes not in ``module`` are
        regarded as start_points.
    end_points : ndarray, shape(e, )
        Specifiy nodes to which the shortest path through the module are
        counted. If ``end_points=None`` end_points are assumed to be the same
        as start_points.
    impl : string ['nx'|'ig']
        Shortest path implementation. Calculation on 'ig' is much faster.

    Returns
    -------
    v : float
        Module centrality.

    Notes
    -----
    With default parameters the function calculates the 'core centrality' [1]_,
    which is related to 'rich-club' centrality used in [2]_.

    References
    ----------
    .. [1] Ekman, M., Derrfuss, J., Tittgemeyer, M., & Fiebach, C. J. (2012).
           Predicting errors from reconfiguration patterns in human brain
           networks. Proceedings of the National Academy of Sciences, 109(41),
           16714–16719. doi:10.1073/pnas.1207523109
    .. [2] van den Heuvel, M. P., & Sporns, O. (2011). Rich-Club Organization
           of the Human Connectome. Journal of Neuroscience, 31(44),
           15775–15786. doi:10.1523/JNEUROSCI.3539-11.2011

    Examples
    --------
    >>> from nigraph import get_graph
    >>> A = get_graph(weighted=True)
    >>> m=np.arange(20)
    >>> print module_centrality(A, weighted=True, module=m)
    0.47058823529411764
    """

    if weighted:
        if auto_inv:
            A = inverse_adj(A, method='inv')

    n_nodes = A.shape[0]

    # chose core from k-core decomposition if module is None
    if module is None:

        if graph_type(A) is 'nx':
            M = np.asarray(nx.algorithms.find_cores(A).values())

        elif graph_type(A) == 'ig':
            M = np.asarray(A.coreness())

        elif graph_type(A) == 'np':
            G = convert_to_graph(A, fmt='ig')
            M = np.asarray(G.coreness())

        module = np.where(M == np.max(M))[0]  # k_max/core

    if start_points is None:
        tmp = np.zeros(n_nodes)
        tmp[module] = 1
        start_points = np.where(tmp == 0)[0]

    if end_points is None:
        end_points = start_points

    # n_nodes_module = len(module)
    n_path_p = 0.  # number of total shortest paths
    sp_path_counter = 0.  # number sp crossing the module

    if impl is 'nx':
        # pure nx implementation
        G = nx.Graph(A)

        for s in start_points:
            for e in end_points:
                if s != e:
                    if weighted:
                        # TODO inefficient, seems to ignore  target paramter?
                        _, b = nx.single_source_dijkstra(G, s, target=e, cutoff=None, weight='weight')
                        if e in b.keys():
                            path = b[e]
                            n_path_p += 1.
                            if len(path) > 2:
                                add_counter = 0.
                                c = 0
                                while (add_counter == 0.) and (c <= len(module)):
                                    for i in range(len(module)):
                                        if module[i] in path[1:-1]:  # exclude start/endpoints
                                            add_counter = 1.
                                        c += 1

                                sp_path_counter += add_counter
                    else:
                        raise NotImplementedError

    else:
        # ig based implementation - this handles weighted and unweighted paths
        # the versions nx vs ig don't match 100%, pobably due to differences
        # in the shortest path calculation

        A = convert_to_graph(A, weighted, fmt='ig')

        weights = None
        if weighted:
            weights = 'weight'  # = g.edge_attributes()

        for s in start_points:
            s_paths = A.get_shortest_paths(s, to=end_points, weights=weights,
                                           output="vpath")
            for path in s_paths:
                if len(path) > 2:
                    n_path_p += 1.
                    # path[1:-1] # exclude start/endpoints
                    if len(set(module).intersection(set(path[1:-1]))) > 0:
                        sp_path_counter += 1.

    # norm
    # norm = (n_nodes*(n_nodes - 1)) / 2. # all possible path in the network
    return sp_path_counter / n_path_p


def diversity_coefficient(A, partition=None):
    r"""diversity coefficient

    The diversity coefficient can be used to characterize how each node's
    connectivity is distributed across different modules.

    .. math::

        h_i = \frac{1}{\log m} \sum_{u \in M}{p_i(u) \log p_i (u)}

    where :math:`p_i(u) = \frac{s_i(u)}{s_i}`. :math:`s_i(u)` is the
    strength/degree of a node i in module u, and m is the number of modules in
    the partition M.

    nodes with high diversity coefficient *h* show a relatively even
    distribution of connectivity across all modules (i.e., they support
    functional integration between modules).

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted.

    Returns
    -------
    h : ndarray, shape(n,)
        diversity coefficient h

    Notes
    -----
    The metric was used in [1]_.

    See Also
    --------
    participation_coefficient: participation coefficient *p*

    References
    ----------
    .. [1] Fornito, A., Harrison, B. J., Zalesky, A., & Simons, J. S. (2012).
           Competitive and cooperative dynamics of large-scale brain functional
           networks supporting recollection. Proceedings of the National
           Academy of Sciences, 109(31), 12788–12793.
           doi:10.1073/pnas.1204185109

    Examples
    --------
    >>> from nigraph import karate_club, leading_eigenvector
    >>> from scipy.stats import pearsonr
    >>> A = karate_club()
    >>> n2c, _ = leading_eigenvector(A) # find a graph partition
    >>> h = diversity_coefficient(A, partition=n2c)

    >>> # compare with participation coefficient p
    >>> p = participation_coefficient(A, partition=n2c)
    >>> print pearsonr(h, p)[0]
    #
    """

    # assert partition != None, 'No graph partition specified'
    s = A.sum(axis=0)
    h = np.empty(A.shape[0])
    for u in np.unique(partition):
        # idx = np.where(partition != u)[0]
        # id = np.where(partition == u)[0]
        # ADJ = A.copy()
        # XXX not working correctly remove non module weights
        # ADJ[np.ix_(idx, idx)] = 0.
        # s_u = ADJ.sum(axis=0)
        # p_u = s_u[id] / s[id]
        # h[id] = p_u * np.log(p_u)

        id = np.where(partition == u)[0]
        # s_u = A[np.ix_(id, id)].sum(axis=0)
        p_u = A[np.ix_(id, id)].sum(axis=0) / s[id]
        h[id] = p_u * np.log(p_u)

    h /= np.log(np.unique(partition).size)
    return h


def google_matrix(A, alpha=0.85, personalization=None):
    """Google matrix of the graph

    Parameters
    -----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    alpha : float
      The damping factor.
    personalization : ndarray, shape(n, )
       The "personalization vector" consisting of a nonzero personalization
       value for each node.

    Returns
    -------
    M : ndarray, shape(n, n)
       Google matrix of the graph

    See Also
    --------
    pagerank

    Examples
    --------
    >>> from nigraph import get_graph
    >>> A = get_graph()
    >>> M = google_matrix(A, alpha=0.85)
    """

    # Based on implementation from NetworkX
    n_nodes = A.shape[0]

    A = np.matrix(A)

    # add constant to dangling nodes' row
    dangling = np.where(A.sum(axis=1) == 0)
    for d in dangling[0]:
        A[d] = 1.0/n_nodes

    # normalize
    A = A/A.sum(axis=1)

    # add "teleportation"/personalization
    e = np.ones((n_nodes))

    if personalization is not None:
        v = personalization
    else:
        v = e

    v = v/v.sum()
    P = alpha * A + (1 - alpha) * np.outer(e, v)
    return P


def eigenvector_centrality(A):
    r"""eigenvector centrality (EC) of the nodes in the graph

    TODO add math + more information, difference to DC

    The local EC for each node `i` is defined as

    .. math::

       EC(i) = \mu{1} = \frac{1}{\lambda{1}} ..

    This calculates the weighted or unweighted EC, based on the input adjacency
    matrix `A`.

    Parameters
    -----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.

    Returns
    -------
    v : ndarray, shape(n, )
        Local EC values.

    Notes
    ------
    This algorithm uses the SciPy sparse eigenvalue solver (ARPACK) to
    find the largest eigenvalue/eigenvector pair.

    For very large networks > 5000 nodes this is faster and more memory
    efficient than the corresponding igraph implementation (due to the overhead
    of converting to the igraph format).

    A comparison of ``eigenvector_centrality``, ``pagerank``, ``degree`` and
    ``subgraph_centrality`` was performed in [1]_.

    See Also
    --------
    pagerank: PageRank centrality (PC)
    degree: Degree centrality (DC)
    subgraph_centrality: Subgraph-centrality (SC)

    References
    ----------
    .. [1] Zuo, X.-N., Ehmke, R., Mennes, M., Imperati, D., Castellanos, F. X.,
           Sporns, O., & Milham, M. P. (2011). Network Centrality in the Human
           Functional Connectome. Cerebral cortex (New York, NY : 1991).
           doi:10.1093/cercor/bhr269

    Examples
    --------
    >>> from nigraph import cat_cortex
    >>> A = cat_cortex()
    >>> v = eigenvector_centrality(A)
    """

    # NB this is adopted from nx
    eigenvalue, eigenvector = linalg.eigs(sp.csr_matrix(A).T, k=1, which='LR')
    largest = eigenvector.flatten().real
    norm = scp.sign(largest.sum()) * scp.linalg.norm(largest)
    return largest/norm


def subgraph_centrality(A):
    r"""subgraph centrality (SC) of the nodes in the graph

    The subgraph centrality, also called communicability centrality [1]_, of a
    node `i` is a weighted sum of closed walks of different lengths in the
    network starting and ending at the node `i`.

    TODO add math

    .. math::

       SC(i) = \sum ...

    SC is only defined for *unweighted* graphs, the input adjacency matrix `A`
    must therefore be binary. Providing a weighted matrix will give erroneous
    results.

    Parameters
    -----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.

    Returns
    -------
    v : ndarray, shape(n, )
        Local SC values.

    Notes
    ------
    This algorithm uses the SciPy sparse eigenvalue solver (ARPACK) to
    find the largest eigenvalue/eigenvector pair.

    A comparison of ``eigenvector_centrality``, ``pagerank``, ``degree`` and
    ``subgraph_centrality`` was performed in [2]_. While these metrics appear
    to be generally consistent and identifed similar hub regions, the study also
    revealed some differences, e.g. ``DC`` revealed age-related differences in
    precuneus and posterior cingulate regions, but ``EC`` did not.

    ``SC`` particularly emphasizes the centrality of parietal and frontal
    regions commonly implicated in attentional control. Furthermore, the
    centrality of the posterior IFG, a key region implicated in top down control
    of attention was only appreciated by ``SC`` [2]_.

    See Also
    --------
    pagerank: PageRank centrality (PC)
    degree: Degree centrality (DC)
    eigenvector_centrality: Eigenvector centrality (EC)
    resolvent_centrality: Version of subgraph centrality (SC) better suited for
        larger networks

    References
    ----------
    .. [1] Estrada, E., & Hatano, N. (2008). Communicability in complex
           networks. Physical Review E, Statistical, Nonlinear, and Soft Matter
           Physics, 77(3 Pt 2), 036111.
    .. [2] Zuo, X.-N., Ehmke, R., Mennes, M., Imperati, D., Castellanos, F. X.,
           Sporns, O., & Milham, M. P. (2011). Network Centrality in the Human
           Functional Connectome. Cerebral cortex (New York, NY : 1991).
           doi:10.1093/cercor/bhr269

    Examples
    --------
    >>> from nigraph import cat_cortex
    >>> A = cat_cortex()
    >>> A[A!=0] = 1 # make unweighted
    >>> v = subgraph_centrality(A)
    """

    # XXX untested; transpose A?; check against BCT impl.

    # TODO convert to scipy matrix for better performance?
    # TODO Zuo mentions a modification for large networks that otherwise might
    # give problems with machine accuracy
    eigenvalue, eigenvector = np.linalg.eigh(A)
    return np.dot(eigenvector**2, np.exp(eigenvalue))


def resolvent_centrality(A):
    r"""resolvent centrality/subgraph centrality (RC) of the nodes in the graph

    The subgraph centrality (SC), also called communicability centrality [1]_,
    of a node `i` is a weighted sum of closed walks of different lengths in the
    network starting and ending at the node `i`.

    This general version of SC, also called resolvent centrality, works better
    with larger networks than the classical subgraph centrality, which runs
    into problems with machine accuracy and can return subgraph walks with
    infinite length.

    TODO add math

    .. math::

       SC(i) = \sum ...

    RC is only defined for *unweighted* graphs, the input adjacency matrix `A`
    must therefore be binary. Providing a weighted matrix will result in
    erroneous results.

    Parameters
    -----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.

    Returns
    -------
    v : ndarray, shape(n, )
        Local RC values.

    Notes
    ------
    This is the modified version of SC used in [1]_.

    This algorithm uses the SciPy sparse eigenvalue solver (ARPACK) to
    find the largest eigenvalue/eigenvector pair.

    See Also
    --------
    pagerank: PageRank centrality (PC)
    degree: Degree centrality (DC)
    eigenvector_centrality: Eigenvector centrality (EC)
    subgraph_centrality: Subgraph centrality (SC)

    References
    ----------
    .. [1] Zuo, X.-N., Ehmke, R., Mennes, M., Imperati, D., Castellanos, F. X.,
           Sporns, O., & Milham, M. P. (2011). Network Centrality in the Human
           Functional Connectome. Cerebral cortex (New York, NY : 1991).
           doi:10.1093/cercor/bhr269

    Examples
    --------
    >>> from nigraph import cat_cortex
    >>> A = cat_cortex()
    >>> A[A!=0] = 1 # make unweighted
    >>> v = resolvent_centrality(A)
    """

    # TODO: add to warehouse
    # TODO: taking only the first 20 eigenvalues seems quite arbitrary to me
    # check original paper to see what's behind that
    n_nodes = A.shape[0]
    eigenvalue, eigenvector = np.linalg.eigh(A)
    return np.dot(eigenvector[:, 0:20]**2, (n_nodes - 1./(n_nodes-1-eigenvalue[0:20])))


def pagerank(A, alpha=0.85, personalization=None):
    """PageRank centrality (PC) of the nodes in the graph

    PageRank(TM) computes a ranking of the nodes in the network A based on the
    structure of the incoming links. It was originally designed as an algorithm
    to rank web pages.

    Parameters
    -----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    alpha : float, optional
        Damping parameter for PageRank (default=0.85).
    personalization : ndarray, shape(n, )
        The "personalization vector" consisting of a nonzero personalization
        value for each node.

    Returns
    -------
    pagerank : ndarray, shape(n, )
        Local PageRank values.

    Notes
    -----
    The eigenvector calculation uses NumPy's interface to the LAPACK
    eigenvalue solvers. This will be the fastest and most accurate for small
    graphs.

    See Also
    --------
    subgraph_centrality: Subgraph-centrality (SC)
    degree: Degree centrality (DC)
    eigenvector_centrality: Eigenvector centrality (EC)

    References
    ----------
    .. [1] A. Langville and C. Meyer,
           "A survey of eigenvector methods of web information retrieval."
           http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
           The PageRank citation ranking: Bringing order to the Web. 1999
           http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf

    Examples
    --------
    >>> from nigraph import get_graph
    >>> A = get_graph()
    >>> p = pagerank(A, alpha=0.85)
    """

    # Based on implementation from NetworkX
    M = google_matrix(A, alpha, personalization=personalization)

    # use numpy LAPACK solver
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = eigenvalues.argsort()

    # eigenvector of largest eigenvalue at ind[-1], normalized
    largest = np.array(eigenvectors[:, ind[-1]]).flatten().real
    norm = float(largest.sum())

    # centrality=dict(zip(nodelist,map(float,largest/norm)))
    return largest/norm


def spread_of_infection(A, weighted=False, n_steps=100, initial_infected=None,
                        p_travel=1.0, n_iter=10, seed=None,
                        verbose=False, return_n_iter_data=False):
    r"""simulates a spread-of-infection (SI) model

    Simulates a Reed-Frost type spread-of-infection model on a graph A. Each
    node can be in either of two states: Susceptible (not infected) or
    Infected. Each infected individual in generation t (t = 1,2,...)
    independently infects each susceptible node to which it is connected. When
    a node becomes infected, it is removed from the Susceptible class and added
    to the Infected class.

    At each time step, neighbours of infected nodes are selected to also
    become infected. If ``p_travel < 1``, an independent Bernoulli trial is
    carried out for each edge from an infected node to a susceptible node,
    with chance of success equal to ``p_travel``. The disease only spreads via
    this edge if the trial is successful.

    Edge weights in the network are interpreted the same. For every infected
    node i and susceptible node j, the existence of an edge
    :math:`i\xrightarrow{p}j` means that the probability of j being contacted
    by i (the infection spreading to j) in each timestep is equal to p. The
    probability of success in the Bernoulli trial for each edge is thus
    multiplied by the edge weight. Because edge weights are interpreted as
    probabilities, this requires all edge weights to be in the interval [0,1].

    The function determines how long it takes for a particular node to become
    infected. If node i becomes infected at time t, ``visit_time[i] = t``.
    Nodes can only become infected once. If ``n_steps`` is set, the number of
    timesteps is limited to this value.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph. If ``weighted=True``, 0 <= A[i,j] <= 1
        for all i, j.
    weighted : bool
        The adjacency matrix is weighted (default=False).
    n_steps : int
        Maximum number of timesteps.
    initial_infected : int | list | None
        Initially infected node(s). If ``None`` one random node is chosen.
    p_travel : float [0..1]
        Baseline probability for neighbour of infected node to also become
        infected.
    seed : int | None
        Random seed (default=None).
    n_iter : int
        Number of interations. The simulation should be repeated many times
        to get a reliable estimate (default=10).

    Notes
    -----
    A visualization of a random walker can be found here [4]_. In general one
    could define the infection model based on edges instead of nodes.

    The adjacency matrix ``A`` must be connecetd, otherwise the results are
    not menaingful. This function does not check for connectedness.

    Returns
    -------
    visit_time : ndarray, shape(n, )
        Spread of disease over time. ``visit_time[i]`` indicates the timepoint
        at which node ``i`` was visited for the first time. Note that
        ``visit_time[initial_infected]=0``. Average over iterations if
        ``n_iter>0``.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Epidemic_model
    .. [2] Deijfen, M. (2011) Epidemics and vaccination on weighted graphs.
           Math Biosci 232(1) 57-65. doi:10.1016/j.mbs.2011.04.003
    .. [3] Daudert B., Li B-L. Spreading of infectious diseases on complex
           networks with non-symmetric transmission probabilities.
           eprint arXiv:math/0611730
    .. [4] http://www.mapequation.org/apps/MapDemo.html

    Examples
    --------
    >>> from nigraph import spread_of_infection
    >>> import matplotlib.pylab as plt
    >>> A = np.array([[ 0. ,  1. ,  0. ,  0. ,  1. ,  0. ],
                      [ 1. ,  0. ,  1. ,  0. ,  0. ,  0. ],
                      [ 0. ,  1. ,  0. ,  1. ,  0. ,  0. ],
                      [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
                      [ 1. ,  0. ,  0. ,  0. ,  0. ,  0.5],
                      [ 0. ,  0. ,  0. ,  0. ,  0.5,  0. ]])
    >>> visit_time = spread_of_infection(A, weighted=True, initial_infected=0, p_travel=0.1, seed=3)
    >>> print visit_time
    [  0.    8.1  23.9  33.6  11.6  34.7]

    Plot percentage of visited nodes over time

    >>> time = np.arange(np.ceil(np.amax(visit_time) + 1))
    >>> cum_time = np.array([np.where(visit_time < time[i])[0].shape[0] for i in range(len(time))]) / float(A.shape[0])
    >>> plt.plot(time, cum_time)

    In a slightyl more complex example let's simulate the spread of information
    in a functional brain network and investiagte whether the infection time
    ``T`` more strongly correleates with the *spatial*, or *topological*
    distance in the network.

    First, we generate our functional network based on the AAL template.

    >>> from nigraph import aal, get_fmri_rss_data, load_roi_mri, adj_static, thresholding_prop, dist_matrix_spatial
    >>> _, mpath, labels, coords = aal(n=116, space='3mm')
    >>> _, fpath = get_fmri_rss_data()
    >>> ts = load_roi_mri(fpath, mpath)
    >>> A = adj_static(ts)
    >>> A[A<0]=0
    >>> A = thresholding_prop(A, 0.2)

    Then, we can simulate the spread of information from the left Precuneus.

    >>> node_id = 66 # left precuneus
    >>> T = spread_of_infection(A, weighted=True, initial_infected=node_id, p_travel=0.01, seed=3)
    >>> time = np.arange(np.ceil(np.amax(T) + 1))
    >>> cum_time = np.array([np.where(T < time[i])[0].shape[0] for i in range(len(time))]) / float(A.shape[0])
    >>> plt.plot(time, cum_time)

    Now, we calculate the *spatial* and *topological* distance from the
    Precuneus node to all other nodes in the network.

    >>> Ds = dist_matrix_spatial(coords)
    >>> d_spatial = Ds[node_id,:]

    >>> Dt = dist_matrix_topological(A, weighted=1)
    >>> d_topo = Dt[node_id,:]

    Finally, we can compare the correlations of *spatial* and *topological*
    distance with the infection time ``T``.

    >>> from scipy.stats import spearmanr
    >>> corr_spatial = spearmanr(d_spatial, T)[0]
    >>> corr_topo = spearmanr(d_topo, T)[0]
    >>> ratio = corr_topo / corr_spatial
    >>> print ratio
    3.59382401965

    The results indicate that the spread of information depends about 3.5 times
    more on the *topological* than the *spatial* distance in the network.
    """

    def _run_network():
        # initially infected population
        if initial_infected is None:
            # pick random starting node
            infected = set([np.random.randint(0, n_nodes)])
        else:
            infected = set([initial_infected])

        # main loop
        visit_time = -1 * np.ones(n_nodes)
        visit_time[initial_infected] = 0
        for i in range(n_steps):
            for source in infected:
                # print '*** ' + str(i)
                # find all uninfected neighbours and
                # select neighbours for infection
                neighbours = set(np.unique(np.where(A[source, :] != 0)[-1]))
                neighbours -= infected

                _neighbours = set()
                for neig in neighbours:
                    p = float(p_travel)
                    if weighted:
                        p *= A[source, neig]    # multiply by edge weight
                    # weight is interpreted as P(success) in a Bernoulli trial
                    if binomial(n=1, p=p) == 1:
                        _neighbours.add(neig)
                neighbours = _neighbours

                # print 'Neighbours to be infected: ' + str(neighbours) + '\n'

                # infect neighbours
                visit_time[list(neighbours)] = i + 1  # mark nodes as visited
                infected = infected.union(neighbours)

            if len(infected) == n_nodes:
                # all nodes are infected
                break

        return visit_time

    # disable this check to speed up calculation for larger networks
    # assert is_connected(A), 'graph not connected'
    assert 0 <= p_travel and p_travel <= 1

    assert n_steps > 0
    if weighted:
        # weights are probabilities
        assert np.amax(A) <= 1.
        assert np.amin(A) >= 0.

    if seed is not None:
        np.random.seed(seed)

    n_nodes = A.shape[0]

    if verbose:
        t0 = time.time()

    # in case one needs e.g. the variance of the infection model
    if return_n_iter_data:
        visit_times = np.zeros((n_iter, n_nodes))
        for iter in range(n_iter):
            if verbose:
                print ' #iter %s // %s' % (iter + 1, n_iter)
            visit_time = _run_network()
            visit_times[iter] = visit_time

        if verbose:
            print '--time', (time.time()-t0)/60.

        return np.mean(visit_times, axis=0), visit_times

    else:
        visit_times = np.zeros(n_nodes)
        for iter in range(n_iter):
            if verbose:
                print ' #iter %s // %s' % (iter + 1, n_iter)
            visit_time = _run_network()
            if visit_time[visit_time < 0].size > 0:
                print 'warning: not all nodes were visited, \
                       consider increasing n_steps'
            visit_times += visit_time

        visit_times /= float(n_iter)

        if verbose:
            print '--time', (time.time()-t0)/60.

        return visit_times
