#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Commonly used utility functions."""

# mainly backports from future numpy here

from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import igraph as ig
import networkx as nx
from copy import deepcopy
import tempfile

# __all__ = ["mask_indices", "triu_indices", "tril_indices"]


def is_weighted(A):
    """tests whether the adjacency matrix is weighted

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph

    Returns
    -------
    v : boolean
        The adjacency matrix is weighted.

    Examples
    --------
    >>> A = get_random_graph(weighted=True)
    >>> print is_weighted(A)
    True
    >>> print is_weighted(get_random_graph(weighted=False))
    False
    """
    return np.unique(A).size > 2


def is_directed(A):
    """tests whether the adjacency matrix is directed

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.

    Returns
    -------
    v : boolean
        The adjacency matrix is directed.

    Examples
    --------
    >>> A = get_random_graph(directed=True)
    >>> print is_directed(A)
    True
    >>> print is_weighted(get_random_graph(directed=False))
    False
    """
    d = True
    if (A.transpose(1, 0) == A).all():
        d = False
    return d


def is_dynamic_graph(A):
    """tests whether the graph is a dynamic graph

    Parameters
    ----------
    A : ndarray, shape(n, n) **or** (n_tps, n, n) **or** (n_tps, n_edges)
        Adjacency matrix of the graph.

    Returns
    -------
    v : boolean
        The graph is a dynamic graph.

    Examples
    --------
    >>> DG = get_dynamic_graph(weighted=True, reduce=False)
    >>> is_dynamic_graph(DG)
    True
    >>> A = get_graph()
    >>> is_dynamic_graph(A)
    False
    """

    if len(A.shape) == 3:
        DG = True
    elif A.shape[0] == A.shape[1]:
        DG = False
    else:
        DG = True
    return DG


def make_unweighted(A, copy=False):
    """converts a weighted to an unweigted adjacency matrix

    All edge weights different from zero (including negative weights) will
    become an edge with weight 1.

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    copy : boolean
        create a copy of the data (default=False). Otherwise the data is
        changed in-place.

    Returns
    -------
    UW : ndarray, shape(n, n)
        Unweighted adjacency matrix of the graph.

    Notes
    -----
    Supports also **dynamic graphs**.

    Examples
    --------
    >>> A = get_random_graph(weighted=True)
    >>> is_weighted(A)
    True
    >>> A = make_unweighted(A)
    >>> is_weighted(A)
    False
    """
    if copy:
        A = deepcopy(A)
    A[np.abs(A) > 0.] = 1.
    return A


def fill_diagonal(A, val):
    """fill diagonal of an adjacency matrix

    Populate the diagonal with a specified value

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.

    val : float
      Value to be written on the diagonal, its type must be compatible with
      that of the array a.

    Notes
    -----
    The adjacency matrix is **changed in-place**.

    Examples
    --------
    >>> A = get_graph()
    >>> print A[0,0]
    1.0
    >>> fill_diagonal(A, 0.) # A is changed in-place(!)
    >>> print A[0,0]
    0.0
    """

    # code from nitime: http://nipy.sourceforge.net/nitime/
    # This functionality can be obtained via ``np.diag_indices``,but internally
    # this version uses a much faster implementation that never constructs the
    # indices and uses simple slicing.
    step = A.shape[1] + 1

    # Write the value out into the diagonal.
    A.flat[::step] = val


def mask_indices(n, mask_func, k=0):
    """Return the indices to access (n,n) arrays, given a masking function"""
    m = np.ones((n, n), int)
    a = mask_func(m, k)
    return np.where(a != 0)


def triu_indices(n, k=0):
    """Return the indices for the upper-triangle of an (n,n) array.

    Parameters
    ----------
    n : integer
        Number of nodes
    k : integer
        Diagonal offset

    Returns
    -------
    val : list, shape(2, )
        two ndarray's with the cell ids

    Examples
    --------
    >>> A = np.zeros((3,3))
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]]
    >>> idx=triu_indices(3,0)
    >>> A[idx] = 1
    array([[ 1.,  1.,  1.],
           [ 0.,  1.,  1.],
           [ 0.,  0.,  1.]])

    >>> A = np.zeros((3,3))
    >>> idx=triu_indices(3,1)
    >>> A[idx] = 1
    array([[ 0.,  1.,  1.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  0.]])
    """

    return mask_indices(n, np.triu, k)


def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    Parameters
    ----------
    n : integer
        Number of nodes
    k : integer
        Diagonal offset

    Returns
    -------
    val : list, shape(2, )
        two ndarray's with the cell ids
    """
    return mask_indices(n, np.tril, k)


def edges(A, nodes=None):
    """returns the edges of a graph

    Parameters
    ----------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    nodes : ndarray, shape(i, )
        return only edges for the given nodes. Per default all edges of the
        graph are returned.

    Returns
    -------
    el : ndarray, shape(e_i, e_j)
        Edge list.
    w : ndarray, shape(n_edges, )
        Edge weights.

    Notes
    -----
    For directed graphs, the edges are returned for the upper and lower
    triangle, i.e. :math:`e_{ij}` and :math:`e_{ji}`. For undirected graphs
    :math:`e_{ij}` is returned.

    Examples
    --------
    >>> A = get_graph(weighted=False)
    >>> el, weights = edges(A)
    >>> print weights[0:5] # first 5 weights
    [ 1.  1.  1.  1.  1. ]

    >>> A = get_graph(weighted=True)
    >>> el, weights = edges(A)
    >>> print weights[0:5] # first 5 weights
    [ 1., 0.55037578, 0.79052192, 0.55037578, 1. ])

    >>> # get only edges of a certain community
    >>> A = get_graph(weighted=False)
    >>> n2c, extras = ward(A, n_partitions=2) # 2 communities
    >>> print n2c
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1])
    >>> edge_list, w = edges(A, np.where(n2c == 1)[0])
    """

    ADJ = deepcopy(A)
    if not is_directed(ADJ):
        idx = triu_indices(ADJ.shape[0], k=0)
        ADJ[idx] = 0

    if nodes is not None:
        T = np.zeros((ADJ.shape[0], ADJ.shape[0]))
        T[np.ix_(nodes, nodes)] = 1.
        ADJ = ADJ * T  # zero out all other nodes

    el = np.where(ADJ != 0)
    w = np.asarray(ADJ[el])
    return el, w.flatten()


def graph_type(G):
    """returns the graph type

    Parameters
    ----------
    G : graph object
        Representation of the adjacency matrix.

    Returns
    -------
    gtype : string ['unknown'|'nx'|'ig'|'np'|'bct'|'mat'|'edge_list']
        Graph format.

    See Also
    --------
    convert_to_graph: Convert a numpy array into a graph object

    Examples
    --------
    >>> A = get_random_graph(fmt='np')
    >>> graph_type(A)
    'np'
    >>> A = get_random_graph(fmt='ig')
    >>> graph_type(A)
    'ig'
    """
    import graph_tool.all as gt

    gtype = 'unknown'
    if type(G) is nx.Graph:
        gtype = 'nx'
    elif type(G) is np.matrix:
        gtype = 'np'
    elif isinstance(G, np.ndarray):
        # NB this will also work for memmap arrays
        gtype = 'np'
    elif type(G) is ig.Graph:
        gtype = 'ig'
    elif type(G) is gt.Graph:
        gtype = 'gt'
    elif type(G) is str:
        # TODO this is not a strong indication for mat!
        gtype = 'mat'
    elif type(G) is list:
        gtype = 'edge_list'
    # elif type(G) is list:
    #     gtype = 'bct'

    return gtype


def convert_to_graph(A, weighted=False, directed=False, fmt='nx',
                     struc_array_name=None, rm_self_loops=True):
    """converts a numpy array into a graph object

    Parameters
    ----------
    A : ndarray, shape(n, n) **or** string if input is a ``.mat`` file
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted.
    directed : boolean
        The adjacency matrix is directed.
    fmt : string ['nx'|'ig'|'gt'|'gt*'|'bct'|'mat'|'snp']
        Output format

        * ``nx``: networkx.Graph
        * ``ig``: igraph.Graph
        * ``gt``: graph_tool.Graph
        * ``gt*``: graph_tool.Graph conversion is done using graphml file
        * ``bct``: Brain connectivity toolbox (BCT) graph (C++ version)
        * ``mat``: matlab file
        * ``snp``: sparse numpy graph (scipy lil_matrix)

    rm_self_loops : boolean
        Remove self-loops (default=True).

    Returns
    -------
    g : graph
        Graph object in the specified format.

    Notes
    -----
    In case the adjacency matrix A is weighted and ``weighted=False`` the
    resulting graph ``g`` will be unweighted.

    Examples
    --------
    >>> A = get_random_graph(fmt='np')
    >>> graph_type(A)
    'np'

    >>> G = convert_to_graph(A, fmt='ig')
    >>> graph_type(G)
    'ig'

    >>> G = convert_to_graph(A, fmt='nx')
    >>> graph_type(G)
    'nx'

    >>> # convert .mat into graph format. In this case the input is a string
    >>> G = convert_to_graph('A.mat', fmt='ig')
    >>> graph_type(G)
    'mat'

    >>> # a graph can also be saved to a .mat file
    >>> G = convert_to_graph(A, fmt='mat')
    >>> print G
    'adj.mat' # the graph is written to disk

    .. warning::

        The parameter ``directed=True`` needs more testing for ``fmt=ig``
        and ``fmt=gt``

    """
    if type(A) is str:
        A_ = sio.loadmat(A)
        if struc_array_name is None:
            # TODO is there a better way to infer this?
            struc_array_name = A_.keys()[0]
        A = A_[struc_array_name]
        A = np.asarray(A, dtype=np.float64)

    # remove self-loops
    if rm_self_loops:
        fill_diagonal(A, 0)

    if fmt in ['nx', 'gt*']:
        if not weighted:
            if is_weighted(A):
                A = make_unweighted(A, copy=True)

        if directed:
            g = nx.DiGraph(A)
        else:
            g = nx.Graph(A)

        if fmt == 'gt*':
            import graph_tool.all as gt
            # convert to nx.Graph, save as xml and load in graph_tool
            # is faster than np -> graph_tool
            graph_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml',
                                                     dir='./', delete=True)
            nx.write_graphml(g, graph_file)
            graph_file.flush()
            # in gt 2.2.15: file_format='xml', later fmt='xml'
            g = gt.load_graph(graph_file.name)
            graph_file.close()

    elif fmt == 'gt':

        # gt is faster with a dense! graph
        # A=get_random_graph(5000)
        # %timeit g=convert_to_graph(A, fmt='gt')
        # 1 loops, best of 3: 3.48 s per loop
        # %timeit g=convert_to_graph(A, fmt='gt*')
        # 1 loops, best of 3: 5.96 s per loop

        # # but gt* is faster with a dense graph
        # ts=np.random.randn(500,250) # for n=1000 gt* 50% (!) faster
        # A=adj_static(ts)
        # A[A<0]=0.0
        # %timeit g=convert_to_graph(A, fmt='gt*')
        # 1 loops, best of 3: 7.86 s per loop
        # %timeit g=convert_to_graph(A, fmt='gt')
        # 1 loops, best of 3: 11.6 s per loop

        n_nodes = A.shape[0]
        g = gt.Graph(directed=directed)
        g.add_vertex(n_nodes)

        # in recent git version this is now possible
        # and should provide a huge speed-up
        # g.add_edge_list(transpose(nonzero(a)))

        for i in xrange(n_nodes):
            for j in xrange(i + 1, n_nodes):
                if A[i, j]:
                    g.add_edge(g.vertex(i), g.vertex(j))

        if weighted:
            edge_weights = g.new_edge_property("double")

            # BUG: something wrong here
            # XXX maybe gt assumes to get the upper not lower triangle?

            edge_list, w = edges(A)
            edge_weights.a = w
            g.edge_properties["weight"] = edge_weights

    # elif fmt == 'gt**':
    #
    #     n_nodes = A.shape[0]
    #     g = gt.Graph(directed=directed)
    #     _ = g.add_vertex(n_nodes)
    #
    #     for i in xrange(n_nodes):
    #         neig_i = np.where(A[i, :] != 0)[0]
    #         for u, w in combinations(neig_i, 2):
    #             g.add_edge(g.vertex(u), g.vertex(w))

    elif fmt == 'ig':
        if directed:
            raise NotImplementedError

        if is_dynamic_graph(A):
            g = []

            n_tps = A.shape[0]
            for i in range(n_tps):
                e = ig.Graph(0)
                if weighted:
                    g_ = e.Weighted_Adjacency(A[i].tolist(), mode=ig.ADJ_MAX)
                else:
                    g_ = e.Adjacency(A[i].tolist(), mode=ig.ADJ_MAX)
                g.append(g_)

        else:
            e = ig.Graph(0)
            if weighted:
                g = e.Weighted_Adjacency(A.tolist(), mode=ig.ADJ_MAX)
            else:
                g = e.Adjacency(A.tolist(), mode=ig.ADJ_MAX)

    elif fmt == 'ig*':
        # ig is much slower, but slighty better in term sof memory consumption
        # A=get_random_graph(5000)
        # In [8]: %timeit G = convert_to_graph(A, fmt='ig')
        # 1 loops, best of 3: 8.14 s per loop
        #
        # In [9]: %timeit G = convert_to_graph(A, fmt='ig*')
        # 1 loops, best of 3: 1.54 s per loop
        #
        # In [10]: %memit G = convert_to_graph(A, fmt='ig*')
        # maximum of 1: 1423.792969 MB per loop
        #
        # In [11]: %memit G = convert_to_graph(A, fmt='ig')
        # maximum of 1: 1253.480469 MB per loop

        if directed:
            raise NotImplementedError

        # if weighted:
        #     # only unweighted so far
        #     raise NotImplementedError

        # this might be more memory efficient for larger graphs see
        # http://lists.gnu.org/archive/html/igraph-help/2009-11/msg00213.html
        # edgelist = []
        # for v1, row in enumerate(A):
        #   edgelist.extend( [v2 for v2, element in row if element > 0])
        # g = ig.Graph(edgelist, directed=False)

        idx, idy = np.where(np.tril(A, -1) != 0)
        edgelist = zip(idx.tolist(), idy.tolist())
        g = ig.Graph(edgelist, directed=False)

        if weighted:
            # weights = g.es["weight"]
            g.es["weight"] = A[idx, idy]

        # faster?, more memory efficient?
        # idx, idy = np.where(A != 0)
        # edgelist = zip(idx.tolist(), idy.tolist())
        # g = ig.Graph(edgelist, directed=True)

        # convert back
        # M = np.asarray(g.get_adjacency().data)

    elif fmt == 'bct':
        g = A.tolist()

    elif fmt is 'mat':
        sio.savemat('adj.mat', {'ADJ': A})
        g = 'adj.mat'

    elif fmt is 'nt':
        # shouldn't happen
        g = A.copy()

    elif fmt is 'snp':
        g = sp.lil_matrix(A)

    else:
        raise ValueError('Specified format is not known')

    return g
