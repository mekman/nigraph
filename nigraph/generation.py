#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Commonly used utility functions."""

# mainly backports from future numpy here

from __future__ import absolute_import, division, print_function
import numpy as np
import networkx as nx

from ..utilities import tril_indices, triu_indices, convert_to_graph


def get_random_graph(n=30, weighted=False, directed=False, fmt='np'):
    """returns a random graph for testing purpose

    Parameters
    ----------
    n : integer
        Number of nodes.
    weighted : boolean
        The adjacency matrix is weighted.
    directed : boolean
        The adjacency matrix is directed.
    fmt : string ['nx'|'ig'|'gt'|'bct'|'mat'|'snp']
        Output format

        * ``nx``: networkx.Graph
        * ``ig``: igraph.Graph
        * ``gt``: graph_tool.Graph
        * ``bct``: Brain connectivity toolbox (BCT) graph (C++ version)
        * ``mat``: matlab file
        * ``snp``: sparse numpy graph (scipy lil_matrix)

    Returns
    -------
    graphs : graph object

    Examples
    --------
    >>> A = get_random_graph(30, directed=True)
    >>> is_directed(A)
    True
    """

    G = nx.random_graphs.watts_strogatz_graph(n, np.int(n * 3 / float(n)), 0.2)
    ADJ = np.asarray(nx.to_numpy_matrix(G))

    if directed:
        idx = tril_indices(n, -1)
        edge_info_permutation = np.random.permutation(ADJ[idx])
        ADJ[idx] = edge_info_permutation

    if weighted:
        # idx = tril_indices(n, -1)
        idy = triu_indices(n, 0)
        id = np.where(ADJ == 0)
        weights = np.random.rand(n, n)
        weights[idy] = 0.  # zero upper triangle
        weights[id] = 0.
        ADJ = weights + weights.T

    if fmt is not 'np':
        ADJ = convert_to_graph(ADJ, weighted=weighted, directed=directed,
                               fmt=fmt)
    return ADJ
