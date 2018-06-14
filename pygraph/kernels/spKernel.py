"""
@author: linlin
@references: Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
"""

import sys
import pathlib
sys.path.insert(0, "../")
from tqdm import tqdm
import time

import networkx as nx
import numpy as np

from pygraph.utils.utils import getSPGraph
from pygraph.utils.graphdataset import get_dataset_attributes


def spkernel(*args, node_label='atom', edge_weight=None, node_kernels=None):
    """Calculate shortest-path kernels between graphs.

    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    edge_weight : string
        Edge attribute name corresponding to the edge weight.
    node_kernels: dict
        A dictionary of kernel functions for nodes, including 3 items: 'symb' for symbolic node labels, 'nsymb' for non-symbolic node labels, 'mix' for both labels. The first 2 functions take two node labels as parameters, and the 'mix' function takes 4 parameters, a symbolic and a non-symbolic label for each the two nodes. Each label is in form of 2-D dimension array (n_samples, n_features). Each function returns an number as the kernel value. Ignored when nodes are unlabeled.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the sp kernel between 2 praphs.
    """
    # pre-process
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]

    Gn = [nx.to_directed(G) for G in Gn]

    weight = None
    if edge_weight == None:
        print('\n None edge weight specified. Set all weight to 1.\n')
    else:
        try:
            some_weight = list(
                nx.get_edge_attributes(Gn[0], edge_weight).values())[0]
            if isinstance(some_weight, float) or isinstance(some_weight, int):
                weight = edge_weight
            else:
                print(
                    '\n Edge weight with name %s is not float or integer. Set all weight to 1.\n'
                    % edge_weight)
        except:
            print(
                '\n Edge weight with name "%s" is not found in the edge attributes. Set all weight to 1.\n'
                % edge_weight)
    ds_attrs = get_dataset_attributes(
        Gn,
        attr_names=['node_labeled', 'node_attr_dim', 'is_directed'],
        node_label=node_label)

    # remove graphs with no edges, as no sp can be found in their structures, so the kernel between such a graph and itself will be zero.
    len_gn = len(Gn)
    Gn = [(idx, G) for idx, G in enumerate(Gn) if nx.number_of_edges(G) != 0]
    idx = [G[0] for G in Gn]
    Gn = [G[1] for G in Gn]
    if len(Gn) != len_gn:
        print('\n %d graphs are removed as they don\'t contain edges.\n' %
              (len_gn - len(Gn)))

    start_time = time.time()

    # get shortest path graphs of Gn
    Gn = [
        getSPGraph(G, edge_weight=edge_weight)
        for G in tqdm(Gn, desc='getting sp graphs', file=sys.stdout)
    ]

    Kmatrix = np.zeros((len(Gn), len(Gn)))
    pbar = tqdm(
        total=((len(Gn) + 1) * len(Gn) / 2),
        desc='calculating kernels',
        file=sys.stdout)
    if ds_attrs['node_labeled']:
        # node symb and non-synb labeled
        if ds_attrs['node_attr_dim'] > 0:
            if ds_attrs['is_directed']:
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        for e1 in Gn[i].edges(data=True):
                            for e2 in Gn[j].edges(data=True):
                                if e1[2]['cost'] == e2[2]['cost']:
                                    kn = node_kernels['mix']
                                    try:
                                        n11, n12, n21, n22 = Gn[i].nodes[e1[
                                            0]], Gn[i].nodes[e1[1]], Gn[
                                                j].nodes[e2[0]], Gn[j].nodes[
                                                    e2[1]]
                                        kn1 = kn(n11[node_label], n21[
                                            node_label], [n11['attributes']],
                                                 [n21['attributes']]) * kn(
                                                     n12[node_label],
                                                     n22[node_label],
                                                     [n12['attributes']],
                                                     [n22['attributes']])
                                        Kmatrix[i][j] += kn1
                                    except KeyError:  # missing labels or attributes
                                        pass
                        Kmatrix[j][i] = Kmatrix[i][j]
                        pbar.update(1)

            else:
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        for e1 in Gn[i].edges(data=True):
                            for e2 in Gn[j].edges(data=True):
                                if e1[2]['cost'] == e2[2]['cost']:
                                    kn = node_kernels['mix']
                                    try:
                                        # each edge walk is counted twice, starting from both its extreme nodes.
                                        n11, n12, n21, n22 = Gn[i].nodes[e1[
                                            0]], Gn[i].nodes[e1[1]], Gn[
                                                j].nodes[e2[0]], Gn[j].nodes[
                                                    e2[1]]
                                        kn1 = kn(n11[node_label], n21[
                                            node_label], [n11['attributes']],
                                                 [n21['attributes']]) * kn(
                                                     n12[node_label],
                                                     n22[node_label],
                                                     [n12['attributes']],
                                                     [n22['attributes']])
                                        kn2 = kn(n11[node_label], n22[
                                            node_label], [n11['attributes']],
                                                 [n22['attributes']]) * kn(
                                                     n12[node_label],
                                                     n21[node_label],
                                                     [n12['attributes']],
                                                     [n21['attributes']])
                                        Kmatrix[i][j] += kn1 + kn2
                                    except KeyError:  # missing labels or attributes
                                        pass
                        Kmatrix[j][i] = Kmatrix[i][j]
                        pbar.update(1)
        # node symb labeled
        else:
            if ds_attrs['is_directed']:
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        for e1 in Gn[i].edges(data=True):
                            for e2 in Gn[j].edges(data=True):
                                if e1[2]['cost'] == e2[2]['cost']:
                                    kn = node_kernels['symb']
                                    try:
                                        n11, n12, n21, n22 = Gn[i].nodes[e1[
                                            0]], Gn[i].nodes[e1[1]], Gn[
                                                j].nodes[e2[0]], Gn[j].nodes[
                                                    e2[1]]
                                        kn1 = kn(n11[node_label],
                                                 n21[node_label]) * kn(
                                                     n12[node_label],
                                                     n22[node_label])
                                        Kmatrix[i][j] += kn1
                                    except KeyError:  # missing labels
                                        pass
                        Kmatrix[j][i] = Kmatrix[i][j]
                        pbar.update(1)

            else:
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        for e1 in Gn[i].edges(data=True):
                            for e2 in Gn[j].edges(data=True):
                                if e1[2]['cost'] == e2[2]['cost']:
                                    kn = node_kernels['symb']
                                    try:
                                        # each edge walk is counted twice, starting from both its extreme nodes.
                                        n11, n12, n21, n22 = Gn[i].nodes[e1[
                                            0]], Gn[i].nodes[e1[1]], Gn[
                                                j].nodes[e2[0]], Gn[j].nodes[
                                                    e2[1]]
                                        kn1 = kn(n11[node_label],
                                                 n21[node_label]) * kn(
                                                     n12[node_label],
                                                     n22[node_label])
                                        kn2 = kn(n11[node_label],
                                                 n22[node_label]) * kn(
                                                     n12[node_label],
                                                     n21[node_label])
                                        Kmatrix[i][j] += kn1 + kn2
                                    except KeyError:  # missing labels
                                        pass
                        Kmatrix[j][i] = Kmatrix[i][j]
                        pbar.update(1)
    else:
        # node non-synb labeled
        if ds_attrs['node_attr_dim'] > 0:
            if ds_attrs['is_directed']:
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        for e1 in Gn[i].edges(data=True):
                            for e2 in Gn[j].edges(data=True):
                                if e1[2]['cost'] == e2[2]['cost']:
                                    kn = node_kernels['nsymb']
                                    try:
                                        # each edge walk is counted twice, starting from both its extreme nodes.
                                        n11, n12, n21, n22 = Gn[i].nodes[e1[
                                            0]], Gn[i].nodes[e1[1]], Gn[
                                                j].nodes[e2[0]], Gn[j].nodes[
                                                    e2[1]]
                                        kn1 = kn([n11['attributes']],
                                                 [n21['attributes']]) * kn(
                                                     [n12['attributes']],
                                                     [n22['attributes']])
                                        Kmatrix[i][j] += kn1
                                    except KeyError:  # missing attributes
                                        pass
                        Kmatrix[j][i] = Kmatrix[i][j]
                        pbar.update(1)
            else:
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        for e1 in Gn[i].edges(data=True):
                            for e2 in Gn[j].edges(data=True):
                                if e1[2]['cost'] == e2[2]['cost']:
                                    kn = node_kernels['nsymb']
                                    try:
                                        # each edge walk is counted twice, starting from both its extreme nodes.
                                        n11, n12, n21, n22 = Gn[i].nodes[e1[
                                            0]], Gn[i].nodes[e1[1]], Gn[
                                                j].nodes[e2[0]], Gn[j].nodes[
                                                    e2[1]]
                                        kn1 = kn([n11['attributes']],
                                                 [n21['attributes']]) * kn(
                                                     [n12['attributes']],
                                                     [n22['attributes']])
                                        kn2 = kn([n11['attributes']],
                                                 [n22['attributes']]) * kn(
                                                     [n12['attributes']],
                                                     [n21['attributes']])
                                        Kmatrix[i][j] += kn1 + kn2
                                    except KeyError:  # missing attributes
                                        pass
                        Kmatrix[j][i] = Kmatrix[i][j]
                        pbar.update(1)

        # node unlabeled
        else:
            for i in range(0, len(Gn)):
                for j in range(i, len(Gn)):
                    for e1 in Gn[i].edges(data=True):
                        for e2 in Gn[j].edges(data=True):
                            if e1[2]['cost'] == e2[2]['cost']:
                                Kmatrix[i][j] += 1
                    Kmatrix[j][i] = Kmatrix[i][j]
                    pbar.update(1)

    run_time = time.time() - start_time
    print(
        "\n --- shortest path kernel matrix of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time, idx
