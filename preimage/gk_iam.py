#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:07:43 2019

A graph pre-image method combining iterative pre-image method in reference [1] 
and the iterative alternate minimizations (IAM) in reference [2].
@author: ljia
@references:
    [1] Gökhan H Bakir, Alexander Zien, and Koji Tsuda. Learning to and graph 
    pre-images. In Joint Pattern Re ognition Symposium , pages 253-261. Springer, 2004.
    [2] Generalized median graph via iterative alternate minimization.
"""
import sys
import numpy as np
import multiprocessing
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from iam import iam, test_iam_with_more_graphs_as_init, test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations
sys.path.insert(0, "../")
from pygraph.kernels.marginalizedKernel import marginalizedkernel
from pygraph.kernels.untilHPathKernel import untilhpathkernel


def gk_iam(Gn, alpha):
    """This function constructs graph pre-image by the iterative pre-image 
    framework in reference [1], algorithm 1, where the step of generating new 
    graphs randomly is replaced by the IAM algorithm in reference [2].
    
    notes
    -----
    Every time a better graph is acquired, the older one is replaced by it.
    """
    pass
#    # compute k nearest neighbors of phi in DN.
#    dis_list = [] # distance between g_star and each graph.
#    for ig, g in tqdm(enumerate(Gn), desc='computing distances', file=sys.stdout):
#        dtemp = k_list[ig] - 2 * (alpha * k_g1_list[ig] + (1 - alpha) * 
#                      k_g2_list[ig]) + (alpha * alpha * k_list[idx1] + alpha * 
#                      (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
#                      k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2])
#        dis_list.append(dtemp)
#        
#    # sort
#    sort_idx = np.argsort(dis_list)
#    dis_gs = [dis_list[idis] for idis in sort_idx[0:k]]
#    g0hat = Gn[sort_idx[0]] # the nearest neighbor of phi in DN
#    if dis_gs[0] == 0: # the exact pre-image.
#        print('The exact pre-image is found from the input dataset.')
#        return 0, g0hat
#    dhat = dis_gs[0] # the nearest distance
#    Gk = [Gn[ig] for ig in sort_idx[0:k]] # the k nearest neighbors
#    gihat_list = []
#    
##    i = 1
#    r = 1
#    while r < r_max:
#        print('r =', r)
##        found = False
#        Gs_nearest = Gk + gihat_list
#        g_tmp = iam(Gs_nearest)
#        
#        # compute distance between phi and the new generated graph.
#        knew = marginalizedkernel([g_tmp, g1, g2], node_label='atom', edge_label=None,
#                       p_quit=lmbda, n_iteration=20, remove_totters=False,
#                       n_jobs=multiprocessing.cpu_count(), verbose=False)
#        dnew = knew[0][0, 0] - 2 * (alpha * knew[0][0, 1] + (1 - alpha) * 
#              knew[0][0, 2]) + (alpha * alpha * k_list[idx1] + alpha * 
#              (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
#              k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2])
#        if dnew <= dhat: # the new distance is smaller
#            print('I am smaller!')
#            dhat = dnew
#            g_new = g_tmp.copy() # found better graph.
#            gihat_list = [g_new]
#            dis_gs.append(dhat)
#            r = 0
#        else:
#            r += 1
#            
#    ghat = ([g0hat] if len(gihat_list) == 0 else gihat_list)
#    
#    return dhat, ghat


def gk_iam_nearest(Gn, alpha, idx_gi, Kmatrix, k, r_max):
    """This function constructs graph pre-image by the iterative pre-image 
    framework in reference [1], algorithm 1, where the step of generating new 
    graphs randomly is replaced by the IAM algorithm in reference [2].
    
    notes
    -----
    Every time a better graph is acquired, its distance in kernel space is
    compared with the k nearest ones, and the k nearest distances from the k+1
    distances will be used as the new ones.
    """
    # compute k nearest neighbors of phi in DN.
    dis_list = [] # distance between g_star and each graph.
    for ig, g in tqdm(enumerate(Gn), desc='computing distances', file=sys.stdout):
        dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix)
#        dtemp = k_list[ig] - 2 * (alpha * k_g1_list[ig] + (1 - alpha) * 
#                      k_g2_list[ig]) + (alpha * alpha * k_list[0] + alpha * 
#                      (1 - alpha) * k_g2_list[0] + (1 - alpha) * alpha * 
#                      k_g1_list[6] + (1 - alpha) * (1 - alpha) * k_list[6])
        dis_list.append(dtemp)
        
    # sort
    sort_idx = np.argsort(dis_list)
    dis_gs = [dis_list[idis] for idis in sort_idx[0:k]] # the k shortest distances
    g0hat = Gn[sort_idx[0]] # the nearest neighbor of phi in DN
    if dis_gs[0] == 0: # the exact pre-image.
        print('The exact pre-image is found from the input dataset.')
        return 0, g0hat
    dhat = dis_gs[0] # the nearest distance
    ghat = g0hat.copy()
    Gk = [Gn[ig].copy() for ig in sort_idx[0:k]] # the k nearest neighbors
    for gi in Gk:
        nx.draw_networkx(gi)
        plt.show()
    Gs_nearest = Gk.copy()
#    gihat_list = []
    
#    i = 1
    r = 1
    while r < r_max:
        print('r =', r)
#        found = False
#        Gs_nearest = Gk + gihat_list
#        g_tmp = iam(Gs_nearest)
        g_tmp = test_iam_with_more_graphs_as_init(Gs_nearest, Gs_nearest, c_ei=1, c_er=1, c_es=1)
        nx.draw_networkx(g_tmp)
        plt.show()
        
        # compute distance between phi and the new generated graph.
        gi_list = [Gn[i] for i in idx_gi]
        knew = compute_kernel([g_tmp] + gi_list, 'untilhpathkernel', False)
        dnew = dis_gstar(0, range(1, len(gi_list) + 1), alpha, knew)
        
#        dnew = knew[0, 0] - 2 * (alpha[0] * knew[0, 1] + alpha[1] * 
#              knew[0, 2]) + (alpha[0] * alpha[0] * k_list[0] + alpha[0] * 
#              alpha[1] * k_g2_list[0] + alpha[1] * alpha[0] * 
#              k_g1_list[1] + alpha[1] * alpha[1] * k_list[1])
        if dnew <= dhat and g_tmp != ghat: # the new distance is smaller
            print('I am smaller!')
            print(str(dhat) + '->' + str(dnew))
#            nx.draw_networkx(ghat)
#            plt.show()
#            print('->')
#            nx.draw_networkx(g_tmp)
#            plt.show()
            
            dhat = dnew
            g_new = g_tmp.copy() # found better graph.
            ghat = g_tmp.copy()
            dis_gs.append(dhat) # add the new nearest distance.
            Gs_nearest.append(g_new) # add the corresponding graph.
            sort_idx = np.argsort(dis_gs)
            dis_gs = [dis_gs[idx] for idx in sort_idx[0:k]] # the new k nearest distances.
            Gs_nearest = [Gs_nearest[idx] for idx in sort_idx[0:k]]
            r = 0
        else:
            r += 1
    
    return dhat, ghat


def dis_gstar(idx_g, idx_gi, alpha, Kmatrix):
    term1 = Kmatrix[idx_g, idx_g]
    term2 = 0
    for i, a in enumerate(alpha):
        term2 += a * Kmatrix[idx_g, idx_gi[i]]
    term2 *= 2
    term3 = 0
    for i1, a1 in enumerate(alpha):
        for i2, a2 in enumerate(alpha):
            term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    return np.sqrt(term1 - term2 + term3)


def compute_kernel(Gn, graph_kernel, verbose):
    if graph_kernel == 'marginalizedkernel':
        Kmatrix, _ = marginalizedkernel(Gn, node_label='atom', edge_label=None,
                                  p_quit=0.3, n_iteration=19, remove_totters=False,
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'untilhpathkernel':
        Kmatrix, _ = untilhpathkernel(Gn, node_label='atom', edge_label='bond_type',
                                  depth=2, k_func='MinMax', compute_method='trie',
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
        
    # normalization
    Kmatrix_diag = Kmatrix.diagonal().copy()
    for i in range(len(Kmatrix)):
        for j in range(i, len(Kmatrix)):
            Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
            Kmatrix[j][i] = Kmatrix[i][j]
    return Kmatrix
            

def gram2distances(Kmatrix):
    dmatrix = np.zeros((len(Kmatrix), len(Kmatrix)))
    for i1 in range(len(Kmatrix)):
        for i2 in range(len(Kmatrix)):
            dmatrix[i1, i2] = Kmatrix[i1, i1] + Kmatrix[i2, i2] - 2 * Kmatrix[i1, i2]
    dmatrix = np.sqrt(dmatrix)
    return dmatrix

# --------------------------- These are tests --------------------------------#
    
def test_who_is_the_closest_in_kernel_space(Gn):
    idx_gi = [0, 6]
    g1 = Gn[idx_gi[0]]
    g2 = Gn[idx_gi[1]]
    # create the "median" graph.
    gnew = g2.copy()
    gnew.remove_node(0)
    nx.draw_networkx(gnew)
    plt.show()
    print(gnew.nodes(data=True))
    Gn = [gnew] + Gn
    
    # compute gram matrix
    Kmatrix = compute_kernel(Gn, 'untilhpathkernel', True)
    # the distance matrix
    dmatrix = gram2distances(Kmatrix)
    print(np.sort(dmatrix[idx_gi[0] + 1]))
    print(np.argsort(dmatrix[idx_gi[0] + 1]))
    print(np.sort(dmatrix[idx_gi[1] + 1]))
    print(np.argsort(dmatrix[idx_gi[1] + 1]))
    # for all g in Gn, compute (d(g1, g) + d(g2, g)) / 2
    dis_median = [(dmatrix[i, idx_gi[0] + 1] + dmatrix[i, idx_gi[1] + 1]) / 2 for i in range(len(Gn))]
    print(np.sort(dis_median))
    print(np.argsort(dis_median))
    return


def test_who_is_the_closest_in_GED_space(Gn):
    from iam import GED
    idx_gi = [0, 6]
    g1 = Gn[idx_gi[0]]
    g2 = Gn[idx_gi[1]]
    # create the "median" graph.
    gnew = g2.copy()
    gnew.remove_node(0)
    nx.draw_networkx(gnew)
    plt.show()
    print(gnew.nodes(data=True))
    Gn = [gnew] + Gn
    
    # compute GEDs
    ged_matrix = np.zeros((len(Gn), len(Gn)))
    for i1 in tqdm(range(len(Gn)), desc='computing GEDs', file=sys.stdout):
        for i2 in range(len(Gn)):
            dis, _, _ = GED(Gn[i1], Gn[i2], lib='gedlib')
            ged_matrix[i1, i2] = dis
    print(np.sort(ged_matrix[idx_gi[0] + 1]))
    print(np.argsort(ged_matrix[idx_gi[0] + 1]))
    print(np.sort(ged_matrix[idx_gi[1] + 1]))
    print(np.argsort(ged_matrix[idx_gi[1] + 1]))
    # for all g in Gn, compute (GED(g1, g) + GED(g2, g)) / 2
    dis_median = [(ged_matrix[i, idx_gi[0] + 1] + ged_matrix[i, idx_gi[1] + 1]) / 2 for i in range(len(Gn))]
    print(np.sort(dis_median))
    print(np.argsort(dis_median))
    return


def test_will_IAM_give_the_median_graph_we_wanted(Gn):
    idx_gi = [0, 6]
    g1 = Gn[idx_gi[0]].copy()
    g2 = Gn[idx_gi[1]].copy()
#    del Gn[idx_gi[0]]
#    del Gn[idx_gi[1] - 1]
    g_median = test_iam_with_more_graphs_as_init([g1, g2], [g1, g2], c_ei=1, c_er=1, c_es=1)
#    g_median = test_iam_with_more_graphs_as_init(Gn, Gn, c_ei=1, c_er=1, c_es=1)
    nx.draw_networkx(g_median)
    plt.show()
    print(g_median.nodes(data=True))
    print(g_median.edges(data=True))
    
    
def test_new_IAM_allGraph_deleteNodes(Gn):
    idx_gi = [0, 6]
#    g1 = Gn[idx_gi[0]].copy()
#    g2 = Gn[idx_gi[1]].copy()

    g1 = nx.Graph(name='haha')
    g1.add_nodes_from([(2, {'atom': 'C'}), (3, {'atom': 'O'}), (4, {'atom': 'C'})])
    g1.add_edges_from([(2, 3, {'bond_type': '1'}), (3, 4, {'bond_type': '1'})])
    g2 = nx.Graph(name='hahaha')
    g2.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'O'}), (2, {'atom': 'C'}),
                       (3, {'atom': 'O'}), (4, {'atom': 'C'})])
    g2.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'}),
                       (2, 3, {'bond_type': '1'}), (3, 4, {'bond_type': '1'})])
#    g2 = g1.copy()
#    g2.add_nodes_from([(3, {'atom': 'O'})])
#    g2.add_nodes_from([(4, {'atom': 'C'})])
#    g2.add_edges_from([(1, 3, {'bond_type': '1'})])
#    g2.add_edges_from([(3, 4, {'bond_type': '1'})])

#    del Gn[idx_gi[0]]
#    del Gn[idx_gi[1] - 1]
    g_median = test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations([g1, g2], [g1, g2], c_ei=1, c_er=1, c_es=1)
#    g_median = test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations(Gn, Gn, c_ei=1, c_er=1, c_es=1)
    nx.draw_networkx(g_median)
    plt.show()
    print(g_median.nodes(data=True))
    print(g_median.edges(data=True))


if __name__ == '__main__':
    from pygraph.utils.graphfiles import loadDataset
#    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG.mat',
#          'extra_params': {'am_sp_al_nl_el': [0, 0, 3, 1, 2]}}  # node/edge symb
#    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
#          'extra_params': {}} # node nsymb
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/monoterpenoides/trainset_9.ds',
#          'extra_params': {}}
    ds = {'name': 'Acyclic', 'dataset': '../datasets/acyclic/dataset_bps.ds',
        'extra_params': {}} # node symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:20]
    
    test_new_IAM_allGraph_deleteNodes(Gn)
    test_will_IAM_give_the_median_graph_we_wanted(Gn)
    test_who_is_the_closest_in_GED_space(Gn)
    test_who_is_the_closest_in_kernel_space(Gn)
    
    lmbda = 0.03 # termination probalility
    r_max = 10 # recursions
    l = 500
    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 20 # k nearest neighbors
    
    # randomly select two molecules
    np.random.seed(1)
    idx_gi = [0, 6] # np.random.randint(0, len(Gn), 2)
    g1 = Gn[idx_gi[0]]
    g2 = Gn[idx_gi[1]]
    
#    g_tmp = iam([g1, g2])
#    nx.draw_networkx(g_tmp)
#    plt.show()
    
    # compute 
#    k_list = [] # kernel between each graph and itself.
#    k_g1_list = [] # kernel between each graph and g1
#    k_g2_list = [] # kernel between each graph and g2
#    for ig, g in tqdm(enumerate(Gn), desc='computing self kernels', file=sys.stdout): 
#        ktemp = compute_kernel([g, g1, g2], 'marginalizedkernel', False)
#        k_list.append(ktemp[0][0, 0])
#        k_g1_list.append(ktemp[0][0, 1])
#        k_g2_list.append(ktemp[0][0, 2])
        
    km = compute_kernel(Gn, 'untilhpathkernel', True)
#    k_list = np.diag(km) # kernel between each graph and itself.
#    k_g1_list = km[idx_gi[0]] # kernel between each graph and g1
#    k_g2_list = km[idx_gi[1]] # kernel between each graph and g2    

    g_best = []
    dis_best = []
    # for each alpha
    for alpha in alpha_range:
        print('alpha =', alpha)
        dhat, ghat = gk_iam_nearest(Gn, [alpha, 1 - alpha], idx_gi, km, k, r_max)
        dis_best.append(dhat)
        g_best.append(ghat)
        
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', dis_best[idx])
        print('the corresponding pre-image is')
        nx.draw_networkx(g_best[idx])
        plt.show()