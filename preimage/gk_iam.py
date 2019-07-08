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
from pygraph.kernels.spKernel import spkernel
import functools
from pygraph.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from pygraph.kernels.structuralspKernel import structuralspkernel
from median import draw_Letter_graph


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
        print(gi.nodes(data=True))
        print(gi.edges(data=True))
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
        print(g_tmp.nodes(data=True))
        print(g_tmp.edges(data=True))
        
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


#def gk_iam_nearest_multi(Gn, alpha, idx_gi, Kmatrix, k, r_max):
#    """This function constructs graph pre-image by the iterative pre-image 
#    framework in reference [1], algorithm 1, where the step of generating new 
#    graphs randomly is replaced by the IAM algorithm in reference [2].
#    
#    notes
#    -----
#    Every time a set of n better graphs is acquired, their distances in kernel space are
#    compared with the k nearest ones, and the k nearest distances from the k+n
#    distances will be used as the new ones.
#    """
#    Gn_median = [Gn[idx].copy() for idx in idx_gi]
#    # compute k nearest neighbors of phi in DN.
#    dis_list = [] # distance between g_star and each graph.
#    for ig, g in tqdm(enumerate(Gn), desc='computing distances', file=sys.stdout):
#        dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix)
##        dtemp = k_list[ig] - 2 * (alpha * k_g1_list[ig] + (1 - alpha) * 
##                      k_g2_list[ig]) + (alpha * alpha * k_list[0] + alpha * 
##                      (1 - alpha) * k_g2_list[0] + (1 - alpha) * alpha * 
##                      k_g1_list[6] + (1 - alpha) * (1 - alpha) * k_list[6])
#        dis_list.append(dtemp)
#        
#    # sort
#    sort_idx = np.argsort(dis_list)
#    dis_gs = [dis_list[idis] for idis in sort_idx[0:k]] # the k shortest distances
#    nb_best = len(np.argwhere(dis_gs == dis_gs[0]).flatten().tolist())
#    g0hat_list = [Gn[idx] for idx in sort_idx[0:nb_best]] # the nearest neighbors of phi in DN
#    if dis_gs[0] == 0: # the exact pre-image.
#        print('The exact pre-image is found from the input dataset.')
#        return 0, g0hat_list
#    dhat = dis_gs[0] # the nearest distance
#    ghat_list = [g.copy() for g in g0hat_list]
#    for g in ghat_list:
#        nx.draw_networkx(g)
#        plt.show()
#        print(g.nodes(data=True))
#        print(g.edges(data=True))
#    Gk = [Gn[ig].copy() for ig in sort_idx[0:k]] # the k nearest neighbors
#    for gi in Gk:
#        nx.draw_networkx(gi)
#        plt.show()
#        print(gi.nodes(data=True))
#        print(gi.edges(data=True))
#    Gs_nearest = Gk.copy()
##    gihat_list = []
#    
##    i = 1
#    r = 1
#    while r < r_max:
#        print('r =', r)
##        found = False
##        Gs_nearest = Gk + gihat_list
##        g_tmp = iam(Gs_nearest)
#        g_tmp_list = test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations(
#                Gn_median, Gs_nearest, c_ei=1, c_er=1, c_es=1)
#        for g in g_tmp_list:
#            nx.draw_networkx(g)
#            plt.show()
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
#        
#        # compute distance between phi and the new generated graphs.
#        gi_list = [Gn[i] for i in idx_gi]
#        knew = compute_kernel(g_tmp_list + gi_list, 'marginalizedkernel', False)
#        dnew_list = []
#        for idx, g_tmp in enumerate(g_tmp_list):
#            dnew_list.append(dis_gstar(idx, range(len(g_tmp_list), 
#                            len(g_tmp_list) + len(gi_list) + 1), alpha, knew))
#        
##        dnew = knew[0, 0] - 2 * (alpha[0] * knew[0, 1] + alpha[1] * 
##              knew[0, 2]) + (alpha[0] * alpha[0] * k_list[0] + alpha[0] * 
##              alpha[1] * k_g2_list[0] + alpha[1] * alpha[0] * 
##              k_g1_list[1] + alpha[1] * alpha[1] * k_list[1])
#            
#        # find the new k nearest graphs.
#        dis_gs = dnew_list + dis_gs # add the new nearest distances.
#        Gs_nearest = [g.copy() for g in g_tmp_list] + Gs_nearest # add the corresponding graphs.
#        sort_idx = np.argsort(dis_gs)
#        if len([i for i in sort_idx[0:k] if i < len(dnew_list)]) > 0:
#            print('We got better k nearest neighbors! Hurray!')
#            dis_gs = [dis_gs[idx] for idx in sort_idx[0:k]] # the new k nearest distances.
#            print(dis_gs[-1])
#            Gs_nearest = [Gs_nearest[idx] for idx in sort_idx[0:k]]
#            nb_best = len(np.argwhere(dis_gs == dis_gs[0]).flatten().tolist())
#            if len([i for i in sort_idx[0:nb_best] if i < len(dnew_list)]) > 0:
#                print('I have smaller or equal distance!')
#                dhat = dis_gs[0]
#                print(str(dhat) + '->' + str(dhat))
#                idx_best_list = np.argwhere(dnew_list == dhat).flatten().tolist()
#                ghat_list = [g_tmp_list[idx].copy() for idx in idx_best_list]
#                for g in ghat_list:
#                    nx.draw_networkx(g)
#                    plt.show()
#                    print(g.nodes(data=True))
#                    print(g.edges(data=True))
#            r = 0
#        else:
#            r += 1
#    
#    return dhat, ghat_list


def gk_iam_nearest_multi(Gn_init, Gn_median, alpha, idx_gi, Kmatrix, k, r_max, gkernel):
    """This function constructs graph pre-image by the iterative pre-image 
    framework in reference [1], algorithm 1, where the step of generating new 
    graphs randomly is replaced by the IAM algorithm in reference [2].
    
    notes
    -----
    Every time a set of n better graphs is acquired, their distances in kernel space are
    compared with the k nearest ones, and the k nearest distances from the k+n
    distances will be used as the new ones.
    """
    # compute k nearest neighbors of phi in DN.
    dis_list = [] # distance between g_star and each graph.
    term3 = 0
    for i1, a1 in enumerate(alpha):
        for i2, a2 in enumerate(alpha):
            term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    for ig, g in tqdm(enumerate(Gn_init), desc='computing distances', file=sys.stdout):
        dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix, term3=term3)
#        dtemp = k_list[ig] - 2 * (alpha * k_g1_list[ig] + (1 - alpha) * 
#                      k_g2_list[ig]) + (alpha * alpha * k_list[0] + alpha * 
#                      (1 - alpha) * k_g2_list[0] + (1 - alpha) * alpha * 
#                      k_g1_list[6] + (1 - alpha) * (1 - alpha) * k_list[6])
        dis_list.append(dtemp)
        
    # sort
    sort_idx = np.argsort(dis_list)
    dis_gs = [dis_list[idis] for idis in sort_idx[0:k]] # the k shortest distances
    nb_best = len(np.argwhere(dis_gs == dis_gs[0]).flatten().tolist())
    g0hat_list = [Gn_init[idx] for idx in sort_idx[0:nb_best]] # the nearest neighbors of phi in DN
    if dis_gs[0] == 0: # the exact pre-image.
        print('The exact pre-image is found from the input dataset.')
        return 0, g0hat_list
    dhat = dis_gs[0] # the nearest distance
    ghat_list = [g.copy() for g in g0hat_list]
    for g in ghat_list:
        draw_Letter_graph(g)
#        nx.draw_networkx(g)
#        plt.show()
        print(g.nodes(data=True))
        print(g.edges(data=True))
    Gk = [Gn_init[ig].copy() for ig in sort_idx[0:k]] # the k nearest neighbors
    for gi in Gk:
#        nx.draw_networkx(gi)
#        plt.show()
        draw_Letter_graph(g)
        print(gi.nodes(data=True))
        print(gi.edges(data=True))
    Gs_nearest = Gk.copy()
#    gihat_list = []
    
#    i = 1
    r = 1
    while r < r_max:
        print('r =', r)
#        found = False
#        Gs_nearest = Gk + gihat_list
#        g_tmp = iam(Gs_nearest)
        g_tmp_list = test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations(
                Gn_median, Gs_nearest, c_ei=1, c_er=1, c_es=1)
        for g in g_tmp_list:
#            nx.draw_networkx(g)
#            plt.show()
            draw_Letter_graph(g)
            print(g.nodes(data=True))
            print(g.edges(data=True))
        
        # compute distance between phi and the new generated graphs.
        knew = compute_kernel(g_tmp_list + Gn_median, gkernel, False)
        dnew_list = []
        for idx, g_tmp in enumerate(g_tmp_list):
            dnew_list.append(dis_gstar(idx, range(len(g_tmp_list), 
                            len(g_tmp_list) + len(Gn_median) + 1), alpha, knew,
                            withterm3=False))
        
#        dnew = knew[0, 0] - 2 * (alpha[0] * knew[0, 1] + alpha[1] * 
#              knew[0, 2]) + (alpha[0] * alpha[0] * k_list[0] + alpha[0] * 
#              alpha[1] * k_g2_list[0] + alpha[1] * alpha[0] * 
#              k_g1_list[1] + alpha[1] * alpha[1] * k_list[1])
            
        # find the new k nearest graphs.
        dis_gs = dnew_list + dis_gs # add the new nearest distances.
        Gs_nearest = [g.copy() for g in g_tmp_list] + Gs_nearest # add the corresponding graphs.
        sort_idx = np.argsort(dis_gs)
        if len([i for i in sort_idx[0:k] if i < len(dnew_list)]) > 0:
            print('We got better k nearest neighbors! Hurray!')
            dis_gs = [dis_gs[idx] for idx in sort_idx[0:k]] # the new k nearest distances.
            print(dis_gs[-1])
            Gs_nearest = [Gs_nearest[idx] for idx in sort_idx[0:k]]
            nb_best = len(np.argwhere(dis_gs == dis_gs[0]).flatten().tolist())
            if len([i for i in sort_idx[0:nb_best] if i < len(dnew_list)]) > 0:
                print('I have smaller or equal distance!')
                print(str(dhat) + '->' + str(dis_gs[0]))
                dhat = dis_gs[0]
                idx_best_list = np.argwhere(dnew_list == dhat).flatten().tolist()
                ghat_list = [g_tmp_list[idx].copy() for idx in idx_best_list]
                for g in ghat_list:
#                    nx.draw_networkx(g)
#                    plt.show()
                    draw_Letter_graph(g)
                    print(g.nodes(data=True))
                    print(g.edges(data=True))
            r = 0
        else:
            r += 1
    
    return dhat, ghat_list


def dis_gstar(idx_g, idx_gi, alpha, Kmatrix, term3=0, withterm3=True):
    term1 = Kmatrix[idx_g, idx_g]
    term2 = 0
    for i, a in enumerate(alpha):
        term2 += a * Kmatrix[idx_g, idx_gi[i]]
    term2 *= 2
    if withterm3 == False:
        for i1, a1 in enumerate(alpha):
            for i2, a2 in enumerate(alpha):
                term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    return np.sqrt(term1 - term2 + term3)


def compute_kernel(Gn, graph_kernel, verbose):
    if graph_kernel == 'marginalizedkernel':
        Kmatrix, _ = marginalizedkernel(Gn, node_label='atom', edge_label=None,
                                  p_quit=0.03, n_iteration=20, remove_totters=False,
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'untilhpathkernel':
        Kmatrix, _ = untilhpathkernel(Gn, node_label='atom', edge_label='bond_type',
                                  depth=10, k_func='MinMax', compute_method='trie',
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'spkernel':
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        Kmatrix, _, _ = spkernel(Gn, node_label='atom', node_kernels=
                              {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel},
                              n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'structuralspkernel':
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        Kmatrix, _ = structuralspkernel(Gn, node_label='atom', node_kernels=
                              {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel},
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