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
import numpy as np
import multiprocessing
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from iam import iam


def gk_iam(Gn, alpha):
    """This function constructs graph pre-image by the iterative pre-image 
    framework in reference [1], algorithm 1, where the step of generating new 
    graphs randomly is replaced by the IAM algorithm in reference [2].
    
    notes
    -----
    Every time a better graph is acquired, the older one is replaced by it.
    """
    # compute k nearest neighbors of phi in DN.
    dis_list = [] # distance between g_star and each graph.
    for ig, g in tqdm(enumerate(Gn), desc='computing distances', file=sys.stdout):
        dtemp = k_list[ig] - 2 * (alpha * k_g1_list[ig] + (1 - alpha) * 
                      k_g2_list[ig]) + (alpha * alpha * k_list[idx1] + alpha * 
                      (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
                      k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2])
        dis_list.append(dtemp)
        
    # sort
    sort_idx = np.argsort(dis_list)
    dis_gs = [dis_list[idis] for idis in sort_idx[0:k]]
    g0hat = Gn[sort_idx[0]] # the nearest neighbor of phi in DN
    if dis_gs[0] == 0: # the exact pre-image.
        print('The exact pre-image is found from the input dataset.')
        return 0, g0hat
    dhat = dis_gs[0] # the nearest distance
    Gk = [Gn[ig] for ig in sort_idx[0:k]] # the k nearest neighbors
    gihat_list = []
    
#    i = 1
    r = 1
    while r < r_max:
        print('r =', r)
#        found = False
        Gs_nearest = Gk + gihat_list
        g_tmp = iam(Gs_nearest)
        
        # compute distance between phi and the new generated graph.
        knew = marginalizedkernel([g_tmp, g1, g2], node_label='atom', edge_label=None,
                       p_quit=lmbda, n_iteration=20, remove_totters=False,
                       n_jobs=multiprocessing.cpu_count(), verbose=False)
        dnew = knew[0][0, 0] - 2 * (alpha * knew[0][0, 1] + (1 - alpha) * 
              knew[0][0, 2]) + (alpha * alpha * k_list[idx1] + alpha * 
              (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
              k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2])
        if dnew <= dhat: # the new distance is smaller
            print('I am smaller!')
            dhat = dnew
            g_new = g_tmp.copy() # found better graph.
            gihat_list = [g_new]
            dis_gs.append(dhat)
            r = 0
        else:
            r += 1
            
    ghat = ([g0hat] if len(gihat_list) == 0 else gihat_list)
    
    return dhat, ghat


def gk_iam_nearest(Gn, alpha):
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
        dtemp = k_list[ig] - 2 * (alpha * k_g1_list[ig] + (1 - alpha) * 
                      k_g2_list[ig]) + (alpha * alpha * k_list[idx1] + alpha * 
                      (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
                      k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2])
        dis_list.append(dtemp)
        
    # sort
    sort_idx = np.argsort(dis_list)
    dis_gs = [dis_list[idis] for idis in sort_idx[0:k]] # the k shortest distances
    g0hat = Gn[sort_idx[0]] # the nearest neighbor of phi in DN
    if dis_gs[0] == 0: # the exact pre-image.
        print('The exact pre-image is found from the input dataset.')
        return 0, g0hat
    dhat = dis_gs[0] # the nearest distance
    ghat = g0hat
    Gk = [Gn[ig] for ig in sort_idx[0:k]] # the k nearest neighbors
    Gs_nearest = Gk
#    gihat_list = []
    
#    i = 1
    r = 1
    while r < r_max:
        print('r =', r)
#        found = False
#        Gs_nearest = Gk + gihat_list
        g_tmp = iam(Gs_nearest)
        
        # compute distance between phi and the new generated graph.
        knew = marginalizedkernel([g_tmp, g1, g2], node_label='atom', edge_label=None,
                       p_quit=lmbda, n_iteration=20, remove_totters=False,
                       n_jobs=multiprocessing.cpu_count(), verbose=False)
        dnew = knew[0][0, 0] - 2 * (alpha * knew[0][0, 1] + (1 - alpha) * 
              knew[0][0, 2]) + (alpha * alpha * k_list[idx1] + alpha * 
              (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
              k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2])
        if dnew <= dhat: # the new distance is smaller
            print('I am smaller!')
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
            

if __name__ == '__main__':
    import sys
    sys.path.insert(0, "../")
    from pygraph.kernels.marginalizedKernel import marginalizedkernel
    from pygraph.utils.graphfiles import loadDataset
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG.mat',
          'extra_params': {'am_sp_al_nl_el': [0, 0, 3, 1, 2]}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:10]
    
    lmbda = 0.03 # termination probalility
    r_max = 10 # recursions
    l = 500
    alpha_range = np.linspace(0.1, 0.9, 9)
    k = 5 # k nearest neighbors
    
    # randomly select two molecules
    np.random.seed(1)
    idx1, idx2 = np.random.randint(0, len(Gn), 2)
    g1 = Gn[idx1]
    g2 = Gn[idx2]
    
    # compute 
    k_list = [] # kernel between each graph and itself.
    k_g1_list = [] # kernel between each graph and g1
    k_g2_list = [] # kernel between each graph and g2
    for ig, g in tqdm(enumerate(Gn), desc='computing self kernels', file=sys.stdout): 
        ktemp = marginalizedkernel([g, g1, g2], node_label='atom', edge_label=None,
                                   p_quit=lmbda, n_iteration=20, remove_totters=False,
                                   n_jobs=multiprocessing.cpu_count(), verbose=False)
        k_list.append(ktemp[0][0, 0])
        k_g1_list.append(ktemp[0][0, 1])
        k_g2_list.append(ktemp[0][0, 2])

    g_best = []
    dis_best = []
    # for each alpha
    for alpha in alpha_range:
        print('alpha =', alpha)
        dhat, ghat = gk_iam_nearest(Gn, alpha)
        dis_best.append(dhat)
        g_best.append(ghat)
        
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', dis_best[idx])
        print('the corresponding pre-image is')
        nx.draw_networkx(g_best[idx])
        plt.show()