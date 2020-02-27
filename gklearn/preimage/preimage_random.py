#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:03:11 2019

pre-image
@author: ljia
"""

import sys
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


sys.path.insert(0, "../")

from utils import compute_kernel, dis_gstar


def preimage_random(Gn_init, Gn_median, alpha, idx_gi, Kmatrix, k, r_max, l, gkernel):
    Gn_init = [nx.convert_node_labels_to_integers(g) for g in Gn_init]
    
    # compute k nearest neighbors of phi in DN.
    dis_list = [] # distance between g_star and each graph.
    term3 = 0
    for i1, a1 in enumerate(alpha):
        for i2, a2 in enumerate(alpha):
            term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    for ig, g in tqdm(enumerate(Gn_init), desc='computing distances', file=sys.stdout):
        dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix, term3=term3)
        dis_list.append(dtemp)
#    print(np.max(dis_list))
#    print(np.min(dis_list))
#    print(np.min([item for item in dis_list if item != 0]))
#    print(np.mean(dis_list))
        
    # sort
    sort_idx = np.argsort(dis_list)
    dis_gs = [dis_list[idis] for idis in sort_idx[0:k]] # the k shortest distances
    nb_best = len(np.argwhere(dis_gs == dis_gs[0]).flatten().tolist())
    g0hat_list = [Gn_init[idx] for idx in sort_idx[0:nb_best]] # the nearest neighbors of phi in DN
    if dis_gs[0] == 0: # the exact pre-image.
        print('The exact pre-image is found from the input dataset.')
        return 0, g0hat_list[0], 0
    dhat = dis_gs[0] # the nearest distance
#    ghat_list = [g.copy() for g in g0hat_list]
#    for g in ghat_list:
#        draw_Letter_graph(g)
#        nx.draw_networkx(g)
#        plt.show()
#        print(g.nodes(data=True))
#        print(g.edges(data=True))
    Gk = [Gn_init[ig].copy() for ig in sort_idx[0:k]] # the k nearest neighbors
#    for gi in Gk:
##        nx.draw_networkx(gi)
##        plt.show()
#        draw_Letter_graph(g)
#        print(gi.nodes(data=True))
#        print(gi.edges(data=True))
    Gs_nearest = [g.copy() for g in Gk]
    gihat_list = []
    dihat_list = []
    
#    i = 1
    r = 0
#    sod_list = [dhat]
#    found = False
    dis_of_each_itr = [dhat]
    nb_updated = 0
    g_best = []
    while r < r_max:
        print('\nr =', r)
        print('itr for gk =', nb_updated, '\n')
        found = False
        dis_bests = dis_gs + dihat_list
        # @todo what if the log is negetive? how to choose alpha (scalar)?
        fdgs_list = np.array(dis_bests)
        if np.min(fdgs_list) < 1:
            fdgs_list /= np.min(dis_bests)
        fdgs_list = [int(item) for item in np.ceil(np.log(fdgs_list))]
        if np.min(fdgs_list) < 1:
            fdgs_list = np.array(fdgs_list) + 1
            
        for ig, gs in enumerate(Gs_nearest + gihat_list):
#            nx.draw_networkx(gs)
#            plt.show()
            for trail in range(0, l):
#            for trail in tqdm(range(0, l), desc='l loops', file=sys.stdout):
                # add and delete edges.
                gtemp = gs.copy()
                np.random.seed()
                # which edges to change.
                # @todo: should we use just half of the adjacency matrix for undirected graphs?
                nb_vpairs = nx.number_of_nodes(gs) * (nx.number_of_nodes(gs) - 1)
                # @todo: what if fdgs is bigger than nb_vpairs?
                idx_change = random.sample(range(nb_vpairs), fdgs_list[ig] if 
                                           fdgs_list[ig] < nb_vpairs else nb_vpairs)
#                idx_change = np.random.randint(0, nx.number_of_nodes(gs) * 
#                                               (nx.number_of_nodes(gs) - 1), fdgs)
                for item in idx_change:
                    node1 = int(item / (nx.number_of_nodes(gs) - 1))
                    node2 = (item - node1 * (nx.number_of_nodes(gs) - 1))
                    if node2 >= node1: # skip the self pair.
                        node2 += 1
                    # @todo: is the randomness correct?
                    if not gtemp.has_edge(node1, node2):
                        gtemp.add_edge(node1, node2)
#                        nx.draw_networkx(gs)
#                        plt.show()
#                        nx.draw_networkx(gtemp)
#                        plt.show()
                    else:
                        gtemp.remove_edge(node1, node2)
#                        nx.draw_networkx(gs)
#                        plt.show()
#                        nx.draw_networkx(gtemp)
#                        plt.show()
#                nx.draw_networkx(gtemp)
#                plt.show()
                
                # compute distance between \psi and the new generated graph.
#                knew = marginalizedkernel([gtemp, g1, g2], node_label='atom', edge_label=None,
#                               p_quit=lmbda, n_iteration=20, remove_totters=False,
#                               n_jobs=multiprocessing.cpu_count(), verbose=False)
                knew = compute_kernel([gtemp] + Gn_median, gkernel, verbose=False)
                dnew = dis_gstar(0, range(1, len(Gn_median) + 1), alpha, knew, 
                                 withterm3=False)
                if dnew <= dhat: # @todo: the new distance is smaller or also equal?
                    if dnew < dhat:
                        print('\nI am smaller!')
                        print('ig =', str(ig), ', l =', str(trail))
                        print(dhat, '->', dnew)
                        nb_updated += 1
                    elif dnew == dhat:                   
                        print('I am equal!') 
#                    nx.draw_networkx(gtemp)
#                    plt.show()
#                    print(gtemp.nodes(data=True))
#                    print(gtemp.edges(data=True))
                    dhat = dnew
                    gnew = gtemp.copy()
                    found = True # found better graph.                  
        if found:
            r = 0
            gihat_list = [gnew]
            dihat_list = [dhat]
        else:
            r += 1
            
        dis_of_each_itr.append(dhat)
        print('the shortest distances for previous iterations are', dis_of_each_itr)
#    dis_best.append(dhat)
    g_best = (g0hat_list[0] if len(gihat_list) == 0 else gihat_list[0])
    print('distances in kernel space:', dis_of_each_itr, '\n')
    
    return dhat, g_best, nb_updated
#    return 0, 0, 0


if __name__ == '__main__':
    from gklearn.utils.graphfiles import loadDataset
    
#    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
#          'extra_params': {}}  # node/edge symb
    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
          'extra_params': {}} # node nsymb
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/monoterpenoides/trainset_9.ds',
#          'extra_params': {}}
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/acyclic/dataset_bps.ds',
#            'extra_params': {}} # node symb
    
    DN, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    #DN = DN[0:10]
    
    lmbda = 0.03 # termination probalility
    r_max = 3 # 10 # iteration limit.
    l = 500
    alpha_range = np.linspace(0.5, 0.5, 1)
    #alpha_range = np.linspace(0.1, 0.9, 9)
    k = 10 # 5 # k nearest neighbors
    
    # randomly select two molecules
    #np.random.seed(1)
    #idx1, idx2 = np.random.randint(0, len(DN), 2)
    #g1 = DN[idx1]
    #g2 = DN[idx2]
    idx1 = 0
    idx2 = 6
    g1 = DN[idx1]
    g2 = DN[idx2]
    
    # compute 
    k_list = [] # kernel between each graph and itself.
    k_g1_list = [] # kernel between each graph and g1
    k_g2_list = [] # kernel between each graph and g2
    for ig, g in tqdm(enumerate(DN), desc='computing self kernels', file=sys.stdout): 
    #    ktemp = marginalizedkernel([g, g1, g2], node_label='atom', edge_label=None,
    #                               p_quit=lmbda, n_iteration=20, remove_totters=False,
    #                               n_jobs=multiprocessing.cpu_count(), verbose=False)
        ktemp = compute_kernel([g, g1, g2], 'untilhpathkernel', verbose=False)
        k_list.append(ktemp[0, 0])
        k_g1_list.append(ktemp[0, 1])
        k_g2_list.append(ktemp[0, 2])
    
    g_best = []
    dis_best = []
    # for each alpha
    for alpha in alpha_range:
        print('alpha =', alpha)
        # compute k nearest neighbors of phi in DN.
        dis_list = [] # distance between g_star and each graph.
        for ig, g in tqdm(enumerate(DN), desc='computing distances', file=sys.stdout):
            dtemp = k_list[ig] - 2 * (alpha * k_g1_list[ig] + (1 - alpha) * 
                          k_g2_list[ig]) + (alpha * alpha * k_list[idx1] + alpha * 
                          (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
                          k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2])
            dis_list.append(np.sqrt(dtemp))
        
        # sort
        sort_idx = np.argsort(dis_list)
        dis_gs = [dis_list[idis] for idis in sort_idx[0:k]]
        g0hat = DN[sort_idx[0]] # the nearest neighbor of phi in DN
        if dis_gs[0] == 0: # the exact pre-image.
            print('The exact pre-image is found from the input dataset.')
            g_pimg = g0hat
            break
        dhat = dis_gs[0] # the nearest distance
        Dk = [DN[ig] for ig in sort_idx[0:k]] # the k nearest neighbors
        gihat_list = []
        
        i = 1
        r = 1
        while r < r_max:
            print('r =', r)
            found = False
            for ig, gs in enumerate(Dk + gihat_list):
    #            nx.draw_networkx(gs)
    #            plt.show()
                # @todo what if the log is negetive?
                fdgs = int(np.abs(np.ceil(np.log(alpha * dis_gs[ig]))))
                for trail in tqdm(range(0, l), desc='l loop', file=sys.stdout):
                    # add and delete edges.
                    gtemp = gs.copy()
                    np.random.seed()
                    # which edges to change.
                    # @todo: should we use just half of the adjacency matrix for undirected graphs?
                    nb_vpairs = nx.number_of_nodes(gs) * (nx.number_of_nodes(gs) - 1)
                    # @todo: what if fdgs is bigger than nb_vpairs?
                    idx_change = random.sample(range(nb_vpairs), fdgs if fdgs < nb_vpairs else nb_vpairs)
    #                idx_change = np.random.randint(0, nx.number_of_nodes(gs) * 
    #                                               (nx.number_of_nodes(gs) - 1), fdgs)
                    for item in idx_change:
                        node1 = int(item / (nx.number_of_nodes(gs) - 1))
                        node2 = (item - node1 * (nx.number_of_nodes(gs) - 1))
                        if node2 >= node1: # skip the self pair.
                            node2 += 1
                        # @todo: is the randomness correct?
                        if not gtemp.has_edge(node1, node2):
                            # @todo: how to update the bond_type? 0 or 1?
                            gtemp.add_edges_from([(node1, node2, {'bond_type': 1})])
    #                        nx.draw_networkx(gs)
    #                        plt.show()
    #                        nx.draw_networkx(gtemp)
    #                        plt.show()
                        else:
                            gtemp.remove_edge(node1, node2)
    #                        nx.draw_networkx(gs)
    #                        plt.show()
    #                        nx.draw_networkx(gtemp)
    #                        plt.show()
    #                nx.draw_networkx(gtemp)
    #                plt.show()
                    
                    # compute distance between phi and the new generated graph.
    #                knew = marginalizedkernel([gtemp, g1, g2], node_label='atom', edge_label=None,
    #                               p_quit=lmbda, n_iteration=20, remove_totters=False,
    #                               n_jobs=multiprocessing.cpu_count(), verbose=False)
                    knew = compute_kernel([gtemp, g1, g2], 'untilhpathkernel', verbose=False)
                    dnew = np.sqrt(knew[0, 0] - 2 * (alpha * knew[0, 1] + (1 - alpha) * 
                          knew[0, 2]) + (alpha * alpha * k_list[idx1] + alpha * 
                          (1 - alpha) * k_g2_list[idx1] + (1 - alpha) * alpha * 
                          k_g1_list[idx2] + (1 - alpha) * (1 - alpha) * k_list[idx2]))
                    if dnew < dhat: # @todo: the new distance is smaller or also equal?
                        print('I am smaller!')
                        print(dhat, '->', dnew)
                        nx.draw_networkx(gtemp)
                        plt.show()
                        print(gtemp.nodes(data=True))
                        print(gtemp.edges(data=True))
                        dhat = dnew
                        gnew = gtemp.copy()
                        found = True # found better graph.
                        r = 0
                    elif dnew == dhat:                   
                        print('I am equal!')                   
            if found:
                gihat_list = [gnew]
                dis_gs.append(dhat)
            else:
                r += 1
        dis_best.append(dhat)
        g_best += ([g0hat] if len(gihat_list) == 0 else gihat_list)       
    
    
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', dis_best[idx])
        print('the corresponding pre-image is')
        nx.draw_networkx(g_best[idx])
        plt.show()