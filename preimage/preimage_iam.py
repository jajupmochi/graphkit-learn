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
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import random

from iam import iam_upgraded
from utils import dis_gstar, compute_kernel


def preimage_iam(Gn_init, Gn_median, alpha, idx_gi, Kmatrix, k, r_max, 
                 gkernel, epsilon=0.001, InitIAMWithAllDk=False,
                 params_iam={'c_ei': 1, 'c_er': 1, 'c_es': 1, 
                             'ite_max': 50, 'epsilon': 0.001, 
                             'removeNodes': True, 'connected': False},
                 params_ged={'ged_cost': 'CHEM_1', 'ged_method': 'IPFP', 
                             'saveGXL': 'benoit'}):
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
    dis_all = [] # distance between g_star and each graph.
    term3 = 0
    for i1, a1 in enumerate(alpha):
        for i2, a2 in enumerate(alpha):
            term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    for ig, g in tqdm(enumerate(Gn_init), desc='computing distances', file=sys.stdout):
        dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix, term3=term3)
        dis_all.append(dtemp)
        
    # sort
    sort_idx = np.argsort(dis_all)
    dis_k = [dis_all[idis] for idis in sort_idx[0:k]] # the k shortest distances
    nb_best = len(np.argwhere(dis_k == dis_k[0]).flatten().tolist())
    ghat_list = [Gn_init[idx].copy() for idx in sort_idx[0:nb_best]] # the nearest neighbors of phi in DN
    if dis_k[0] == 0: # the exact pre-image.
        print('The exact pre-image is found from the input dataset.')
        return 0, ghat_list, 0, 0
    dhat = dis_k[0] # the nearest distance
#    for g in ghat_list:
#        draw_Letter_graph(g)
#        nx.draw_networkx(g)
#        plt.show()
#        print(g.nodes(data=True))
#        print(g.edges(data=True))
    Gk = [Gn_init[ig].copy() for ig in sort_idx[0:k]] # the k nearest neighbors
#    for gi in Gk:
#        nx.draw(gi, labels=nx.get_node_attributes(gi, 'atom'), with_labels=True)
##        nx.draw_networkx(gi)
#        plt.show()
##        draw_Letter_graph(g)
#        print(gi.nodes(data=True))
#        print(gi.edges(data=True))
    
#    i = 1
    r = 0
    itr_total = 0
    dis_of_each_itr = [dhat]
    found = False
    nb_updated = 0
    nb_updated_k = 0
    while r < r_max:# and not found: # @todo: if not found?# and np.abs(old_dis - cur_dis) > epsilon:
        print('\n-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
        print('Current preimage iteration =', r)
        print('Total preimage iteration =', itr_total, '\n')
        found = False
        
        Gn_nearest_median = [g.copy() for g in Gk]
        if InitIAMWithAllDk: # each graph in D_k is used to initialize IAM.
            ghat_new_list = []
            for g_tmp in Gk:
                Gn_nearest_init = [g_tmp.copy()]
                ghat_new_list_tmp, _ = iam_upgraded(Gn_nearest_median, 
                        Gn_nearest_init, params_ged=params_ged, **params_iam)
                ghat_new_list += ghat_new_list_tmp
        else: # only the best graph in D_k is used to initialize IAM.
            Gn_nearest_init = [g.copy() for g in Gk]
            ghat_new_list, _ = iam_upgraded(Gn_nearest_median, Gn_nearest_init, 
                    params_ged=params_ged, **params_iam)

#        for g in g_tmp_list:
#            nx.draw_networkx(g)
#            plt.show()
#            draw_Letter_graph(g)
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
            
        # compute distance between \psi and the new generated graphs.
        knew = compute_kernel(ghat_new_list + Gn_median, gkernel, False)
        dhat_new_list = []
        for idx, g_tmp in enumerate(ghat_new_list):
            # @todo: the term3 below could use the one at the beginning of the function.
            dhat_new_list.append(dis_gstar(idx, range(len(ghat_new_list), 
                                len(ghat_new_list) + len(Gn_median) + 1), 
                                alpha, knew, withterm3=False))
        
        for idx_g, ghat_new in enumerate(ghat_new_list):          
            dhat_new = dhat_new_list[idx_g]
            
            # if the new distance is smaller than the max of D_k.           
            if dhat_new < dis_k[-1] and np.abs(dhat_new - dis_k[-1]) >= epsilon:
                # check if the new distance is the same as one in D_k.
                is_duplicate = False
                for dis_tmp in dis_k[1:-1]:
                    if np.abs(dhat_new - dis_tmp) < epsilon:
                        is_duplicate = True
                        print('IAM: duplicate k nearest graph generated.')
                        break
                if not is_duplicate:
                    if np.abs(dhat_new - dhat) < epsilon:
                        print('IAM: I am equal!')
#                        dhat = dhat_new
#                        ghat_list = [ghat_new.copy()]
                    else:
                        print('IAM: we got better k nearest neighbors!')
                        nb_updated_k += 1
                        print('the k nearest neighbors are updated', 
                              nb_updated_k, 'times.')
                        
                        dis_k = [dhat_new] + dis_k[0:k-1] # add the new nearest distance.
                        Gk = [ghat_new.copy()] + Gk[0:k-1] # add the corresponding graph.
                        sort_idx = np.argsort(dis_k)
                        dis_k = [dis_k[idx] for idx in sort_idx[0:k]] # the new k nearest distances.
                        Gk = [Gk[idx] for idx in sort_idx[0:k]]
                        if dhat_new < dhat:
                            print('IAM: I have smaller distance!')
                            print(str(dhat) + '->' + str(dhat_new))
                            dhat = dhat_new
                            ghat_list = [Gk[0].copy()]
                            r = 0
                            nb_updated += 1
                        
                            print('the graph is updated', nb_updated, 'times.')                       
                            nx.draw(Gk[0], labels=nx.get_node_attributes(Gk[0], 'atom'), 
                                with_labels=True)
                    ##            plt.savefig("results/gk_iam/simple_two/xx" + str(i) + ".png", format="PNG")
                            plt.show()
                        
                        found = True
        if not found:
            r += 1            

        dis_of_each_itr.append(dhat)
        itr_total += 1
        print('\nthe k shortest distances are', dis_k)
        print('the shortest distances for previous iterations are', dis_of_each_itr)
        
    print('\n\nthe graph is updated', nb_updated, 'times.')
    print('\nthe k nearest neighbors are updated', nb_updated_k, 'times.')
    print('distances in kernel space:', dis_of_each_itr, '\n')
    
    return dhat, ghat_list, dis_of_each_itr[-1], nb_updated, nb_updated_k




def preimage_iam_random_mix(Gn_init, Gn_median, alpha, idx_gi, Kmatrix, k, r_max, 
                            l_max, gkernel, epsilon=0.001, 
                            InitIAMWithAllDk=False, InitRandomWithAllDk=True,
                            params_iam={'c_ei': 1, 'c_er': 1, 'c_es': 1, 
                                        'ite_max': 50, 'epsilon': 0.001, 
                                        'removeNodes': True, 'connected': False},
                            params_ged={'ged_cost': 'CHEM_1', 'ged_method': 'IPFP', 
                                        'saveGXL': 'benoit'}):
    """This function constructs graph pre-image by the iterative pre-image 
    framework in reference [1], algorithm 1, where new graphs are generated 
    randomly and by the IAM algorithm in reference [2].
    
    notes
    -----
    Every time a set of n better graphs is acquired, their distances in kernel space are
    compared with the k nearest ones, and the k nearest distances from the k+n
    distances will be used as the new ones.
    """
    Gn_init = [nx.convert_node_labels_to_integers(g) for g in Gn_init]
    # compute k nearest neighbors of phi in DN.
    dis_all = [] # distance between g_star and each graph.
    term3 = 0
    for i1, a1 in enumerate(alpha):
        for i2, a2 in enumerate(alpha):
            term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    for ig, g in tqdm(enumerate(Gn_init), desc='computing distances', file=sys.stdout):
        dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix, term3=term3)
        dis_all.append(dtemp)
        
    # sort
    sort_idx = np.argsort(dis_all)
    dis_k = [dis_all[idis] for idis in sort_idx[0:k]] # the k shortest distances
    nb_best = len(np.argwhere(dis_k == dis_k[0]).flatten().tolist())
    ghat_list = [Gn_init[idx].copy() for idx in sort_idx[0:nb_best]] # the nearest neighbors of psi in DN
    if dis_k[0] == 0: # the exact pre-image.
        print('The exact pre-image is found from the input dataset.')
        return 0, ghat_list, 0, 0
    dhat = dis_k[0] # the nearest distance
#    for g in ghat_list:
#        draw_Letter_graph(g)
#        nx.draw_networkx(g)
#        plt.show()
#        print(g.nodes(data=True))
#        print(g.edges(data=True))
    Gk = [Gn_init[ig].copy() for ig in sort_idx[0:k]] # the k nearest neighbors
#    for gi in Gk:
#        nx.draw(gi, labels=nx.get_node_attributes(gi, 'atom'), with_labels=True)
##        nx.draw_networkx(gi)
#        plt.show()
##        draw_Letter_graph(g)
#        print(gi.nodes(data=True))
#        print(gi.edges(data=True))
    
    r = 0
    itr_total = 0
    dis_of_each_itr = [dhat]
    nb_updated_iam = 0
    nb_updated_k_iam = 0
    nb_updated_random = 0
    nb_updated_k_random = 0
#    is_iam_duplicate = False
    while r < r_max: # and not found: # @todo: if not found?# and np.abs(old_dis - cur_dis) > epsilon:
        print('\n-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
        print('Current preimage iteration =', r)
        print('Total preimage iteration =', itr_total, '\n')
        found_iam = False

        Gn_nearest_median = [g.copy() for g in Gk]
        if InitIAMWithAllDk: # each graph in D_k is used to initialize IAM.
            ghat_new_list = []
            for g_tmp in Gk:
                Gn_nearest_init = [g_tmp.copy()]
                ghat_new_list_tmp, _ = iam_upgraded(Gn_nearest_median, 
                        Gn_nearest_init, params_ged=params_ged, **params_iam)
                ghat_new_list += ghat_new_list_tmp
        else: # only the best graph in D_k is used to initialize IAM.
            Gn_nearest_init = [g.copy() for g in Gk]
            ghat_new_list, _ = iam_upgraded(Gn_nearest_median, Gn_nearest_init, 
                    params_ged=params_ged, **params_iam)

#        for g in g_tmp_list:
#            nx.draw_networkx(g)
#            plt.show()
#            draw_Letter_graph(g)
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
            
        # compute distance between \psi and the new generated graphs.
        knew = compute_kernel(ghat_new_list + Gn_median, gkernel, False)
        dhat_new_list = []
        
        for idx, g_tmp in enumerate(ghat_new_list):
            # @todo: the term3 below could use the one at the beginning of the function.
            dhat_new_list.append(dis_gstar(idx, range(len(ghat_new_list), 
                            len(ghat_new_list) + len(Gn_median) + 1), 
                            alpha, knew, withterm3=False))
                
        # find the new k nearest graphs. 
        for idx_g, ghat_new in enumerate(ghat_new_list):          
            dhat_new = dhat_new_list[idx_g]
            
            # if the new distance is smaller than the max of D_k.           
            if dhat_new < dis_k[-1] and np.abs(dhat_new - dis_k[-1]) >= epsilon:
                # check if the new distance is the same as one in D_k.
                is_duplicate = False
                for dis_tmp in dis_k[1:-1]:
                    if np.abs(dhat_new - dis_tmp) < epsilon:
                        is_duplicate = True
                        print('IAM: duplicate k nearest graph generated.')
                        break
                if not is_duplicate:
                    if np.abs(dhat_new - dhat) < epsilon:
                        print('IAM: I am equal!')
#                        dhat = dhat_new
#                        ghat_list = [ghat_new.copy()]
                    else:
                        print('IAM: we got better k nearest neighbors!')
                        nb_updated_k_iam += 1
                        print('the k nearest neighbors are updated', 
                              nb_updated_k_iam, 'times.')
                        
                        dis_k = [dhat_new] + dis_k[0:k-1] # add the new nearest distance.
                        Gk = [ghat_new.copy()] + Gk[0:k-1] # add the corresponding graph.
                        sort_idx = np.argsort(dis_k)
                        dis_k = [dis_k[idx] for idx in sort_idx[0:k]] # the new k nearest distances.
                        Gk = [Gk[idx] for idx in sort_idx[0:k]]
                        if dhat_new < dhat:
                            print('IAM: I have smaller distance!')
                            print(str(dhat) + '->' + str(dhat_new))
                            dhat = dhat_new
                            ghat_list = [Gk[0].copy()]
                            r = 0
                            nb_updated_iam += 1
                        
                            print('the graph is updated by IAM', nb_updated_iam, 
                                  'times.')                       
                            nx.draw(Gk[0], labels=nx.get_node_attributes(Gk[0], 'atom'), 
                                with_labels=True)
                    ##            plt.savefig("results/gk_iam/simple_two/xx" + str(i) + ".png", format="PNG")
                            plt.show()
                        
                        found_iam = True
                        
        # when new distance is not smaller than the max of D_k, use random generation.
        if not found_iam:
            print('Distance not better, switching to random generation now.')
            print(str(dhat) + '->' + str(dhat_new))
            
            if InitRandomWithAllDk: # use all k nearest graphs as the initials.
                init_list = [g_init.copy() for g_init in Gk]
            else: # use just the nearest graph as the initial.
                init_list = [Gk[0].copy()]
            
            # number of edges to be changed.
            if len(init_list) == 1:
                # @todo what if the log is negetive? how to choose alpha (scalar)? seems fdgs is always 1.
    #            fdgs = dhat_new
                fdgs = nb_updated_random + 1
                if fdgs < 1:
                    fdgs = 1
                fdgs = int(np.ceil(np.log(fdgs)))
                if fdgs < 1:
                    fdgs += 1
    #            fdgs = nb_updated_random + 1 # @todo:
                fdgs_list = [fdgs]
            else:
                # @todo what if the log is negetive? how to choose alpha (scalar)?
                fdgs_list = np.array(dis_k[:])
                if np.min(fdgs_list) < 1:
                    fdgs_list /= dis_k[0]
                fdgs_list = [int(item) for item in np.ceil(np.log(fdgs_list))]
                if np.min(fdgs_list) < 1:
                    fdgs_list = np.array(fdgs_list) + 1
                
            l = 0
            found_random = False
            while l < l_max and not found_random:
                for idx_g, g_tmp in enumerate(init_list):
                    # add and delete edges.
                    ghat_new = nx.convert_node_labels_to_integers(g_tmp.copy())
                    # @todo: should we use just half of the adjacency matrix for undirected graphs?
                    nb_vpairs = nx.number_of_nodes(ghat_new) * (nx.number_of_nodes(ghat_new) - 1)
                    np.random.seed()
                    # which edges to change.                
                    # @todo: what if fdgs is bigger than nb_vpairs?
                    idx_change = random.sample(range(nb_vpairs), fdgs_list[idx_g] if 
                                               fdgs_list[idx_g] < nb_vpairs else nb_vpairs)
#                idx_change = np.random.randint(0, nx.number_of_nodes(gs) * 
#                                               (nx.number_of_nodes(gs) - 1), fdgs)
                    for item in idx_change:
                        node1 = int(item / (nx.number_of_nodes(ghat_new) - 1))
                        node2 = (item - node1 * (nx.number_of_nodes(ghat_new) - 1))
                        if node2 >= node1: # skip the self pair.
                            node2 += 1
                        # @todo: is the randomness correct?
                        if not ghat_new.has_edge(node1, node2):
                            ghat_new.add_edge(node1, node2)
    #                        nx.draw_networkx(gs)
    #                        plt.show()
    #                        nx.draw_networkx(ghat_new)
    #                        plt.show()
                        else:
                            ghat_new.remove_edge(node1, node2)
    #                        nx.draw_networkx(gs)
    #                        plt.show()
    #                        nx.draw_networkx(ghat_new)
    #                        plt.show()
    #                nx.draw_networkx(ghat_new)
    #                plt.show()
                            
                    # compute distance between \psi and the new generated graph.
                    knew = compute_kernel([ghat_new] + Gn_median, gkernel, verbose=False)
                    dhat_new = dis_gstar(0, range(1, len(Gn_median) + 1), 
                                         alpha, knew, withterm3=False)
                    # @todo: the new distance is smaller or also equal?
                    if dhat_new < dis_k[-1] and np.abs(dhat_new - dis_k[-1]) >= epsilon:
                        # check if the new distance is the same as one in D_k.
                        is_duplicate = False
                        for dis_tmp in dis_k[1:-1]:
                            if np.abs(dhat_new - dis_tmp) < epsilon:
                                is_duplicate = True
                                print('Random: duplicate k nearest graph generated.')
                                break
                        if not is_duplicate:
                            if np.abs(dhat_new - dhat) < epsilon:
                                print('Random: I am equal!')
        #                        dhat = dhat_new
        #                        ghat_list = [ghat_new.copy()]
                            else:
                                print('Random: we got better k nearest neighbors!')
                                print('l =', str(l))
                                nb_updated_k_random += 1
                                print('the k nearest neighbors are updated by random generation', 
                                          nb_updated_k_random, 'times.')
                                
                                dis_k = [dhat_new] + dis_k # add the new nearest distances.
                                Gk = [ghat_new.copy()] + Gk # add the corresponding graphs.
                                sort_idx = np.argsort(dis_k)
                                dis_k = [dis_k[idx] for idx in sort_idx[0:k]] # the new k nearest distances.
                                Gk = [Gk[idx] for idx in sort_idx[0:k]]
                                if dhat_new < dhat:
                                    print('\nRandom: I am smaller!')
                                    print('l =', str(l))
                                    print(dhat, '->', dhat_new)                       
                                    dhat = dhat_new
                                    ghat_list = [ghat_new.copy()]
                                    r = 0
                                    nb_updated_random += 1
        
                                    print('the graph is updated by random generation', 
                                          nb_updated_random, 'times.')
                                             
                                    nx.draw(ghat_new, labels=nx.get_node_attributes(ghat_new, 'atom'), 
                                        with_labels=True)
        ##            plt.savefig("results/gk_iam/simple_two/xx" + str(i) + ".png", format="PNG")
                                    plt.show()
                                found_random = True
                                break
                l += 1
            if not found_random: # l == l_max:
                r += 1            
            
        dis_of_each_itr.append(dhat)
        itr_total += 1
        print('\nthe k shortest distances are', dis_k)
        print('the shortest distances for previous iterations are', dis_of_each_itr)
        
    print('\n\nthe graph is updated by IAM', nb_updated_iam, 'times, and by random generation',
          nb_updated_random, 'times.')
    print('\nthe k nearest neighbors are updated by IAM', nb_updated_k_iam, 
          'times, and by random generation', nb_updated_k_random, 'times.')
    print('distances in kernel space:', dis_of_each_itr, '\n')
    
    return dhat, ghat_list, dis_of_each_itr[-1], \
            nb_updated_iam, nb_updated_random, nb_updated_k_iam, nb_updated_k_random


###############################################################################
# Old implementations.
    
#def gk_iam(Gn, alpha):
#    """This function constructs graph pre-image by the iterative pre-image 
#    framework in reference [1], algorithm 1, where the step of generating new 
#    graphs randomly is replaced by the IAM algorithm in reference [2].
#    
#    notes
#    -----
#    Every time a better graph is acquired, the older one is replaced by it.
#    """
#    pass
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
#        # compute distance between \psi and the new generated graph.
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


#def gk_iam_nearest(Gn, alpha, idx_gi, Kmatrix, k, r_max):
#    """This function constructs graph pre-image by the iterative pre-image 
#    framework in reference [1], algorithm 1, where the step of generating new 
#    graphs randomly is replaced by the IAM algorithm in reference [2].
#    
#    notes
#    -----
#    Every time a better graph is acquired, its distance in kernel space is
#    compared with the k nearest ones, and the k nearest distances from the k+1
#    distances will be used as the new ones.
#    """
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
#    g0hat = Gn[sort_idx[0]] # the nearest neighbor of phi in DN
#    if dis_gs[0] == 0: # the exact pre-image.
#        print('The exact pre-image is found from the input dataset.')
#        return 0, g0hat
#    dhat = dis_gs[0] # the nearest distance
#    ghat = g0hat.copy()
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
#        g_tmp = test_iam_with_more_graphs_as_init(Gs_nearest, Gs_nearest, c_ei=1, c_er=1, c_es=1)
#        nx.draw_networkx(g_tmp)
#        plt.show()
#        print(g_tmp.nodes(data=True))
#        print(g_tmp.edges(data=True))
#        
#        # compute distance between \psi and the new generated graph.
#        gi_list = [Gn[i] for i in idx_gi]
#        knew = compute_kernel([g_tmp] + gi_list, 'untilhpathkernel', False)
#        dnew = dis_gstar(0, range(1, len(gi_list) + 1), alpha, knew)
#        
##        dnew = knew[0, 0] - 2 * (alpha[0] * knew[0, 1] + alpha[1] * 
##              knew[0, 2]) + (alpha[0] * alpha[0] * k_list[0] + alpha[0] * 
##              alpha[1] * k_g2_list[0] + alpha[1] * alpha[0] * 
##              k_g1_list[1] + alpha[1] * alpha[1] * k_list[1])
#        if dnew <= dhat and g_tmp != ghat: # the new distance is smaller
#            print('I am smaller!')
#            print(str(dhat) + '->' + str(dnew))
##            nx.draw_networkx(ghat)
##            plt.show()
##            print('->')
##            nx.draw_networkx(g_tmp)
##            plt.show()
#            
#            dhat = dnew
#            g_new = g_tmp.copy() # found better graph.
#            ghat = g_tmp.copy()
#            dis_gs.append(dhat) # add the new nearest distance.
#            Gs_nearest.append(g_new) # add the corresponding graph.
#            sort_idx = np.argsort(dis_gs)
#            dis_gs = [dis_gs[idx] for idx in sort_idx[0:k]] # the new k nearest distances.
#            Gs_nearest = [Gs_nearest[idx] for idx in sort_idx[0:k]]
#            r = 0
#        else:
#            r += 1
#    
#    return dhat, ghat


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
#        # compute distance between \psi and the new generated graphs.
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