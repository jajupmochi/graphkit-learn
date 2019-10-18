#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:59:00 2019

@author: ljia
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
#from tqdm import tqdm

#import os
import sys
sys.path.insert(0, "../")
from pygraph.utils.graphfiles import loadDataset

from preimage_random import preimage_random
from ged import ged_median
from utils import compute_kernel, get_same_item_indices, remove_edges


###############################################################################
# tests on different values on grid of median-sets and k.

def test_preimage_random_grid_k_median_nb():    
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 5 # iteration limit for pre-image.
    l = 500 # update limit for random generation
#    alpha_range = np.linspace(0.5, 0.5, 1)
#    k = 5 # k nearest neighbors
    # parameters for GED function
    ged_cost='CHEM_1'
    ged_method='IPFP'
    saveGXL='gedlib'
    
    # number of graphs; we what to compute the median of these graphs. 
    nb_median_range = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    # number of nearest neighbors.
    k_range = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100]
    
    # find out all the graphs classified to positive group 1.
    idx_dict = get_same_item_indices(y_all)
    Gn = [Gn[i] for i in idx_dict[1]]
    
#    # compute Gram matrix.
#    time0 = time.time()
#    km = compute_kernel(Gn, gkernel, True)
#    time_km = time.time() - time0    
#    # write Gram matrix to file.
#    np.savez('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm', gm=km, gmtime=time_km)
        
    
    time_list = []
    dis_ks_min_list = []
    sod_gs_list = []
    sod_gs_min_list = []
    nb_updated_list = []
    g_best = []
    for idx_nb, nb_median in enumerate(nb_median_range):
        print('\n-------------------------------------------------------')
        print('number of median graphs =', nb_median)
        random.seed(1)
        idx_rdm = random.sample(range(len(Gn)), nb_median)
        print('graphs chosen:', idx_rdm)
        Gn_median = [Gn[idx].copy() for idx in idx_rdm]
        
#        for g in Gn_median:
#            nx.draw(g, labels=nx.get_node_attributes(g, 'atom'), with_labels=True)
##            plt.savefig("results/preimage_mix/mutag.png", format="PNG")
#            plt.show()
#            plt.clf()                         
                    
        ###################################################################
        gmfile = np.load('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm.npz')
        km_tmp = gmfile['gm']
        time_km = gmfile['gmtime']
        # modify mixed gram matrix.
        km = np.zeros((len(Gn) + nb_median, len(Gn) + nb_median))
        for i in range(len(Gn)):
            for j in range(i, len(Gn)):
                km[i, j] = km_tmp[i, j]
                km[j, i] = km[i, j]
        for i in range(len(Gn)):
            for j, idx in enumerate(idx_rdm):
                km[i, len(Gn) + j] = km[i, idx]
                km[len(Gn) + j, i] = km[i, idx]
        for i, idx1 in enumerate(idx_rdm):
            for j, idx2 in enumerate(idx_rdm):
                km[len(Gn) + i, len(Gn) + j] = km[idx1, idx2]
                
        ###################################################################
        alpha_range = [1 / nb_median] * nb_median
        
        time_list.append([])
        dis_ks_min_list.append([])
        sod_gs_list.append([])
        sod_gs_min_list.append([])
        nb_updated_list.append([])
        g_best.append([])   
        
        for k in k_range:
            print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            print('k =', k)
            time0 = time.time()
            dhat, ghat, nb_updated = preimage_random(Gn, Gn_median, alpha_range, 
                range(len(Gn), len(Gn) + nb_median), km, k, r_max, l, gkernel)
                
            time_total = time.time() - time0 + time_km
            print('time: ', time_total)
            time_list[idx_nb].append(time_total)
            print('\nsmallest distance in kernel space: ', dhat) 
            dis_ks_min_list[idx_nb].append(dhat)
            g_best[idx_nb].append(ghat)
            print('\nnumber of updates of the best graph: ', nb_updated)
            nb_updated_list[idx_nb].append(nb_updated)
            
            # show the best graph and save it to file.
            print('the shortest distance is', dhat)
            print('one of the possible corresponding pre-images is')
            nx.draw(ghat, labels=nx.get_node_attributes(ghat, 'atom'), 
                    with_labels=True)
            plt.savefig('results/preimage_random/mutag_median_nb' + str(nb_median) + 
                        '_k' + str(k) + '.png', format="PNG")
    #        plt.show()
            plt.clf()
    #        print(ghat_list[0].nodes(data=True))
    #        print(ghat_list[0].edges(data=True))
        
            # compute the corresponding sod in graph space.
            sod_tmp, _ = ged_median([ghat], Gn_median, ged_cost=ged_cost, 
                                         ged_method=ged_method, saveGXL=saveGXL)
            sod_gs_list[idx_nb].append(sod_tmp)
            sod_gs_min_list[idx_nb].append(np.min(sod_tmp))
            print('\nsmallest sod in graph space: ', np.min(sod_tmp))
        
    print('\nsods in graph space: ', sod_gs_list)
    print('\nsmallest sod in graph space for each set of median graphs and k: ', 
          sod_gs_min_list)  
    print('\nsmallest distance in kernel space for each set of median graphs and k: ', 
          dis_ks_min_list) 
    print('\nnumber of updates of the best graph for each set of median graphs and k by IAM: ', 
          nb_updated_list)
    print('\ntimes:', time_list)
    



###############################################################################
# tests on different numbers of median-sets.

def test_preimage_random_median_nb():
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 5 # iteration limit for pre-image.
    l = 500 # update limit for random generation
#    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 5 # k nearest neighbors
    # parameters for GED function
    ged_cost='CHEM_1'
    ged_method='IPFP'
    saveGXL='gedlib'
    
    # number of graphs; we what to compute the median of these graphs. 
    nb_median_range = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    
    # find out all the graphs classified to positive group 1.
    idx_dict = get_same_item_indices(y_all)
    Gn = [Gn[i] for i in idx_dict[1]]
    
#    # compute Gram matrix.
#    time0 = time.time()
#    km = compute_kernel(Gn, gkernel, True)
#    time_km = time.time() - time0    
#    # write Gram matrix to file.
#    np.savez('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm', gm=km, gmtime=time_km)
        
    
    time_list = []
    dis_ks_min_list = []
    sod_gs_list = []
    sod_gs_min_list = []
    nb_updated_list = []
    g_best = []
    for nb_median in nb_median_range:
        print('\n-------------------------------------------------------')
        print('number of median graphs =', nb_median)
        random.seed(1)
        idx_rdm = random.sample(range(len(Gn)), nb_median)
        print('graphs chosen:', idx_rdm)
        Gn_median = [Gn[idx].copy() for idx in idx_rdm]
        
#        for g in Gn_median:
#            nx.draw(g, labels=nx.get_node_attributes(g, 'atom'), with_labels=True)
##            plt.savefig("results/preimage_mix/mutag.png", format="PNG")
#            plt.show()
#            plt.clf()                         
                    
        ###################################################################
        gmfile = np.load('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm.npz')
        km_tmp = gmfile['gm']
        time_km = gmfile['gmtime']
        # modify mixed gram matrix.
        km = np.zeros((len(Gn) + nb_median, len(Gn) + nb_median))
        for i in range(len(Gn)):
            for j in range(i, len(Gn)):
                km[i, j] = km_tmp[i, j]
                km[j, i] = km[i, j]
        for i in range(len(Gn)):
            for j, idx in enumerate(idx_rdm):
                km[i, len(Gn) + j] = km[i, idx]
                km[len(Gn) + j, i] = km[i, idx]
        for i, idx1 in enumerate(idx_rdm):
            for j, idx2 in enumerate(idx_rdm):
                km[len(Gn) + i, len(Gn) + j] = km[idx1, idx2]
                
        ###################################################################
        alpha_range = [1 / nb_median] * nb_median
        time0 = time.time()
        dhat, ghat, nb_updated = preimage_random(Gn, Gn_median, alpha_range, 
            range(len(Gn), len(Gn) + nb_median), km, k, r_max, l, gkernel)
            
        time_total = time.time() - time0 + time_km
        print('time: ', time_total)
        time_list.append(time_total)
        print('\nsmallest distance in kernel space: ', dhat) 
        dis_ks_min_list.append(dhat)
        g_best.append(ghat)
        print('\nnumber of updates of the best graph: ', nb_updated)
        nb_updated_list.append(nb_updated)
        
        # show the best graph and save it to file.
        print('the shortest distance is', dhat)
        print('one of the possible corresponding pre-images is')
        nx.draw(ghat, labels=nx.get_node_attributes(ghat, 'atom'), 
                with_labels=True)
        plt.savefig('results/preimage_random/mutag_median_nb' + str(nb_median) + 
                    '.png', format="PNG")
#        plt.show()
        plt.clf()
#        print(ghat_list[0].nodes(data=True))
#        print(ghat_list[0].edges(data=True))
    
        # compute the corresponding sod in graph space.
        sod_tmp, _ = ged_median([ghat], Gn_median, ged_cost=ged_cost, 
                                     ged_method=ged_method, saveGXL=saveGXL)
        sod_gs_list.append(sod_tmp)
        sod_gs_min_list.append(np.min(sod_tmp))
        print('\nsmallest sod in graph space: ', np.min(sod_tmp))
        
    print('\nsods in graph space: ', sod_gs_list)
    print('\nsmallest sod in graph space for each set of median graphs: ', sod_gs_min_list)  
    print('\nsmallest distance in kernel space for each set of median graphs: ', 
          dis_ks_min_list) 
    print('\nnumber of updates of the best graph for each set of median graphs: ', 
          nb_updated_list)
    print('\ntimes:', time_list)
    
    

###############################################################################
# test on the combination of the two randomly chosen graphs. (the same as in the
# random pre-image paper.)
    
def test_random_preimage_2combination():
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:12]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
#    dis_mat, dis_max, dis_min, dis_mean = kernel_distance_matrix(Gn, gkernel=gkernel)
#    print(dis_max, dis_min, dis_mean)
    
    lmbda = 0.03 # termination probalility
    r_max = 10 # iteration limit for pre-image.
    l = 500
    alpha_range = np.linspace(0, 1, 11)
    k = 5 # k nearest neighbors
    
    # randomly select two molecules
    np.random.seed(1)
    idx_gi = [187, 167] # np.random.randint(0, len(Gn), 2)
    g1 = Gn[idx_gi[0]].copy()
    g2 = Gn[idx_gi[1]].copy()
    
#    nx.draw(g1, labels=nx.get_node_attributes(g1, 'atom'), with_labels=True)
#    plt.savefig("results/random_preimage/mutag10.png", format="PNG")
#    plt.show()
#    nx.draw(g2, labels=nx.get_node_attributes(g2, 'atom'), with_labels=True)
#    plt.savefig("results/random_preimage/mutag11.png", format="PNG")
#    plt.show()    
    
    ######################################################################
#    Gn_mix = [g.copy() for g in Gn]
#    Gn_mix.append(g1.copy())
#    Gn_mix.append(g2.copy())
#    
##    g_tmp = iam([g1, g2])
##    nx.draw_networkx(g_tmp)
##    plt.show()
#    
#    # compute 
#    time0 = time.time()
#    km = compute_kernel(Gn_mix, gkernel, True)
#    time_km = time.time() - time0
    
    ###################################################################
    idx1 = idx_gi[0]
    idx2 = idx_gi[1]
    gmfile = np.load('results/gram_matrix_marg_itr10_pq0.03.gm.npz')
    km = gmfile['gm']
    time_km = gmfile['gmtime']
    # modify mixed gram matrix.
    for i in range(len(Gn)):
        km[i, len(Gn)] = km[i, idx1]
        km[i, len(Gn) + 1] = km[i, idx2]
        km[len(Gn), i] = km[i, idx1]
        km[len(Gn) + 1, i] = km[i, idx2]
    km[len(Gn), len(Gn)] = km[idx1, idx1]
    km[len(Gn), len(Gn) + 1] = km[idx1, idx2]
    km[len(Gn) + 1, len(Gn)] = km[idx2, idx1]
    km[len(Gn) + 1, len(Gn) + 1] = km[idx2, idx2]
            
    ###################################################################

    time_list = []
    nb_updated_list = []
    g_best = []
    dis_ks_min_list = []
    # for each alpha
    for alpha in alpha_range:
        print('\n-------------------------------------------------------\n')
        print('alpha =', alpha)
        time0 = time.time()
        dhat, ghat, nb_updated = preimage_random(Gn, [g1, g2], [alpha, 1 - alpha], 
                                          range(len(Gn), len(Gn) + 2), km,
                                          k, r_max, l, gkernel)
        time_total = time.time() - time0 + time_km
        print('time: ', time_total)
        time_list.append(time_total)
        dis_ks_min_list.append(dhat)
        g_best.append(ghat)
        nb_updated_list.append(nb_updated)
        
    # show best graphs and save them to file.
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', dis_ks_min_list[idx])
        print('one of the possible corresponding pre-images is')
        nx.draw(g_best[idx], labels=nx.get_node_attributes(g_best[idx], 'atom'), 
                with_labels=True)
        plt.show()
        plt.savefig('results/random_preimage/mutag_alpha' + str(item) + '.png', format="PNG")
        plt.clf()
        print(g_best[idx].nodes(data=True))
        print(g_best[idx].edges(data=True))
            
#        # compute the corresponding sod in graph space. (alpha range not considered.)
#        sod_tmp, _ = median_distance(g_best[0], Gn_let)
#        sod_gs_list.append(sod_tmp)
#        sod_gs_min_list.append(np.min(sod_tmp))
#        sod_ks_min_list.append(sod_ks)
#        nb_updated_list.append(nb_updated)
                      
#    print('\nsmallest sod in graph space for each alpha: ', sod_gs_min_list)  
    print('\nsmallest distance in kernel space for each alpha: ', dis_ks_min_list) 
    print('\nnumber of updates for each alpha: ', nb_updated_list)             
    print('\ntimes:', time_list)
    
###############################################################################

    
if __name__ == '__main__':
###############################################################################
# test on the combination of the two randomly chosen graphs. (the same as in the
# random pre-image paper.)
#    test_random_preimage_2combination()
    
###############################################################################
# tests all algorithms on different numbers of median-sets.
    test_preimage_random_median_nb()
    
###############################################################################
# tests all algorithms on different values on grid of median-sets and k.
#    test_preimage_random_grid_k_median_nb()