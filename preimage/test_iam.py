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
from iam import iam_upgraded
from utils import remove_edges, compute_kernel, get_same_item_indices
from ged import ged_median

###############################################################################
# tests on different numbers of median-sets.

def test_iam_median_nb():
    
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
#    lmbda = 0.03 # termination probalility
#    r_max = 10 # iteration limit for pre-image.
#    alpha_range = np.linspace(0.5, 0.5, 1)
#    k = 5 # k nearest neighbors
#    epsilon = 1e-6
#    InitIAMWithAllDk = True
    # parameters for GED function
    ged_cost='CHEM_1'
    ged_method='IPFP'
    saveGXL='gedlib'
    # parameters for IAM function
    c_ei=1
    c_er=1
    c_es=1
    ite_max_iam = 50
    epsilon_iam = 0.001
    removeNodes = False
    connected_iam = False
    
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
    nb_updated_k_list = []
    g_best = []
    for nb_median in nb_median_range:
        print('\n-------------------------------------------------------')
        print('number of median graphs =', nb_median)
        random.seed(1)
        idx_rdm = random.sample(range(len(Gn)), nb_median)
        print('graphs chosen:', idx_rdm)
        Gn_median = [Gn[idx].copy() for idx in idx_rdm]
        Gn_candidate = [g.copy() for g in Gn_median]
        
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
        ghat_new_list, dis_min = iam_upgraded(Gn_median, Gn_candidate, 
            c_ei=c_ei, c_er=c_er, c_es=c_es, ite_max=ite_max_iam, 
            epsilon=epsilon_iam, removeNodes=removeNodes, 
            connected=connected_iam, 
            params_ged={'ged_cost': ged_cost, 'ged_method': ged_method, 
                        'saveGXL': saveGXL})
            
        time_total = time.time() - time0
        print('\ntime: ', time_total)
        time_list.append(time_total)
        print('\nsmallest distance in kernel space: ', dhat) 
        dis_ks_min_list.append(dhat)
        g_best.append(ghat_list)
        print('\nnumber of updates of the best graph: ', nb_updated)
        nb_updated_list.append(nb_updated)
        print('\nnumber of updates of k nearest graphs: ', nb_updated_k)
        nb_updated_k_list.append(nb_updated_k)
        
        # show the best graph and save it to file.
        print('the shortest distance is', dhat)
        print('one of the possible corresponding pre-images is')
        nx.draw(ghat_list[0], labels=nx.get_node_attributes(ghat_list[0], 'atom'), 
                with_labels=True)
        plt.show()
        plt.savefig('results/preimage_iam/mutag_median_nb' + str(nb_median) + 
                    '.png', format="PNG")
        plt.clf()
#        print(ghat_list[0].nodes(data=True))
#        print(ghat_list[0].edges(data=True))
    
        # compute the corresponding sod in graph space.
        sod_tmp, _ = ged_median([ghat_list[0]], Gn_median, ged_cost=ged_cost, 
                                     ged_method=ged_method, saveGXL=saveGXL)
        sod_gs_list.append(sod_tmp)
        sod_gs_min_list.append(np.min(sod_tmp))
        print('\nsmallest sod in graph space: ', np.min(sod_tmp))
        
    print('\nsods in graph space: ', sod_gs_list)
    print('\nsmallest sod in graph space for each set of median graphs: ', sod_gs_min_list)  
    print('\nsmallest distance in kernel space for each set of median graphs: ', 
          dis_ks_min_list) 
    print('\nnumber of updates of the best graph for each set of median graphs by IAM: ', 
          nb_updated_list)
    print('\nnumber of updates of k nearest graphs for each set of median graphs by IAM: ', 
          nb_updated_k_list)
    print('\ntimes:', time_list)
    
    
###############################################################################

    
if __name__ == '__main__':
###############################################################################
# tests on different numbers of median-sets.
    test_iam_median_nb()