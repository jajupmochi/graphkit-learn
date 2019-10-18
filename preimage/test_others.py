#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:20:16 2019

@author: ljia
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import sys
sys.path.insert(0, "../")
from pygraph.utils.graphfiles import loadDataset
from median import draw_Letter_graph
from ged import GED, ged_median
from utils import get_same_item_indices, compute_kernel, gram2distances, \
    dis_gstar, remove_edges


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

#    g1 = nx.Graph(name='haha')
#    g1.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'O'}), (2, {'atom': 'C'})])
#    g1.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'})])
#    g2 = nx.Graph(name='hahaha')
#    g2.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'O'}), (2, {'atom': 'C'}),
#                       (3, {'atom': 'O'}), (4, {'atom': 'C'})])
#    g2.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'}),
#                       (2, 3, {'bond_type': '1'}), (3, 4, {'bond_type': '1'})])
    
    g1 = nx.Graph(name='haha')
    g1.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'C'}), (2, {'atom': 'C'}),
                       (3, {'atom': 'S'}), (4, {'atom': 'S'})])
    g1.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'}),
                       (2, 3, {'bond_type': '1'}), (2, 4, {'bond_type': '1'})])
    g2 = nx.Graph(name='hahaha')
    g2.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'C'}), (2, {'atom': 'C'}),
                       (3, {'atom': 'O'}), (4, {'atom': 'O'})])
    g2.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'}),
                       (2, 3, {'bond_type': '1'}), (2, 4, {'bond_type': '1'})])

#    g2 = g1.copy()
#    g2.add_nodes_from([(3, {'atom': 'O'})])
#    g2.add_nodes_from([(4, {'atom': 'C'})])
#    g2.add_edges_from([(1, 3, {'bond_type': '1'})])
#    g2.add_edges_from([(3, 4, {'bond_type': '1'})])

#    del Gn[idx_gi[0]]
#    del Gn[idx_gi[1] - 1]
    
    nx.draw_networkx(g1)
    plt.show()
    print(g1.nodes(data=True))
    print(g1.edges(data=True))
    nx.draw_networkx(g2)
    plt.show()
    print(g2.nodes(data=True))
    print(g2.edges(data=True))
    
    g_median = test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations([g1, g2], [g1, g2], c_ei=1, c_er=1, c_es=1)
#    g_median = test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations(Gn, Gn, c_ei=1, c_er=1, c_es=1)
    nx.draw_networkx(g_median)
    plt.show()
    print(g_median.nodes(data=True))
    print(g_median.edges(data=True))
    
    
def test_the_simple_two(Gn, gkernel):
    from gk_iam import gk_iam_nearest_multi
    lmbda = 0.03 # termination probalility
    r_max = 10 # recursions
    l = 500
    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 2 # k nearest neighbors
    
    # randomly select two molecules
    np.random.seed(1)
    idx_gi = [0, 6] # np.random.randint(0, len(Gn), 2)
    g1 = Gn[idx_gi[0]]
    g2 = Gn[idx_gi[1]]
    Gn_mix = [g.copy() for g in Gn]
    Gn_mix.append(g1.copy())
    Gn_mix.append(g2.copy())
    
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
        
    km = compute_kernel(Gn_mix, gkernel, True)
#    k_list = np.diag(km) # kernel between each graph and itself.
#    k_g1_list = km[idx_gi[0]] # kernel between each graph and g1
#    k_g2_list = km[idx_gi[1]] # kernel between each graph and g2    

    g_best = []
    dis_best = []
    # for each alpha
    for alpha in alpha_range:
        print('alpha =', alpha)
        dhat, ghat_list = gk_iam_nearest_multi(Gn, [g1, g2], [alpha, 1 - alpha], 
                                               range(len(Gn), len(Gn) + 2), km,
                                               k, r_max,gkernel)
        dis_best.append(dhat)
        g_best.append(ghat_list)
        
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', dis_best[idx])
        print('the corresponding pre-images are')
        for g in g_best[idx]:
            nx.draw_networkx(g)
            plt.show()
            print(g.nodes(data=True))
            print(g.edges(data=True))
            
    
def test_remove_bests(Gn, gkernel):
    from gk_iam import gk_iam_nearest_multi
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
    # remove the best 2 graphs.
    del Gn[idx_gi[0]]
    del Gn[idx_gi[1] - 1]
#    del Gn[8]
    
    Gn_mix = [g.copy() for g in Gn]
    Gn_mix.append(g1.copy())
    Gn_mix.append(g2.copy())

    
    # compute
    km = compute_kernel(Gn_mix, gkernel, True)
    g_best = []
    dis_best = []
    # for each alpha
    for alpha in alpha_range:
        print('alpha =', alpha)
        dhat, ghat_list = gk_iam_nearest_multi(Gn, [g1, g2], [alpha, 1 - alpha], 
                                               range(len(Gn), len(Gn) + 2), km, 
                                               k, r_max, gkernel)
        dis_best.append(dhat)
        g_best.append(ghat_list)
        
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', dis_best[idx])
        print('the corresponding pre-images are')
        for g in g_best[idx]:
            draw_Letter_graph(g)
#            nx.draw_networkx(g)
#            plt.show()
            print(g.nodes(data=True))
            print(g.edges(data=True))
            
            
###############################################################################
# Tests on dataset Letter-H.
            
def test_gkiam_letter_h():
    from gk_iam import gk_iam_nearest_multi
    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
          'extra_params': {}} # node nsymb
#    ds = {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt',
#          'extra_params': {}} # node nsymb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    gkernel = 'structuralspkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 3 # recursions
#    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 10 # k nearest neighbors
    
    # classify graphs according to letters.
    idx_dict = get_same_item_indices(y_all)
    time_list = []
    sod_ks_min_list = []
    sod_gs_list = []
    sod_gs_min_list = []
    nb_updated_list = []
    for letter in idx_dict:
        print('\n-------------------------------------------------------\n')
        Gn_let = [Gn[i].copy() for i in idx_dict[letter]]
        Gn_mix = Gn_let + [g.copy() for g in Gn_let]
        
        alpha_range = np.linspace(1 / len(Gn_let), 1 / len(Gn_let), 1)
        
        # compute
        time0 = time.time()
        km = compute_kernel(Gn_mix, gkernel, True)
        g_best = []
        dis_best = []
        # for each alpha
        for alpha in alpha_range:
            print('alpha =', alpha)
            dhat, ghat_list, sod_ks, nb_updated = gk_iam_nearest_multi(Gn_let, 
                Gn_let, [alpha] * len(Gn_let), range(len(Gn_let), len(Gn_mix)), 
                km, k, r_max, gkernel, c_ei=1.7, c_er=1.7, c_es=1.7,
                ged_cost='LETTER', ged_method='IPFP', saveGXL='gedlib-letter')
            dis_best.append(dhat)
            g_best.append(ghat_list)
        time_list.append(time.time() - time0)
            
        # show best graphs and save them to file.
        for idx, item in enumerate(alpha_range):
            print('when alpha is', item, 'the shortest distance is', dis_best[idx])
            print('the corresponding pre-images are')
            for g in g_best[idx]:
                draw_Letter_graph(g, savepath='results/gk_iam/')
#            nx.draw_networkx(g)
#            plt.show()
                print(g.nodes(data=True))
                print(g.edges(data=True))
                
        # compute the corresponding sod in graph space. (alpha range not considered.)
        sod_tmp, _ = ged_median(g_best[0], Gn_let, ged_cost='LETTER', 
                                     ged_method='IPFP', saveGXL='gedlib-letter')
        sod_gs_list.append(sod_tmp)
        sod_gs_min_list.append(np.min(sod_tmp))
        sod_ks_min_list.append(sod_ks)
        nb_updated_list.append(nb_updated)
        
                
    print('\nsods in graph space: ', sod_gs_list)
    print('\nsmallest sod in graph space for each letter: ', sod_gs_min_list)  
    print('\nsmallest sod in kernel space for each letter: ', sod_ks_min_list) 
    print('\nnumber of updates for each letter: ', nb_updated_list)             
    print('\ntimes:', time_list)

#def compute_letter_median_by_average(Gn):
#    return g_median
    

def test_iam_letter_h():
    from iam import test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations
    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
          'extra_params': {}} # node nsymb
#    ds = {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt',
#          'extra_params': {}} # node nsymb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    
    lmbda = 0.03 # termination probalility
#    alpha_range = np.linspace(0.5, 0.5, 1)
    
    # classify graphs according to letters.
    idx_dict = get_same_item_indices(y_all)
    time_list = []
    sod_list = []
    sod_min_list = []
    for letter in idx_dict:        
        Gn_let = [Gn[i].copy() for i in idx_dict[letter]]
        
        alpha_range = np.linspace(1 / len(Gn_let), 1 / len(Gn_let), 1)
        
        # compute
        g_best = []
        dis_best = []
        time0 = time.time()
        # for each alpha
        for alpha in alpha_range:
            print('alpha =', alpha)
            ghat_list, dhat = test_iam_moreGraphsAsInit_tryAllPossibleBestGraphs_deleteNodesInIterations(
                Gn_let, Gn_let, c_ei=1.7, c_er=1.7, c_es=1.7,
                ged_cost='LETTER', ged_method='IPFP', saveGXL='gedlib-letter')
            dis_best.append(dhat)
            g_best.append(ghat_list)
        time_list.append(time.time() - time0)
            
        # show best graphs and save them to file.
        for idx, item in enumerate(alpha_range):
            print('when alpha is', item, 'the shortest distance is', dis_best[idx])
            print('the corresponding pre-images are')
            for g in g_best[idx]:
                draw_Letter_graph(g, savepath='results/iam/')
#            nx.draw_networkx(g)
#            plt.show()
                print(g.nodes(data=True))
                print(g.edges(data=True))
                
        # compute the corresponding sod in kernel space. (alpha range not considered.)
        gkernel = 'structuralspkernel'        
        sod_tmp = []
        Gn_mix = g_best[0] + Gn_let
        km = compute_kernel(Gn_mix, gkernel, True)
        for ig, g in tqdm(enumerate(g_best[0]), desc='computing kernel sod', file=sys.stdout):
            dtemp = dis_gstar(ig, range(len(g_best[0]), len(Gn_mix)), 
                              [alpha_range[0]] * len(Gn_let), km, withterm3=False)
            sod_tmp.append(dtemp)
        sod_list.append(sod_tmp)
        sod_min_list.append(np.min(sod_tmp))
        
                
    print('\nsods in kernel space: ', sod_list)
    print('\nsmallest sod in kernel space for each letter: ', sod_min_list)
    print('\ntimes:', time_list)
    
    
def test_random_preimage_letter_h():
    from preimage_random import preimage_random
    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
          'extra_params': {}} # node nsymb
#    ds = {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt',
#          'extra_params': {}} # node nsymb
    #    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
#          'extra_params': {}}  # node/edge symb
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/monoterpenoides/trainset_9.ds',
#          'extra_params': {}}
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/acyclic/dataset_bps.ds',
#            'extra_params': {}} # node symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    gkernel = 'structuralspkernel'
    
#    lmbda = 0.03 # termination probalility
    r_max = 3 # 10 # recursions
    l = 500
#    alpha_range = np.linspace(0.5, 0.5, 1)
    #alpha_range = np.linspace(0.1, 0.9, 9)
    k = 10 # 5 # k nearest neighbors
    
    # classify graphs according to letters.
    idx_dict = get_same_item_indices(y_all)
    time_list = []
    sod_list = []
    sod_min_list = []
    for letter in idx_dict:
        print('\n-------------------------------------------------------\n')
        Gn_let = [Gn[i].copy() for i in idx_dict[letter]]
        Gn_mix = Gn_let + [g.copy() for g in Gn_let]
        
        alpha_range = np.linspace(1 / len(Gn_let), 1 / len(Gn_let), 1)
        
        # compute
        time0 = time.time()
        km = compute_kernel(Gn_mix, gkernel, True)
        g_best = []
        dis_best = []
        # for each alpha
        for alpha in alpha_range:
            print('alpha =', alpha)
            dhat, ghat_list = preimage_random(Gn_let, Gn_let, [alpha] * len(Gn_let), 
                                                   range(len(Gn_let), len(Gn_mix)), km, 
                                                   k, r_max, gkernel, c_ei=1.7, 
                                                   c_er=1.7, c_es=1.7)
            dis_best.append(dhat)
            g_best.append(ghat_list)
        time_list.append(time.time() - time0)
            
        # show best graphs and save them to file.
        for idx, item in enumerate(alpha_range):
            print('when alpha is', item, 'the shortest distance is', dis_best[idx])
            print('the corresponding pre-images are')
            for g in g_best[idx]:
                draw_Letter_graph(g, savepath='results/gk_iam/')
#            nx.draw_networkx(g)
#            plt.show()
                print(g.nodes(data=True))
                print(g.edges(data=True))
                
        # compute the corresponding sod in graph space. (alpha range not considered.)
        sod_tmp, _ = ged_median(g_best[0], Gn_let)
        sod_list.append(sod_tmp)
        sod_min_list.append(np.min(sod_tmp))
        
                
    print('\nsods in graph space: ', sod_list)
    print('\nsmallest sod in graph space for each letter: ', sod_min_list)               
    print('\ntimes:', time_list)
    
    

    
    
    
    
def test_gkiam_mutag():
    from gk_iam import gk_iam_nearest_multi
    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
          'extra_params': {}} # node nsymb
#    ds = {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt',
#          'extra_params': {}} # node nsymb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    gkernel = 'structuralspkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 3 # recursions
#    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 20 # k nearest neighbors
    
    # classify graphs according to letters.
    idx_dict = get_same_item_indices(y_all)
    time_list = []
    sod_ks_min_list = []
    sod_gs_list = []
    sod_gs_min_list = []
    nb_updated_list = []
    for letter in idx_dict:
        print('\n-------------------------------------------------------\n')
        Gn_let = [Gn[i].copy() for i in idx_dict[letter]]
        Gn_mix = Gn_let + [g.copy() for g in Gn_let]
        
        alpha_range = np.linspace(1 / len(Gn_let), 1 / len(Gn_let), 1)
        
        # compute
        time0 = time.time()
        km = compute_kernel(Gn_mix, gkernel, True)
        g_best = []
        dis_best = []
        # for each alpha
        for alpha in alpha_range:
            print('alpha =', alpha)
            dhat, ghat_list, sod_ks, nb_updated = gk_iam_nearest_multi(Gn_let, Gn_let, [alpha] * len(Gn_let), 
                                                   range(len(Gn_let), len(Gn_mix)), km, 
                                                   k, r_max, gkernel, c_ei=1.7, 
                                                   c_er=1.7, c_es=1.7)
            dis_best.append(dhat)
            g_best.append(ghat_list)
        time_list.append(time.time() - time0)
            
        # show best graphs and save them to file.
        for idx, item in enumerate(alpha_range):
            print('when alpha is', item, 'the shortest distance is', dis_best[idx])
            print('the corresponding pre-images are')
            for g in g_best[idx]:
                draw_Letter_graph(g, savepath='results/gk_iam/')
#            nx.draw_networkx(g)
#            plt.show()
                print(g.nodes(data=True))
                print(g.edges(data=True))
                
        # compute the corresponding sod in graph space. (alpha range not considered.)
        sod_tmp, _ = ged_median(g_best[0], Gn_let)
        sod_gs_list.append(sod_tmp)
        sod_gs_min_list.append(np.min(sod_tmp))
        sod_ks_min_list.append(sod_ks)
        nb_updated_list.append(nb_updated)
        
                
    print('\nsods in graph space: ', sod_gs_list)
    print('\nsmallest sod in graph space for each letter: ', sod_gs_min_list)  
    print('\nsmallest sod in kernel space for each letter: ', sod_ks_min_list) 
    print('\nnumber of updates for each letter: ', nb_updated_list)             
    print('\ntimes:', time_list)
    
    
###############################################################################
# Re-test.
    
def retest_the_simple_two():
    from gk_iam import gk_iam_nearest_multi
    
    # The two simple graphs.
#    g1 = nx.Graph(name='haha')
#    g1.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'O'}), (2, {'atom': 'C'})])
#    g1.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'})])
#    g2 = nx.Graph(name='hahaha')
#    g2.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'O'}), (2, {'atom': 'C'}),
#                       (3, {'atom': 'O'}), (4, {'atom': 'C'})])
#    g2.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'}),
#                       (2, 3, {'bond_type': '1'}), (3, 4, {'bond_type': '1'})])
    
    g1 = nx.Graph(name='haha')
    g1.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'C'}), (2, {'atom': 'C'}),
                       (3, {'atom': 'S'}), (4, {'atom': 'S'})])
    g1.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'}),
                       (2, 3, {'bond_type': '1'}), (2, 4, {'bond_type': '1'})])
    g2 = nx.Graph(name='hahaha')
    g2.add_nodes_from([(0, {'atom': 'C'}), (1, {'atom': 'C'}), (2, {'atom': 'C'}),
                       (3, {'atom': 'O'}), (4, {'atom': 'O'})])
    g2.add_edges_from([(0, 1, {'bond_type': '1'}), (1, 2, {'bond_type': '1'}),
                       (2, 3, {'bond_type': '1'}), (2, 4, {'bond_type': '1'})])
    
#    # randomly select two molecules
#    np.random.seed(1)
#    idx_gi = [0, 6] # np.random.randint(0, len(Gn), 2)
#    g1 = Gn[idx_gi[0]]
#    g2 = Gn[idx_gi[1]]
#    Gn_mix = [g.copy() for g in Gn]
#    Gn_mix.append(g1.copy())
#    Gn_mix.append(g2.copy())
    
    Gn = [g1.copy(), g2.copy()]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 10 # recursions
#    l = 500
    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 2 # k nearest neighbors
    epsilon = 1e-6
    ged_cost='CHEM_1'
    ged_method='IPFP'
    saveGXL='gedlib'
    c_ei=1
    c_er=1
    c_es=1
    
    Gn_mix = Gn + [g1.copy(), g2.copy()]
    
    # compute         
    time0 = time.time()
    km = compute_kernel(Gn_mix, gkernel, True)
    time_km = time.time() - time0

    time_list = []
    sod_ks_min_list = []
    sod_gs_list = []
    sod_gs_min_list = []
    nb_updated_list = []       
    g_best = []
    # for each alpha
    for alpha in alpha_range:
        print('\n-------------------------------------------------------\n')
        print('alpha =', alpha)
        time0 = time.time()
        dhat, ghat_list, sod_ks, nb_updated = gk_iam_nearest_multi(Gn, [g1, g2],
            [alpha, 1 - alpha], range(len(Gn), len(Gn) + 2), km, k, r_max, 
            gkernel, c_ei=c_ei, c_er=c_er, c_es=c_es, epsilon=epsilon, 
            ged_cost=ged_cost, ged_method=ged_method, saveGXL=saveGXL)
        time_total = time.time() - time0 + time_km
        print('time: ', time_total)
        time_list.append(time_total)
        sod_ks_min_list.append(dhat)
        g_best.append(ghat_list)
        nb_updated_list.append(nb_updated)       
        
    # show best graphs and save them to file.
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', sod_ks_min_list[idx])
        print('one of the possible corresponding pre-images is')
        nx.draw(g_best[idx][0], labels=nx.get_node_attributes(g_best[idx][0], 'atom'), 
                with_labels=True)
        plt.savefig('results/gk_iam/mutag_alpha' + str(item) + '.png', format="PNG")
        plt.show()
        print(g_best[idx][0].nodes(data=True))
        print(g_best[idx][0].edges(data=True))
        
#        for g in g_best[idx]:
#            draw_Letter_graph(g, savepath='results/gk_iam/')
##            nx.draw_networkx(g)
##            plt.show()
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
            
    # compute the corresponding sod in graph space.
    for idx, item in enumerate(alpha_range):
        sod_tmp, _ = ged_median(g_best[0], [g1, g2], ged_cost=ged_cost, 
                                     ged_method=ged_method, saveGXL=saveGXL)
        sod_gs_list.append(sod_tmp)
        sod_gs_min_list.append(np.min(sod_tmp))
        
    print('\nsods in graph space: ', sod_gs_list)
    print('\nsmallest sod in graph space for each alpha: ', sod_gs_min_list)  
    print('\nsmallest sod in kernel space for each alpha: ', sod_ks_min_list) 
    print('\nnumber of updates for each alpha: ', nb_updated_list)             
    print('\ntimes:', time_list)
            
        

if __name__ == '__main__':
#    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
#          'extra_params': {}}  # node/edge symb
#    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
#          'extra_params': {}} # node nsymb
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/monoterpenoides/trainset_9.ds',
#          'extra_params': {}}
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/acyclic/dataset_bps.ds',
#        'extra_params': {}} # node symb
#    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:20]
    
#    import networkx.algorithms.isomorphism as iso
#    G1 = nx.MultiDiGraph()
#    G2 = nx.MultiDiGraph()
#    G1.add_nodes_from([1,2,3], fill='red')
#    G2.add_nodes_from([10,20,30,40], fill='red')
#    nx.add_path(G1, [1,2,3,4], weight=3, linewidth=2.5)
#    nx.add_path(G2, [10,20,30,40], weight=3)
#    nm = iso.categorical_node_match('fill', 'red')
#    print(nx.is_isomorphic(G1, G2, node_match=nm))
#    
#    test_new_IAM_allGraph_deleteNodes(Gn)
#    test_will_IAM_give_the_median_graph_we_wanted(Gn)
#    test_who_is_the_closest_in_GED_space(Gn)
#    test_who_is_the_closest_in_kernel_space(Gn)
    
#    test_the_simple_two(Gn, 'untilhpathkernel')
#    test_remove_bests(Gn, 'untilhpathkernel')
#    test_gkiam_letter_h()
#    test_iam_letter_h()
#    test_random_preimage_letter_h
    
###############################################################################
# retests.
    retest_the_simple_two()