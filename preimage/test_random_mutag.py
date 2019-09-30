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
from tqdm import tqdm

import os
import sys
sys.path.insert(0, "../")
from pygraph.utils.graphfiles import loadDataset

###############################################################################
# test on the combination of the two randomly chosen graphs. (the same as in the
# random pre-image paper.)

def test_preimage_mix_2combination_all_pairs():
    from gk_iam import preimage_iam_random_mix, compute_kernel
    from iam import median_distance
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 10 # iteration limit for pre-image.
    l_max = 500 # update limit for random generation
    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 5 # k nearest neighbors
    epsilon = 1e-6
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
    removeNodes = True
    connected_iam = False
    
    nb_update_mat_iam = np.full((len(Gn), len(Gn)), np.inf)
    nb_update_mat_random = np.full((len(Gn), len(Gn)), np.inf)
    # test on each pair of graphs.
#    for idx1 in range(len(Gn) - 1, -1, -1):
#        for idx2 in range(idx1, -1, -1):
    for idx1 in range(187, 188):
        for idx2 in range(167, 168):
            g1 = Gn[idx1].copy()
            g2 = Gn[idx2].copy()
        #    Gn[10] = []
        #    Gn[10] = []
            
            nx.draw(g1, labels=nx.get_node_attributes(g1, 'atom'), with_labels=True)
            plt.savefig("results/preimage_mix/mutag187.png", format="PNG")
            plt.show()
            plt.clf()
            nx.draw(g2, labels=nx.get_node_attributes(g2, 'atom'), with_labels=True)
            plt.savefig("results/preimage_mix/mutag167.png", format="PNG")
            plt.show()
            plt.clf()

            ###################################################################            
#            Gn_mix = [g.copy() for g in Gn]
#            Gn_mix.append(g1.copy())
#            Gn_mix.append(g2.copy())
#            
#            # compute
#            time0 = time.time()
#            km = compute_kernel(Gn_mix, gkernel, True)
#            time_km = time.time() - time0
#            
#            # write Gram matrix to file and read it.
#            np.savez('results/gram_matrix_uhpath_itr7_pq0.8.gm', gm=km, gmtime=time_km)
            
            ###################################################################
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
#            # use only the two graphs in median set as candidates.
#            Gn = [g1.copy(), g2.copy()]
#            Gn_mix = Gn + [g1.copy(), g2.copy()]
#            # compute         
#            time0 = time.time()
#            km = compute_kernel(Gn_mix, gkernel, True)
#            time_km = time.time() - time0
    
            
            time_list = []
            dis_ks_min_list = []
            sod_gs_list = []
            sod_gs_min_list = []
            nb_updated_list_iam = []
            nb_updated_list_random = []
            nb_updated_k_list_iam = []
            nb_updated_k_list_random = []
            g_best = []
            # for each alpha
            for alpha in alpha_range:
                print('\n-------------------------------------------------------\n')
                print('alpha =', alpha)
                time0 = time.time()
                dhat, ghat_list, dis_of_each_itr, nb_updated_iam, nb_updated_random, \
                    nb_updated_k_iam, nb_updated_k_random = \
                    preimage_iam_random_mix(Gn, [g1, g2],
                    [alpha, 1 - alpha], range(len(Gn), len(Gn) + 2), km, k, r_max, 
                    l_max, gkernel, epsilon=epsilon, 
                    params_iam={'c_ei': c_ei, 'c_er': c_er, 'c_es': c_es, 
                                'ite_max': ite_max_iam, 'epsilon': epsilon_iam,
                                'removeNodes': removeNodes, 'connected': connected_iam},
                    params_ged={'ged_cost': ged_cost, 'ged_method': ged_method, 
                                'saveGXL': saveGXL})
                time_total = time.time() - time0 + time_km
                print('time: ', time_total)
                time_list.append(time_total)
                dis_ks_min_list.append(dhat)
                g_best.append(ghat_list)
                nb_updated_list_iam.append(nb_updated_iam)       
                nb_updated_list_random.append(nb_updated_random)
                nb_updated_k_list_iam.append(nb_updated_k_iam)       
                nb_updated_k_list_random.append(nb_updated_k_random) 
                
            # show best graphs and save them to file.
            for idx, item in enumerate(alpha_range):
                print('when alpha is', item, 'the shortest distance is', dis_ks_min_list[idx])
                print('one of the possible corresponding pre-images is')
                nx.draw(g_best[idx][0], labels=nx.get_node_attributes(g_best[idx][0], 'atom'), 
                        with_labels=True)
                plt.savefig('results/preimage_mix/mutag' + str(idx1) + '_' + str(idx2) 
                            + '_alpha' + str(item) + '.png', format="PNG")
#                plt.show()
                plt.clf()
#                print(g_best[idx][0].nodes(data=True))
#                print(g_best[idx][0].edges(data=True))
                
        #        for g in g_best[idx]:
        #            draw_Letter_graph(g, savepath='results/gk_iam/')
        ##            nx.draw_networkx(g)
        ##            plt.show()
        #            print(g.nodes(data=True))
        #            print(g.edges(data=True))
                    
            # compute the corresponding sod in graph space.
            for idx, item in enumerate(alpha_range):
                sod_tmp, _ = median_distance(g_best[0], [g1, g2], ged_cost=ged_cost, 
                                             ged_method=ged_method, saveGXL=saveGXL)
                sod_gs_list.append(sod_tmp)
                sod_gs_min_list.append(np.min(sod_tmp))
                
            print('\nsods in graph space: ', sod_gs_list)
            print('\nsmallest sod in graph space for each alpha: ', sod_gs_min_list)  
            print('\nsmallest distance in kernel space for each alpha: ', dis_ks_min_list) 
            print('\nnumber of updates of the best graph for each alpha by IAM: ', nb_updated_list_iam)
            print('\nnumber of updates of the best graph for each alpha by random generation: ', 
                  nb_updated_list_random)
            print('\nnumber of updates of k nearest graphs for each alpha by IAM: ', 
                  nb_updated_k_list_iam)
            print('\nnumber of updates of k nearest graphs for each alpha by random generation: ', 
                  nb_updated_k_list_random)
            print('\ntimes:', time_list)
            nb_update_mat_iam[idx1, idx2] = nb_updated_list_iam[0]
            nb_update_mat_random[idx1, idx2] = nb_updated_list_random[0]
            
            str_fw = 'graphs %d and %d: %d times by IAM, %d times by random generation.\n' \
                % (idx1, idx2, nb_updated_list_iam[0], nb_updated_list_random[0])
            with open('results/preimage_mix/nb_updates.txt', 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write(str_fw + content)
                
                

def test_gkiam_2combination_all_pairs():
    from gk_iam import gk_iam_nearest_multi, compute_kernel
    from iam import median_distance
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 10 # iteration limit for pre-image.
    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 10 # k nearest neighbors
    epsilon = 1e-6
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
    removeNodes = True
    connected_iam = False
    
    nb_update_mat = np.full((len(Gn), len(Gn)), np.inf)
    # test on each pair of graphs.
#    for idx1 in range(len(Gn) - 1, -1, -1):
#        for idx2 in range(idx1, -1, -1):
    for idx1 in range(187, 188):
        for idx2 in range(167, 168):
            g1 = Gn[idx1].copy()
            g2 = Gn[idx2].copy()
        #    Gn[10] = []
        #    Gn[10] = []
            
            nx.draw(g1, labels=nx.get_node_attributes(g1, 'atom'), with_labels=True)
            plt.savefig("results/gk_iam/all_pairs/mutag187.png", format="PNG")
            plt.show()
            plt.clf()
            nx.draw(g2, labels=nx.get_node_attributes(g2, 'atom'), with_labels=True)
            plt.savefig("results/gk_iam/all_pairs/mutag167.png", format="PNG")
            plt.show()
            plt.clf()

            ###################################################################            
#            Gn_mix = [g.copy() for g in Gn]
#            Gn_mix.append(g1.copy())
#            Gn_mix.append(g2.copy())
#            
#            # compute
#            time0 = time.time()
#            km = compute_kernel(Gn_mix, gkernel, True)
#            time_km = time.time() - time0
#            
#            # write Gram matrix to file and read it.
#            np.savez('results/gram_matrix_uhpath_itr7_pq0.8.gm', gm=km, gmtime=time_km)
            
            ###################################################################
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
#            # use only the two graphs in median set as candidates.
#            Gn = [g1.copy(), g2.copy()]
#            Gn_mix = Gn + [g1.copy(), g2.copy()]
#            # compute         
#            time0 = time.time()
#            km = compute_kernel(Gn_mix, gkernel, True)
#            time_km = time.time() - time0
    
            
            time_list = []
            dis_ks_min_list = []
            sod_gs_list = []
            sod_gs_min_list = []
            nb_updated_list = []
            nb_updated_k_list = [] 
            g_best = []
            # for each alpha
            for alpha in alpha_range:
                print('\n-------------------------------------------------------\n')
                print('alpha =', alpha)
                time0 = time.time()
                dhat, ghat_list, sod_ks, nb_updated, nb_updated_k = \
                    gk_iam_nearest_multi(Gn, [g1, g2],
                    [alpha, 1 - alpha], range(len(Gn), len(Gn) + 2), km, k, r_max, 
                    gkernel, epsilon=epsilon, 
                    params_iam={'c_ei': c_ei, 'c_er': c_er, 'c_es': c_es, 
                                'ite_max': ite_max_iam, 'epsilon': epsilon_iam,
                                'removeNodes': removeNodes, 'connected': connected_iam},
                    params_ged={'ged_cost': ged_cost, 'ged_method': ged_method, 
                                'saveGXL': saveGXL})
                time_total = time.time() - time0 + time_km
                print('time: ', time_total)
                time_list.append(time_total)
                dis_ks_min_list.append(dhat)
                g_best.append(ghat_list)
                nb_updated_list.append(nb_updated)
                nb_updated_k_list.append(nb_updated_k)
                
            # show best graphs and save them to file.
            for idx, item in enumerate(alpha_range):
                print('when alpha is', item, 'the shortest distance is', dis_ks_min_list[idx])
                print('one of the possible corresponding pre-images is')
                nx.draw(g_best[idx][0], labels=nx.get_node_attributes(g_best[idx][0], 'atom'), 
                        with_labels=True)
                plt.savefig('results/gk_iam/mutag' + str(idx1) + '_' + str(idx2) 
                            + '_alpha' + str(item) + '.png', format="PNG")
#                plt.show()
                plt.clf()
#                print(g_best[idx][0].nodes(data=True))
#                print(g_best[idx][0].edges(data=True))
                
        #        for g in g_best[idx]:
        #            draw_Letter_graph(g, savepath='results/gk_iam/')
        ##            nx.draw_networkx(g)
        ##            plt.show()
        #            print(g.nodes(data=True))
        #            print(g.edges(data=True))
                    
            # compute the corresponding sod in graph space.
            for idx, item in enumerate(alpha_range):
                sod_tmp, _ = median_distance(g_best[0], [g1, g2], ged_cost=ged_cost, 
                                             ged_method=ged_method, saveGXL=saveGXL)
                sod_gs_list.append(sod_tmp)
                sod_gs_min_list.append(np.min(sod_tmp))
                
            print('\nsods in graph space: ', sod_gs_list)
            print('\nsmallest sod in graph space for each alpha: ', sod_gs_min_list)  
            print('\nsmallest distance in kernel space for each alpha: ', dis_ks_min_list) 
            print('\nnumber of updates of the best graph for each alpha: ', 
                  nb_updated_list)
            print('\nnumber of updates of the k nearest graphs for each alpha: ', 
                  nb_updated_k_list)
            print('\ntimes:', time_list)
            nb_update_mat[idx1, idx2] = nb_updated_list[0]
            
            str_fw = 'graphs %d and %d: %d.\n' % (idx1, idx2, nb_updated_list[0])
            with open('results/gk_iam/all_pairs/nb_updates.txt', 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write(str_fw + content)
    
    

def test_gkiam_2combination():
    from gk_iam import gk_iam_nearest_multi, compute_kernel
    from iam import median_distance
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
    lmbda = 0.03 # termination probalility
    r_max = 10 # iteration limit for pre-image.
    alpha_range = np.linspace(0.5, 0.5, 1)
    k = 20 # k nearest neighbors
    epsilon = 1e-6
    ged_cost='CHEM_1'
    ged_method='IPFP'
    saveGXL='gedlib'
    c_ei=1
    c_er=1
    c_es=1
    
    # randomly select two molecules
    np.random.seed(1)
    idx_gi = [10, 11] # np.random.randint(0, len(Gn), 2)
    g1 = Gn[idx_gi[0]].copy()
    g2 = Gn[idx_gi[1]].copy()
#    Gn[10] = []
#    Gn[10] = []
    
#    nx.draw(g1, labels=nx.get_node_attributes(g1, 'atom'), with_labels=True)
#    plt.savefig("results/random_preimage/mutag10.png", format="PNG")
#    plt.show()
#    nx.draw(g2, labels=nx.get_node_attributes(g2, 'atom'), with_labels=True)
#    plt.savefig("results/random_preimage/mutag11.png", format="PNG")
#    plt.show() 
    
    Gn_mix = [g.copy() for g in Gn]
    Gn_mix.append(g1.copy())
    Gn_mix.append(g2.copy())
    
    # compute
#    time0 = time.time()
#    km = compute_kernel(Gn_mix, gkernel, True)
#    time_km = time.time() - time0
    
    # write Gram matrix to file and read it.
#    np.savez('results/gram_matrix.gm', gm=km, gmtime=time_km)
    gmfile = np.load('results/gram_matrix.gm.npz')
    km = gmfile['gm']
    time_km = gmfile['gmtime']
    
    time_list = []
    dis_ks_min_list = []
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
        dis_ks_min_list.append(dhat)
        g_best.append(ghat_list)
        nb_updated_list.append(nb_updated)       
        
    # show best graphs and save them to file.
    for idx, item in enumerate(alpha_range):
        print('when alpha is', item, 'the shortest distance is', dis_ks_min_list[idx])
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
        sod_tmp, _ = median_distance(g_best[0], [g1, g2], ged_cost=ged_cost, 
                                     ged_method=ged_method, saveGXL=saveGXL)
        sod_gs_list.append(sod_tmp)
        sod_gs_min_list.append(np.min(sod_tmp))
        
    print('\nsods in graph space: ', sod_gs_list)
    print('\nsmallest sod in graph space for each alpha: ', sod_gs_min_list)  
    print('\nsmallest distance in kernel space for each alpha: ', dis_ks_min_list) 
    print('\nnumber of updates for each alpha: ', nb_updated_list)             
    print('\ntimes:', time_list)
    
    
    
    
def test_random_preimage_2combination():
#    from gk_iam import compute_kernel
    from preimage import random_preimage
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
        dhat, ghat, nb_updated = random_preimage(Gn, [g1, g2], [alpha, 1 - alpha], 
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
        plt.savefig('results/random_preimage/mutag_alpha' + str(item) + '.png', format="PNG")
        plt.show()
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
# help functions

def remove_edges(Gn):
    for G in Gn:
        for _, _, attrs in G.edges(data=True):
            attrs.clear()
            
            
def kernel_distance_matrix(Gn, Kmatrix=None, gkernel=None):
    from gk_iam import compute_kernel
    dis_mat = np.empty((len(Gn), len(Gn)))
    if Kmatrix == None:
        Kmatrix = compute_kernel(Gn, gkernel, True)
    for i in range(len(Gn)):
        for j in range(i, len(Gn)):
            dis = Kmatrix[i, i] + Kmatrix[j, j] - 2 * Kmatrix[i, j]
            if dis < 0:
                if dis > -1e-10:
                    dis = 0
                else:
                    raise ValueError('The distance is negative.')
            dis_mat[i, j] = np.sqrt(dis)
            dis_mat[j, i] = dis_mat[i, j]
    dis_max = np.max(np.max(dis_mat))
    dis_min = np.min(np.min(dis_mat[dis_mat != 0]))
    dis_mean = np.mean(np.mean(dis_mat))
    return dis_mat, dis_max, dis_min, dis_mean
    
###############################################################################

    
if __name__ == '__main__':
###############################################################################
# test on the combination of the two randomly chosen graphs. (the same as in the
# random pre-image paper.)
#    test_random_preimage_2combination()
#    test_gkiam_2combination()
    test_gkiam_2combination_all_pairs()
#    test_preimage_mix_2combination_all_pairs()