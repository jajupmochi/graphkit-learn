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
from gklearn.utils.graphfiles import loadDataset
#from gklearn.utils.logger2file import *
from iam import iam_upgraded
from utils import remove_edges, compute_kernel, get_same_item_indices, dis_gstar
#from ged import ged_median


def test_iam_monoterpenoides_with_init40():
    gkernel = 'untilhpathkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
    # unfitted edit costs.
    c_vi = 3
    c_vr = 3
    c_vs = 1
    c_ei = 3
    c_er = 3
    c_es = 1
    ite_max_iam = 50
    epsilon_iam = 0.0001
    removeNodes = False
    connected_iam = False
    # parameters for IAM function
#    ged_cost = 'CONSTANT'
    ged_cost = 'CONSTANT'
    ged_method = 'IPFP'
    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
    ged_stabilizer = None
#    ged_repeat = 50
    algo_options = '--threads 8 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
                  'edit_cost_constant': edit_cost_constant, 
                  'algo_options': algo_options,
                  'stabilizer': ged_stabilizer}

    
    collection_path = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/generated_datsets/monoterpenoides/'
    graph_dir = collection_path + 'gxl/'
    y_all = ['3', '1', '4', '6', '7', '8', '9', '2']
    repeats = 50
    
    # classify graphs according to classes.
    time_list = []
    dis_ks_min_list = []
    dis_ks_set_median_list = []
    sod_gs_list = []
    g_best = []
    sod_set_median_list = []
    sod_list_list = []
    for y in y_all:
        print('\n-------------------------------------------------------')
        print('class of y:', y)
        
        time_list.append([])
        dis_ks_min_list.append([])
        dis_ks_set_median_list.append([])
        sod_gs_list.append([])
        g_best.append([])
        sod_set_median_list.append([])
        
        for repeat in range(repeats):
            # load median set.
            collection_file = collection_path + 'monoterpenoides_' + y + '_' + str(repeat) + '.xml'
            Gn_median, _ = loadDataset(collection_file, extra_params=graph_dir)
            Gn_candidate = [g.copy() for g in Gn_median]
            
            time0 = time.time()
            G_gen_median_list, sod_gen_median, sod_list, G_set_median_list, sod_set_median \
            = iam_upgraded(Gn_median, 
                Gn_candidate, c_ei=c_ei, c_er=c_er, c_es=c_es, ite_max=ite_max_iam,
                epsilon=epsilon_iam, node_label=node_label, edge_label=edge_label, 
                connected=connected_iam, removeNodes=removeNodes, 
                params_ged=params_ged)
            time_total = time.time() - time0
            print('\ntime: ', time_total)
            time_list[-1].append(time_total)
            g_best[-1].append(G_gen_median_list[0])
            sod_set_median_list[-1].append(sod_set_median)
            print('\nsmallest sod of the set median:', sod_set_median)
            sod_gs_list[-1].append(sod_gen_median)
            print('\nsmallest sod in graph space:', sod_gen_median)
            sod_list_list.append(sod_list)
            
#            # show the best graph and save it to file.
#            print('one of the possible corresponding pre-images is')
#            nx.draw(G_gen_median_list[0], labels=nx.get_node_attributes(G_gen_median_list[0], 'atom'), 
#                    with_labels=True)
##            plt.show()
#    #        plt.savefig('results/iam/mutag_median.fit_costs2.001.nb' + str(nb_median) + 
##            plt.savefig('results/iam/paper_compare/monoter_y' + str(y_class) + 
##                        '_repeat' + str(repeat) + '_' + str(time.time()) +
##                        '.png', format="PNG")
#            plt.clf()
#    #        print(G_gen_median_list[0].nodes(data=True))
#    #        print(G_gen_median_list[0].edges(data=True))
            
        print('\nsods of the set median for this class:', sod_set_median_list[-1])
        print('\nsods in graph space for this class:', sod_gs_list[-1])
#        print('\ndistance in kernel space of set median for this class:', 
#              dis_ks_set_median_list[-1])
#        print('\nsmallest distances in kernel space for this class:', 
#              dis_ks_min_list[-1])   
        print('\ntimes for this class:', time_list[-1])
        
        sod_set_median_list[-1] = np.mean(sod_set_median_list[-1])
        sod_gs_list[-1] = np.mean(sod_gs_list[-1])
#        dis_ks_set_median_list[-1] = np.mean(dis_ks_set_median_list[-1])
#        dis_ks_min_list[-1] = np.mean(dis_ks_min_list[-1])
        time_list[-1] = np.mean(time_list[-1])
        
    print()
    print('\nmean sods of the set median for each class:', sod_set_median_list)
    print('\nmean sods in graph space for each class:', sod_gs_list)
#    print('\ndistances in kernel space of set median for each class:', 
#            dis_ks_set_median_list)
#    print('\nmean smallest distances in kernel space for each class:', 
#            dis_ks_min_list)
    print('\nmean times for each class:', time_list)
    
    print('\nmean sods of the set median of all:', np.mean(sod_set_median_list))
    print('\nmean sods in graph space of all:', np.mean(sod_gs_list))
#    print('\nmean distances in kernel space of set median of all:', 
#            np.mean(dis_ks_set_median_list))
#    print('\nmean smallest distances in kernel space of all:', 
#            np.mean(dis_ks_min_list))
    print('\nmean times of all:', np.mean(time_list))




def test_iam_monoterpenoides():
    ds = {'name': 'monoterpenoides', 
          'dataset': '../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
    gkernel = 'untilhpathkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
    
    # parameters for GED function from the IAM paper.
    # fitted edit costs (Gaussian).
    c_vi = 0.03620133402089074
    c_vr = 0.0417574590207099
    c_vs = 0.009992282328587499
    c_ei = 0.08293120042342755
    c_er = 0.09512220476358019
    c_es = 0.09222529696841467
#    # fitted edit costs (linear combinations).
#    c_vi = 0.1749684054238749
#    c_vr = 0.0734054228711457
#    c_vs = 0.05017781726016715
#    c_ei = 0.1869431164806936
#    c_er = 0.32055856948274
#    c_es = 0.2569469379247611
#    # unfitted edit costs.
#    c_vi = 3
#    c_vr = 3
#    c_vs = 1
#    c_ei = 3
#    c_er = 3
#    c_es = 1
    ite_max_iam = 50
    epsilon_iam = 0.001
    removeNodes = False
    connected_iam = False
    # parameters for IAM function
#    ged_cost = 'CONSTANT'
    ged_cost = 'CONSTANT'
    ged_method = 'IPFP'
    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
#    edit_cost_constant = []
    ged_stabilizer = 'min'
    ged_repeat = 50
    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
                  'edit_cost_constant': edit_cost_constant, 
                  'stabilizer': ged_stabilizer, 'repeat': ged_repeat}
    
    # classify graphs according to letters.
    time_list = []
    dis_ks_min_list = []
    dis_ks_set_median_list = []
    sod_gs_list = []
    g_best = []
    sod_set_median_list = []
    sod_list_list = []
    idx_dict = get_same_item_indices(y_all)
    for y_class in idx_dict:
        print('\n-------------------------------------------------------')
        print('class of y:', y_class)
        Gn_class = [Gn[i].copy() for i in idx_dict[y_class]]
        
        time_list.append([])
        dis_ks_min_list.append([])
        dis_ks_set_median_list.append([])
        sod_gs_list.append([])
        g_best.append([])
        sod_set_median_list.append([])
        
        for repeat in range(50):
            idx_rdm = random.sample(range(len(Gn_class)), 10)
            print('graphs chosen:', idx_rdm)
            Gn_median = [Gn_class[idx].copy() for idx in idx_rdm]
            Gn_candidate = [g.copy() for g in Gn_median]
        
            alpha_range = [1 / len(Gn_median)] * len(Gn_median)
            time0 = time.time()
            G_gen_median_list, sod_gen_median, sod_list, G_set_median_list, sod_set_median \
            = iam_upgraded(Gn_median, 
                Gn_candidate, c_ei=c_ei, c_er=c_er, c_es=c_es, ite_max=ite_max_iam,
                epsilon=epsilon_iam, connected=connected_iam, removeNodes=removeNodes, 
                params_ged=params_ged)
            time_total = time.time() - time0
            print('\ntime: ', time_total)
            time_list[-1].append(time_total)
            g_best[-1].append(G_gen_median_list[0])
            sod_set_median_list[-1].append(sod_set_median)
            print('\nsmallest sod of the set median:', sod_set_median)
            sod_gs_list[-1].append(sod_gen_median)
            print('\nsmallest sod in graph space:', sod_gen_median)
            sod_list_list.append(sod_list)
            
            # show the best graph and save it to file.
            print('one of the possible corresponding pre-images is')
            nx.draw(G_gen_median_list[0], labels=nx.get_node_attributes(G_gen_median_list[0], 'atom'), 
                    with_labels=True)
#            plt.show()
    #        plt.savefig('results/iam/mutag_median.fit_costs2.001.nb' + str(nb_median) + 
#            plt.savefig('results/iam/paper_compare/monoter_y' + str(y_class) + 
#                        '_repeat' + str(repeat) + '_' + str(time.time()) +
#                        '.png', format="PNG")
            plt.clf()
    #        print(G_gen_median_list[0].nodes(data=True))
    #        print(G_gen_median_list[0].edges(data=True))
            
    
            # compute distance between \psi and the set median graph.
            knew_set_median = compute_kernel(G_set_median_list + Gn_median, 
                gkernel, node_label, edge_label, False)
            dhat_new_set_median_list = []
            for idx, g_tmp in enumerate(G_set_median_list):
                # @todo: the term3 below could use the one at the beginning of the function.
                dhat_new_set_median_list.append(dis_gstar(idx, range(len(G_set_median_list), 
                    len(G_set_median_list) + len(Gn_median) + 1), 
                    alpha_range, knew_set_median, withterm3=False))
                
            print('\ndistance in kernel space of set median: ', dhat_new_set_median_list[0]) 
            dis_ks_set_median_list[-1].append(dhat_new_set_median_list[0])
            
            
            # compute distance between \psi and the new generated graphs.
            knew = compute_kernel(G_gen_median_list + Gn_median, gkernel, node_label,
                              edge_label, False)
            dhat_new_list = []
            for idx, g_tmp in enumerate(G_gen_median_list):
                # @todo: the term3 below could use the one at the beginning of the function.
                dhat_new_list.append(dis_gstar(idx, range(len(G_gen_median_list), 
                                    len(G_gen_median_list) + len(Gn_median) + 1), 
                                    alpha_range, knew, withterm3=False))
                
            print('\nsmallest distance in kernel space: ', dhat_new_list[0]) 
            dis_ks_min_list[-1].append(dhat_new_list[0])
            

        print('\nsods of the set median for this class:', sod_set_median_list[-1])
        print('\nsods in graph space for this class:', sod_gs_list[-1])
        print('\ndistance in kernel space of set median for this class:', 
              dis_ks_set_median_list[-1])
        print('\nsmallest distances in kernel space for this class:', 
              dis_ks_min_list[-1])   
        print('\ntimes for this class:', time_list[-1])
        
        sod_set_median_list[-1] = np.mean(sod_set_median_list[-1])
        sod_gs_list[-1] = np.mean(sod_gs_list[-1])
        dis_ks_set_median_list[-1] = np.mean(dis_ks_set_median_list[-1])
        dis_ks_min_list[-1] = np.mean(dis_ks_min_list[-1])
        time_list[-1] = np.mean(time_list[-1])
        
    print()
    print('\nmean sods of the set median for each class:', sod_set_median_list)
    print('\nmean sods in graph space for each class:', sod_gs_list)
    print('\ndistances in kernel space of set median for each class:', 
            dis_ks_set_median_list)
    print('\nmean smallest distances in kernel space for each class:', 
            dis_ks_min_list)
    print('\nmean times for each class:', time_list)
    
    print('\nmean sods of the set median of all:', np.mean(sod_set_median_list))
    print('\nmean sods in graph space of all:', np.mean(sod_gs_list))
    print('\nmean distances in kernel space of set median of all:', 
            np.mean(dis_ks_set_median_list))
    print('\nmean smallest distances in kernel space of all:', 
            np.mean(dis_ks_min_list))
    print('\nmean times of all:', np.mean(time_list))
    
    nb_better_sods = 0
    nb_worse_sods = 0
    nb_same_sods = 0
    for sods in sod_list_list:
        if sods[0] > sods[-1]:
            nb_better_sods += 1
        elif sods[0] < sods[-1]:
            nb_worse_sods += 1
        else:
            nb_same_sods += 1
    print('\n In', str(len(sod_list_list)), 'sod lists,', str(nb_better_sods), 
          'are getting better,', str(nb_worse_sods), 'are getting worse,', 
          str(nb_same_sods), 'are not changed; ', str(nb_better_sods / len(sod_list_list)),
          'sods are improved.')
    
    
def test_iam_mutag():
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    gkernel = 'untilhpathkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
    
    # parameters for GED function from the IAM paper.
    # fitted edit costs.
    c_vi = 0.03523843108436513
    c_vr = 0.03347339739350128
    c_vs = 0.06871290673612238
    c_ei = 0.08591999846720685
    c_er = 0.07962086440894103
    c_es = 0.08596855855478233
    # unfitted edit costs.
#    c_vi = 3
#    c_vr = 3
#    c_vs = 1
#    c_ei = 3
#    c_er = 3
#    c_es = 1
    ite_max_iam = 50
    epsilon_iam = 0.001
    removeNodes = False
    connected_iam = False
    # parameters for IAM function
#    ged_cost = 'CONSTANT'
    ged_cost = 'CONSTANT'
    ged_method = 'IPFP'
    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
#    edit_cost_constant = []
    ged_stabilizer = 'min'
    ged_repeat = 50
    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
                  'edit_cost_constant': edit_cost_constant, 
                  'stabilizer': ged_stabilizer, 'repeat': ged_repeat}
    
    # classify graphs according to letters.
    time_list = []
    dis_ks_min_list = []
    dis_ks_set_median_list = []
    sod_gs_list = []
    g_best = []
    sod_set_median_list = []
    sod_list_list = []
    idx_dict = get_same_item_indices(y_all)
    for y_class in idx_dict:
        print('\n-------------------------------------------------------')
        print('class of y:', y_class)
        Gn_class = [Gn[i].copy() for i in idx_dict[y_class]]
        
        time_list.append([])
        dis_ks_min_list.append([])
        dis_ks_set_median_list.append([])
        sod_gs_list.append([])
        g_best.append([])
        sod_set_median_list.append([])
        
        for repeat in range(50):
            idx_rdm = random.sample(range(len(Gn_class)), 10)
            print('graphs chosen:', idx_rdm)
            Gn_median = [Gn_class[idx].copy() for idx in idx_rdm]
            Gn_candidate = [g.copy() for g in Gn_median]
        
            alpha_range = [1 / len(Gn_median)] * len(Gn_median)
            time0 = time.time()
            G_gen_median_list, sod_gen_median, sod_list, G_set_median_list, sod_set_median \
            = iam_upgraded(Gn_median, 
                Gn_candidate, c_ei=c_ei, c_er=c_er, c_es=c_es, ite_max=ite_max_iam,
                epsilon=epsilon_iam, connected=connected_iam, removeNodes=removeNodes, 
                params_ged=params_ged)
            time_total = time.time() - time0
            print('\ntime: ', time_total)
            time_list[-1].append(time_total)
            g_best[-1].append(G_gen_median_list[0])
            sod_set_median_list[-1].append(sod_set_median)
            print('\nsmallest sod of the set median:', sod_set_median)
            sod_gs_list[-1].append(sod_gen_median)
            print('\nsmallest sod in graph space:', sod_gen_median)
            sod_list_list.append(sod_list)
            
            # show the best graph and save it to file.
            print('one of the possible corresponding pre-images is')
            nx.draw(G_gen_median_list[0], labels=nx.get_node_attributes(G_gen_median_list[0], 'atom'), 
                    with_labels=True)
#            plt.show()
    #        plt.savefig('results/iam/mutag_median.fit_costs2.001.nb' + str(nb_median) + 
#            plt.savefig('results/iam/paper_compare/mutag_y' + str(y_class) + 
#                        '_repeat' + str(repeat) + '_' + str(time.time()) +
#                        '.png', format="PNG")
            plt.clf()
    #        print(G_gen_median_list[0].nodes(data=True))
    #        print(G_gen_median_list[0].edges(data=True))
            
    
            # compute distance between \psi and the set median graph.
            knew_set_median = compute_kernel(G_set_median_list + Gn_median, 
                gkernel, node_label, edge_label, False)
            dhat_new_set_median_list = []
            for idx, g_tmp in enumerate(G_set_median_list):
                # @todo: the term3 below could use the one at the beginning of the function.
                dhat_new_set_median_list.append(dis_gstar(idx, range(len(G_set_median_list), 
                    len(G_set_median_list) + len(Gn_median) + 1), 
                    alpha_range, knew_set_median, withterm3=False))
                
            print('\ndistance in kernel space of set median: ', dhat_new_set_median_list[0]) 
            dis_ks_set_median_list[-1].append(dhat_new_set_median_list[0])
            
            
            # compute distance between \psi and the new generated graphs.
            knew = compute_kernel(G_gen_median_list + Gn_median, gkernel, node_label,
                              edge_label, False)
            dhat_new_list = []
            for idx, g_tmp in enumerate(G_gen_median_list):
                # @todo: the term3 below could use the one at the beginning of the function.
                dhat_new_list.append(dis_gstar(idx, range(len(G_gen_median_list), 
                                    len(G_gen_median_list) + len(Gn_median) + 1), 
                                    alpha_range, knew, withterm3=False))
                
            print('\nsmallest distance in kernel space: ', dhat_new_list[0]) 
            dis_ks_min_list[-1].append(dhat_new_list[0])
            

        print('\nsods of the set median for this class:', sod_set_median_list[-1])
        print('\nsods in graph space for this class:', sod_gs_list[-1])
        print('\ndistance in kernel space of set median for this class:', 
              dis_ks_set_median_list[-1])
        print('\nsmallest distances in kernel space for this class:', 
              dis_ks_min_list[-1])   
        print('\ntimes for this class:', time_list[-1])
        
        sod_set_median_list[-1] = np.mean(sod_set_median_list[-1])
        sod_gs_list[-1] = np.mean(sod_gs_list[-1])
        dis_ks_set_median_list[-1] = np.mean(dis_ks_set_median_list[-1])
        dis_ks_min_list[-1] = np.mean(dis_ks_min_list[-1])
        time_list[-1] = np.mean(time_list[-1])
        
    print()
    print('\nmean sods of the set median for each class:', sod_set_median_list)
    print('\nmean sods in graph space for each class:', sod_gs_list)
    print('\ndistances in kernel space of set median for each class:', 
            dis_ks_set_median_list)
    print('\nmean smallest distances in kernel space for each class:', 
            dis_ks_min_list)
    print('\nmean times for each class:', time_list)
    
    print('\nmean sods of the set median of all:', np.mean(sod_set_median_list))
    print('\nmean sods in graph space of all:', np.mean(sod_gs_list))
    print('\nmean distances in kernel space of set median of all:', 
            np.mean(dis_ks_set_median_list))
    print('\nmean smallest distances in kernel space of all:', 
            np.mean(dis_ks_min_list))
    print('\nmean times of all:', np.mean(time_list))
    
    nb_better_sods = 0
    nb_worse_sods = 0
    nb_same_sods = 0
    for sods in sod_list_list:
        if sods[0] > sods[-1]:
            nb_better_sods += 1
        elif sods[0] < sods[-1]:
            nb_worse_sods += 1
        else:
            nb_same_sods += 1
    print('\n In', str(len(sod_list_list)), 'sod lists,', str(nb_better_sods), 
          'are getting better,', str(nb_worse_sods), 'are getting worse,', 
          str(nb_same_sods), 'are not changed; ', str(nb_better_sods / len(sod_list_list)),
          'sods are improved.')
    

###############################################################################
# tests on different numbers of median-sets.

def test_iam_median_nb():
    
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    
    lmbda = 0.03 # termination probalility
#    # parameters for GED function
#    c_vi = 0.037
#    c_vr = 0.038
#    c_vs = 0.075
#    c_ei = 0.001
#    c_er = 0.001
#    c_es = 0.0
#    ite_max_iam = 50
#    epsilon_iam = 0.001
#    removeNodes = False
#    connected_iam = False
#    # parameters for IAM function
#    ged_cost = 'CONSTANT'
#    ged_method = 'IPFP'
#    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
#    ged_stabilizer = 'min'
#    ged_repeat = 50
#    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
#                  'edit_cost_constant': edit_cost_constant, 
#                  'stabilizer': ged_stabilizer, 'repeat': ged_repeat}
    
    # parameters for GED function
    c_vi = 4
    c_vr = 4
    c_vs = 2
    c_ei = 1
    c_er = 1
    c_es = 1
    ite_max_iam = 50
    epsilon_iam = 0.001
    removeNodes = False
    connected_iam = False
    # parameters for IAM function
    ged_cost = 'CHEM_1'
    ged_method = 'IPFP'
    edit_cost_constant = []
    ged_stabilizer = 'min'
    ged_repeat = 50
    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
                  'edit_cost_constant': edit_cost_constant, 
                  'stabilizer': ged_stabilizer, 'repeat': ged_repeat}
    
    # find out all the graphs classified to positive group 1.
    idx_dict = get_same_item_indices(y_all)
    Gn = [Gn[i] for i in idx_dict[1]]
    
    # number of graphs; we what to compute the median of these graphs. 
#    nb_median_range = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    nb_median_range = [len(Gn)]
    
#    # compute Gram matrix.
#    time0 = time.time()
#    km = compute_kernel(Gn, gkernel, True)
#    time_km = time.time() - time0    
#    # write Gram matrix to file.
#    np.savez('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm', gm=km, gmtime=time_km)
    
    time_list = []
    dis_ks_min_list = []
    sod_gs_list = []
#    sod_gs_min_list = []
#    nb_updated_list = []
#    nb_updated_k_list = []
    g_best = []
    for nb_median in nb_median_range:
        print('\n-------------------------------------------------------')
        print('number of median graphs =', nb_median)
        random.seed(1)
        idx_rdm = random.sample(range(len(Gn)), nb_median)
        print('graphs chosen:', idx_rdm)
        Gn_median = [Gn[idx].copy() for idx in idx_rdm]
        Gn_candidate = [g.copy() for g in Gn]
        
#        for g in Gn_median:
#            nx.draw(g, labels=nx.get_node_attributes(g, 'atom'), with_labels=True)
##            plt.savefig("results/preimage_mix/mutag.png", format="PNG")
#            plt.show()
#            plt.clf()                         
                    
        ###################################################################
#        gmfile = np.load('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm.npz')
#        km_tmp = gmfile['gm']
#        time_km = gmfile['gmtime']
#        # modify mixed gram matrix.
#        km = np.zeros((len(Gn) + nb_median, len(Gn) + nb_median))
#        for i in range(len(Gn)):
#            for j in range(i, len(Gn)):
#                km[i, j] = km_tmp[i, j]
#                km[j, i] = km[i, j]
#        for i in range(len(Gn)):
#            for j, idx in enumerate(idx_rdm):
#                km[i, len(Gn) + j] = km[i, idx]
#                km[len(Gn) + j, i] = km[i, idx]
#        for i, idx1 in enumerate(idx_rdm):
#            for j, idx2 in enumerate(idx_rdm):
#                km[len(Gn) + i, len(Gn) + j] = km[idx1, idx2]
                
        ###################################################################
        alpha_range = [1 / nb_median] * nb_median
        time0 = time.time()
        ghat_new_list, sod_min = iam_upgraded(Gn_median, Gn_candidate, 
            c_ei=c_ei, c_er=c_er, c_es=c_es, ite_max=ite_max_iam,
            epsilon=epsilon_iam, connected=connected_iam, removeNodes=removeNodes, 
            params_ged=params_ged)
            
        time_total = time.time() - time0
        print('\ntime: ', time_total)
        time_list.append(time_total)
        
        # compute distance between \psi and the new generated graphs.
        knew = compute_kernel(ghat_new_list + Gn_median, gkernel, False)
        dhat_new_list = []
        for idx, g_tmp in enumerate(ghat_new_list):
            # @todo: the term3 below could use the one at the beginning of the function.
            dhat_new_list.append(dis_gstar(idx, range(len(ghat_new_list), 
                                len(ghat_new_list) + len(Gn_median) + 1), 
                                alpha_range, knew, withterm3=False))
            
        print('\nsmallest distance in kernel space: ', dhat_new_list[0]) 
        dis_ks_min_list.append(dhat_new_list[0])
        g_best.append(ghat_new_list[0])
        
        # show the best graph and save it to file.
#        print('the shortest distance is', dhat)
        print('one of the possible corresponding pre-images is')
        nx.draw(ghat_new_list[0], labels=nx.get_node_attributes(ghat_new_list[0], 'atom'), 
                with_labels=True)
        plt.show()
#        plt.savefig('results/iam/mutag_median.fit_costs2.001.nb' + str(nb_median) + 
        plt.savefig('results/iam/mutag_median_unfit2.nb' + str(nb_median) + 
                    '.png', format="PNG")
        plt.clf()
#        print(ghat_list[0].nodes(data=True))
#        print(ghat_list[0].edges(data=True))
    
        sod_gs_list.append(sod_min)
#        sod_gs_min_list.append(np.min(sod_min))
        print('\nsmallest sod in graph space: ', sod_min)
        
    print('\nsods in graph space: ', sod_gs_list)
#    print('\nsmallest sod in graph space for each set of median graphs: ', sod_gs_min_list)  
    print('\nsmallest distance in kernel space for each set of median graphs: ', 
          dis_ks_min_list) 
#    print('\nnumber of updates of the best graph for each set of median graphs by IAM: ', 
#          nb_updated_list)
#    print('\nnumber of updates of k nearest graphs for each set of median graphs by IAM: ', 
#          nb_updated_k_list)
    print('\ntimes:', time_list)
    
    
def test_iam_letter_h():
    from median import draw_Letter_graph
    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
          'extra_params': {}} # node nsymb
#    ds = {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt',
#          'extra_params': {}} # node nsymb
#    Gn = Gn[0:50]
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    gkernel = 'structuralspkernel'
    
    # parameters for GED function from the IAM paper.
    c_vi = 3
    c_vr = 3
    c_vs = 1
    c_ei = 3
    c_er = 3
    c_es = 1
    ite_max_iam = 50
    epsilon_iam = 0.001
    removeNodes = False
    connected_iam = False
    # parameters for IAM function
#    ged_cost = 'CONSTANT'
    ged_cost = 'LETTER'
    ged_method = 'IPFP'
#    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
    edit_cost_constant = []
    ged_stabilizer = 'min'
    ged_repeat = 50
    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
                  'edit_cost_constant': edit_cost_constant, 
                  'stabilizer': ged_stabilizer, 'repeat': ged_repeat}
    
    # classify graphs according to letters.
    time_list = []
    dis_ks_min_list = []
    sod_gs_list = []
    g_best = []
    sod_set_median_list = []
    idx_dict = get_same_item_indices(y_all)
    for letter in idx_dict:
        print('\n-------------------------------------------------------')
        print('letter', letter)
        Gn_let = [Gn[i].copy() for i in idx_dict[letter]]
        
        time_list.append([])
        dis_ks_min_list.append([])
        sod_gs_list.append([])
        g_best.append([])
        sod_set_median_list.append([])
        
        for repeat in range(50):
            idx_rdm = random.sample(range(len(Gn_let)), 50)
            print('graphs chosen:', idx_rdm)
            Gn_median = [Gn_let[idx].copy() for idx in idx_rdm]
            Gn_candidate = [g.copy() for g in Gn_median]
        
            alpha_range = [1 / len(Gn_median)] * len(Gn_median)
            time0 = time.time()
            ghat_new_list, sod_min, sod_set_median = iam_upgraded(Gn_median, 
                Gn_candidate, c_ei=c_ei, c_er=c_er, c_es=c_es, ite_max=ite_max_iam,
                epsilon=epsilon_iam, connected=connected_iam, removeNodes=removeNodes, 
                params_ged=params_ged)
            time_total = time.time() - time0
            print('\ntime: ', time_total)
            time_list[-1].append(time_total)
            g_best[-1].append(ghat_new_list[0])
            sod_set_median_list[-1].append(sod_set_median)
            print('\nsmallest sod of the set median:', sod_set_median)
            sod_gs_list[-1].append(sod_min)
            print('\nsmallest sod in graph space:', sod_min)
            
            # show the best graph and save it to file.
            print('one of the possible corresponding pre-images is')
            draw_Letter_graph(ghat_new_list[0], savepath='results/iam/paper_compare/')
            
            # compute distance between \psi and the new generated graphs.
            knew = compute_kernel(ghat_new_list + Gn_median, gkernel, False)
            dhat_new_list = []
            for idx, g_tmp in enumerate(ghat_new_list):
                # @todo: the term3 below could use the one at the beginning of the function.
                dhat_new_list.append(dis_gstar(idx, range(len(ghat_new_list), 
                                    len(ghat_new_list) + len(Gn_median) + 1), 
                                    alpha_range, knew, withterm3=False))
                
            print('\nsmallest distance in kernel space: ', dhat_new_list[0]) 
            dis_ks_min_list[-1].append(dhat_new_list[0])            
        
        print('\nsods of the set median for this letter:', sod_set_median_list[-1])
        print('\nsods in graph space for this letter:', sod_gs_list[-1])
        print('\nsmallest distances in kernel space for this letter:', 
              dis_ks_min_list[-1])
        print('\ntimes for this letter:', time_list[-1])
        
        sod_set_median_list[-1] = np.mean(sod_set_median_list[-1])
        sod_gs_list[-1] = np.mean(sod_gs_list[-1])
        dis_ks_min_list[-1] = np.mean(dis_ks_min_list[-1])
        time_list[-1] = np.mean(time_list[-1])
        
    print('\nmean sods of the set median for each letter:', sod_set_median_list)
    print('\nmean sods in graph space for each letter:', sod_gs_list)
    print('\nmean smallest distances in kernel space for each letter:', 
            dis_ks_min_list)
    print('\nmean times for each letter:', time_list)
    
    print('\nmean sods of the set median of all:', np.mean(sod_set_median_list))
    print('\nmean sods in graph space of all:', np.mean(sod_gs_list))
    print('\nmean smallest distances in kernel space of all:', 
            np.mean(dis_ks_min_list))
    print('\nmean times of all:', np.mean(time_list))
    
    

    


    
    

def test_iam_fitdistance():
    
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:50]
#    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
    
#    lmbda = 0.03 # termination probalility
#    # parameters for GED function
#    c_vi = 0.037
#    c_vr = 0.038
#    c_vs = 0.075
#    c_ei = 0.001
#    c_er = 0.001
#    c_es = 0.0
#    ite_max_iam = 50
#    epsilon_iam = 0.001
#    removeNodes = False
#    connected_iam = False
#    # parameters for IAM function
#    ged_cost = 'CONSTANT'
#    ged_method = 'IPFP'
#    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
#    ged_stabilizer = 'min'
#    ged_repeat = 50
#    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
#                  'edit_cost_constant': edit_cost_constant, 
#                  'stabilizer': ged_stabilizer, 'repeat': ged_repeat}
    
    # parameters for GED function
    c_vi = 4
    c_vr = 4
    c_vs = 2
    c_ei = 1
    c_er = 1
    c_es = 1
    ite_max_iam = 50
    epsilon_iam = 0.001
    removeNodes = False
    connected_iam = False
    # parameters for IAM function
    ged_cost = 'CHEM_1'
    ged_method = 'IPFP'
    edit_cost_constant = []
    ged_stabilizer = 'min'
    ged_repeat = 50
    params_ged = {'lib': 'gedlibpy', 'cost': ged_cost, 'method': ged_method, 
                  'edit_cost_constant': edit_cost_constant, 
                  'stabilizer': ged_stabilizer, 'repeat': ged_repeat}
    
    # find out all the graphs classified to positive group 1.
    idx_dict = get_same_item_indices(y_all)
    Gn = [Gn[i] for i in idx_dict[1]]
    
    # number of graphs; we what to compute the median of these graphs. 
#    nb_median_range = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    nb_median_range = [10]
    
#    # compute Gram matrix.
#    time0 = time.time()
#    km = compute_kernel(Gn, gkernel, True)
#    time_km = time.time() - time0
#    # write Gram matrix to file.
#    np.savez('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm', gm=km, gmtime=time_km)
    
    time_list = []
    dis_ks_min_list = []
    dis_ks_gen_median_list = []
    sod_gs_list = []
#    sod_gs_min_list = []
#    nb_updated_list = []
#    nb_updated_k_list = []
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
#        gmfile = np.load('results/gram_matrix_marg_itr10_pq0.03_mutag_positive.gm.npz')
#        km_tmp = gmfile['gm']
#        time_km = gmfile['gmtime']
#        # modify mixed gram matrix.
#        km = np.zeros((len(Gn) + nb_median, len(Gn) + nb_median))
#        for i in range(len(Gn)):
#            for j in range(i, len(Gn)):
#                km[i, j] = km_tmp[i, j]
#                km[j, i] = km[i, j]
#        for i in range(len(Gn)):
#            for j, idx in enumerate(idx_rdm):
#                km[i, len(Gn) + j] = km[i, idx]
#                km[len(Gn) + j, i] = km[i, idx]
#        for i, idx1 in enumerate(idx_rdm):
#            for j, idx2 in enumerate(idx_rdm):
#                km[len(Gn) + i, len(Gn) + j] = km[idx1, idx2]
                
        ###################################################################
        alpha_range = [1 / nb_median] * nb_median
        time0 = time.time()
        G_gen_median_list, sod_gen_median, sod_list, G_set_median_list, sod_set_median \
            = iam_upgraded(Gn_median, Gn_candidate, 
            c_ei=c_ei, c_er=c_er, c_es=c_es, ite_max=ite_max_iam,
            epsilon=epsilon_iam, connected=connected_iam, removeNodes=removeNodes, 
            params_ged=params_ged)
            
        time_total = time.time() - time0
        print('\ntime: ', time_total)
        time_list.append(time_total)
        
        # compute distance between \psi and the new generated graphs.
        knew = compute_kernel(G_gen_median_list + Gn_median, gkernel, node_label,
                              edge_label, False)
        dhat_new_list = []
        for idx, g_tmp in enumerate(G_gen_median_list):
            # @todo: the term3 below could use the one at the beginning of the function.
            dhat_new_list.append(dis_gstar(idx, range(len(G_gen_median_list), 
                                len(G_gen_median_list) + len(Gn_median) + 1), 
                                alpha_range, knew, withterm3=False))
            
        print('\nsmallest distance in kernel space: ', dhat_new_list[0]) 
        dis_ks_min_list.append(dhat_new_list[0])
        g_best.append(G_gen_median_list[0])
        
        # show the best graph and save it to file.
#        print('the shortest distance is', dhat)
        print('one of the possible corresponding pre-images is')
        nx.draw(G_gen_median_list[0], labels=nx.get_node_attributes(G_gen_median_list[0], 'atom'), 
                with_labels=True)
        plt.show()
#        plt.savefig('results/iam/mutag_median.fit_costs2.001.nb' + str(nb_median) + 
#        plt.savefig('results/iam/mutag_median_unfit2.nb' + str(nb_median) + 
#                    '.png', format="PNG")
        plt.clf()
#        print(ghat_list[0].nodes(data=True))
#        print(ghat_list[0].edges(data=True))
    
        sod_gs_list.append(sod_gen_median)
#        sod_gs_min_list.append(np.min(sod_gen_median))
        print('\nsmallest sod in graph space: ', sod_gen_median)
        print('\nsmallest sod of set median in graph space: ', sod_set_median)
        
    print('\nsods in graph space: ', sod_gs_list)
#    print('\nsmallest sod in graph space for each set of median graphs: ', sod_gs_min_list)  
    print('\nsmallest distance in kernel space for each set of median graphs: ', 
          dis_ks_min_list) 
#    print('\nnumber of updates of the best graph for each set of median graphs by IAM: ', 
#          nb_updated_list)
#    print('\nnumber of updates of k nearest graphs for each set of median graphs by IAM: ', 
#          nb_updated_k_list)
    print('\ntimes:', time_list)
        
    
            
    
    
###############################################################################

    
if __name__ == '__main__':
###############################################################################
# tests on different numbers of median-sets.
#    test_iam_median_nb()
#    test_iam_letter_h()
#    test_iam_monoterpenoides()
#    test_iam_mutag()
    
#    test_iam_fitdistance()
#    print("test log")
    
    test_iam_monoterpenoides_with_init40()
