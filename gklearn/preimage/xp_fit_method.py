#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:39:29 2020

@author: ljia
"""
import numpy as np
import random
import csv
from shutil import copyfile
import networkx as nx
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../")
from gklearn.utils.graphfiles import loadDataset, loadGXL, saveGXL
from preimage.test_k_closest_graphs import median_on_k_closest_graphs, reform_attributes
from preimage.utils import get_same_item_indices, kernel_distance_matrix, compute_kernel
from preimage.find_best_k import getRelations


def get_dataset(ds_name):
    if ds_name == 'Letter-high': # node non-symb
        dataset = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/collections/Letter.xml'
        graph_dir = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/datasets/Letter/HIGH/' 
        Gn, y_all = loadDataset(dataset, extra_params=graph_dir)
        for G in Gn:
            reform_attributes(G)
    elif ds_name == 'Fingerprint':
        dataset = '/media/ljia/DATA/research-repo/codes/Linlin/gedlib/data/collections/Fingerprint.xml'
        graph_dir = '/media/ljia/DATA/research-repo/codes/Linlin/gedlib/data/datasets/Fingerprint/data/'
        Gn, y_all = loadDataset(dataset, extra_params=graph_dir)
        for G in Gn:
            reform_attributes(G)
    elif ds_name == 'SYNTHETIC':
        pass
    elif ds_name == 'SYNTHETICnew':
        dataset = '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/datasets/SYNTHETICnew/SYNTHETICnew_A.txt'
        graph_dir = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/generated_datsets/SYNTHETICnew'
#        dataset = '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/datasets/Letter-high/Letter-high_A.txt'
#        graph_dir = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/datasets/Letter/HIGH/'
        Gn, y_all = loadDataset(dataset)
    elif ds_name == 'Synthie':
        pass
    elif ds_name == 'COIL-RAG':
        pass
    elif ds_name == 'COLORS-3':
        pass
    elif ds_name == 'FRANKENSTEIN':
        pass
    
    return Gn, y_all, graph_dir


def init_output_file(ds_name, gkernel, fit_method, dir_output):
#    fn_output_detail = 'results_detail.' + ds_name + '.' + gkernel + '.' + fit_method + '.csv'
    fn_output_detail = 'results_detail.' + ds_name + '.' + gkernel + '.csv'
    f_detail = open(dir_output + fn_output_detail, 'a')
    csv.writer(f_detail).writerow(['dataset', 'graph kernel', 'edit cost', 
              'GED method', 'attr distance', 'fit method', 'k', 
              'target', 'repeat', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
              'min dis_k gi', 'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
              'dis_k gi -> GM', 'fitting time', 'generating time', 'total time',
              'median set'])
    f_detail.close()
    
#    fn_output_summary = 'results_summary.' + ds_name + '.' + gkernel + '.' + fit_method + '.csv'
    fn_output_summary = 'results_summary.' + ds_name + '.' + gkernel + '.csv'
    f_summary = open(dir_output + fn_output_summary, 'a')
    csv.writer(f_summary).writerow(['dataset', 'graph kernel', 'edit cost', 
              'GED method', 'attr distance', 'fit method', 'k', 
              'target', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
              'min dis_k gi', 'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
              'dis_k gi -> GM', 'fitting time', 'generating time', 'total time',
              '# SOD SM -> GM', '# dis_k SM -> GM', 
              '# dis_k gi -> SM', '# dis_k gi -> GM', 'repeats better SOD SM -> GM', 
              'repeats better dis_k SM -> GM', 'repeats better dis_k gi -> SM', 
              'repeats better dis_k gi -> GM'])
    f_summary.close()
    
    return fn_output_detail, fn_output_summary


def xp_fit_method_for_non_symbolic(parameters, save_results=True, initial_solutions=1,
                                   Gn_data=None, k_dis_data=None, Kmatrix=None):
    
    # 1. set parameters.
    print('1. setting parameters...')
    ds_name = parameters['ds_name']
    gkernel = parameters['gkernel']
    edit_cost_name = parameters['edit_cost_name']
    ged_method = parameters['ged_method']
    attr_distance = parameters['attr_distance']
    fit_method = parameters['fit_method']

    node_label = None
    edge_label = None
    dir_output = 'results/xp_fit_method/'    
      
    
    # 2. get dataset.
    print('2. getting dataset...')
    if Gn_data is None:
        Gn, y_all, graph_dir = get_dataset(ds_name)
    else:
        Gn = Gn_data[0]
        y_all = Gn_data[1]
        graph_dir = Gn_data[2]
        
    
    # 3. compute kernel distance matrix.
    print('3. computing kernel distance matrix...')
    if k_dis_data is None:
        dis_mat, dis_max, dis_min, dis_mean = kernel_distance_matrix(Gn, None, 
            None, Kmatrix=Kmatrix, gkernel=gkernel)
    else:
        dis_mat = k_dis_data[0]
        dis_max = k_dis_data[1]
        dis_min = k_dis_data[2]
        dis_mean = k_dis_data[3]
        print('pair distances - dis_max, dis_min, dis_mean:', dis_max, dis_min, dis_mean)


    if save_results:
        # create result files.
        print('creating output files...')
        fn_output_detail, fn_output_summary = init_output_file(ds_name, gkernel, 
                                                               fit_method, dir_output)

            
    # start repeats.    
    repeats = 1
#    k_list = range(2, 11)
    k_list = [0]
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    random.seed(1)
    rdn_seed_list = random.sample(range(0, repeats * 100), repeats)
    
    for k in k_list:
#        print('\n--------- k =', k, '----------')
        
        sod_sm_mean_list = []
        sod_gm_mean_list = []
        dis_k_sm_mean_list = []
        dis_k_gm_mean_list = []
        dis_k_gi_min_mean_list = []
        time_fitting_mean_list = []
        time_generating_mean_list = []
        time_total_mean_list = []
        
        # 3. start generating and computing over targets.
        print('4. starting generating and computing over targets......')
        for i, (y, values) in enumerate(y_idx.items()):
#            y = 'I'
#            values = y_idx[y]
#            values = values[0:10]            
            print('\ny =', y)
#            if y.strip() == 'A':
#                continue
            
            k = len(values)
            print('\n--------- k =', k, '----------')
            
            sod_sm_list = []
            sod_gm_list = []
            dis_k_sm_list = []
            dis_k_gm_list = []
            dis_k_gi_min_list = []
            time_fitting_list = []
            time_generating_list = []
            time_total_list = []
            nb_sod_sm2gm = [0, 0, 0]
            nb_dis_k_sm2gm = [0, 0, 0]
            nb_dis_k_gi2sm = [0, 0, 0]
            nb_dis_k_gi2gm = [0, 0, 0]
            repeats_better_sod_sm2gm = []
            repeats_better_dis_k_sm2gm = []
            repeats_better_dis_k_gi2sm = []
            repeats_better_dis_k_gi2gm = []
            
            # get Gram matrix for this part of data.
            if Kmatrix is not None:
                Kmatrix_sub = Kmatrix[values,:]
                Kmatrix_sub = Kmatrix_sub[:,values]
            
            for repeat in range(repeats):
                print('\nrepeat =', repeat)
                random.seed(rdn_seed_list[repeat])
                median_set_idx_idx = random.sample(range(0, len(values)), k)
                median_set_idx = [values[idx] for idx in median_set_idx_idx]
                print('median set: ', median_set_idx)
                Gn_median = [Gn[g] for g in values]
#                from notebooks.utils.plot_all_graphs import draw_Fingerprint_graph
#                for Gn in Gn_median:
#                    draw_Fingerprint_graph(Gn, save=None)
                
                # GENERATING & COMPUTING!!
                res_sods, res_dis_ks, res_times = median_on_k_closest_graphs(Gn_median, 
                        node_label, edge_label, 
                        gkernel, k, fit_method=fit_method, graph_dir=graph_dir,
                        edit_cost_constants=None, group_min=median_set_idx_idx, 
                        dataset=ds_name, initial_solutions=initial_solutions,
                        edit_cost_name=edit_cost_name, 
                        Kmatrix=Kmatrix_sub, parallel=False)
                sod_sm = res_sods[0]
                sod_gm = res_sods[1] 
                dis_k_sm = res_dis_ks[0]
                dis_k_gm = res_dis_ks[1]
                dis_k_gi = res_dis_ks[2]
                dis_k_gi_min = res_dis_ks[3]
                idx_dis_k_gi_min = res_dis_ks[4]
                time_fitting = res_times[0]
                time_generating = res_times[1]                    
                
                # write result detail.
                sod_sm2gm = getRelations(np.sign(sod_gm - sod_sm))
                dis_k_sm2gm = getRelations(np.sign(dis_k_gm - dis_k_sm))
                dis_k_gi2sm = getRelations(np.sign(dis_k_sm - dis_k_gi_min))
                dis_k_gi2gm = getRelations(np.sign(dis_k_gm - dis_k_gi_min))
                if save_results:
                    f_detail = open(dir_output + fn_output_detail, 'a')
                    csv.writer(f_detail).writerow([ds_name, gkernel, 
                              edit_cost_name, ged_method, attr_distance,
                              fit_method, k, y, repeat,
                              sod_sm, sod_gm, dis_k_sm, dis_k_gm, 
                              dis_k_gi_min, sod_sm2gm, dis_k_sm2gm, dis_k_gi2sm,
                              dis_k_gi2gm, time_fitting, time_generating,
                              time_fitting + time_generating, median_set_idx])
                    f_detail.close()
                
                # compute result summary.
                sod_sm_list.append(sod_sm)
                sod_gm_list.append(sod_gm)
                dis_k_sm_list.append(dis_k_sm)
                dis_k_gm_list.append(dis_k_gm)
                dis_k_gi_min_list.append(dis_k_gi_min)
                time_fitting_list.append(time_fitting)
                time_generating_list.append(time_generating)
                time_total_list.append(time_fitting + time_generating)
                # # SOD SM -> GM
                if sod_sm > sod_gm:
                    nb_sod_sm2gm[0] += 1
                    repeats_better_sod_sm2gm.append(repeat)
                elif sod_sm == sod_gm:
                    nb_sod_sm2gm[1] += 1
                elif sod_sm < sod_gm:
                    nb_sod_sm2gm[2] += 1
                # # dis_k SM -> GM
                if dis_k_sm > dis_k_gm:
                    nb_dis_k_sm2gm[0] += 1
                    repeats_better_dis_k_sm2gm.append(repeat)
                elif dis_k_sm == dis_k_gm:
                    nb_dis_k_sm2gm[1] += 1
                elif dis_k_sm < dis_k_gm:
                    nb_dis_k_sm2gm[2] += 1
                # # dis_k gi -> SM
                if dis_k_gi_min > dis_k_sm:
                    nb_dis_k_gi2sm[0] += 1
                    repeats_better_dis_k_gi2sm.append(repeat)
                elif dis_k_gi_min == dis_k_sm:
                    nb_dis_k_gi2sm[1] += 1
                elif dis_k_gi_min < dis_k_sm:
                    nb_dis_k_gi2sm[2] += 1
                # # dis_k gi -> GM
                if dis_k_gi_min > dis_k_gm:
                    nb_dis_k_gi2gm[0] += 1
                    repeats_better_dis_k_gi2gm.append(repeat)
                elif dis_k_gi_min == dis_k_gm:
                    nb_dis_k_gi2gm[1] += 1
                elif dis_k_gi_min < dis_k_gm:
                    nb_dis_k_gi2gm[2] += 1
                    
                # save median graphs.
                fname_sm = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/output/tmp_ged/set_median.gxl'
                fn_pre_sm_new = dir_output + 'medians/set_median.' + fit_method \
                    + '.k' + str(int(k)) + '.y' + str(y) + '.repeat' + str(repeat)
                copyfile(fname_sm, fn_pre_sm_new + '.gxl')
                fname_gm = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/output/tmp_ged/gen_median.gxl'
                fn_pre_gm_new = dir_output + 'medians/gen_median.' + fit_method \
                    + '.k' + str(int(k)) + '.y' + str(y) + '.repeat' + str(repeat)
                copyfile(fname_gm, fn_pre_gm_new + '.gxl')
                G_best_kernel = Gn_median[idx_dis_k_gi_min].copy()
#                reform_attributes(G_best_kernel)
                fn_pre_g_best_kernel = dir_output + 'medians/g_best_kernel.' + fit_method \
                    + '.k' + str(int(k)) + '.y' + str(y) + '.repeat' + str(repeat)
                saveGXL(G_best_kernel, fn_pre_g_best_kernel + '.gxl', method='default')
                
                # plot median graphs.
                if ds_name == 'Letter-high':
                    set_median = loadGXL(fn_pre_sm_new + '.gxl')
                    gen_median = loadGXL(fn_pre_gm_new + '.gxl')                
                    draw_Letter_graph(set_median, fn_pre_sm_new)
                    draw_Letter_graph(gen_median, fn_pre_gm_new)
                    draw_Letter_graph(G_best_kernel, fn_pre_g_best_kernel)
                    
            # write result summary for each letter. 
            sod_sm_mean_list.append(np.mean(sod_sm_list))
            sod_gm_mean_list.append(np.mean(sod_gm_list))
            dis_k_sm_mean_list.append(np.mean(dis_k_sm_list))
            dis_k_gm_mean_list.append(np.mean(dis_k_gm_list))
            dis_k_gi_min_mean_list.append(np.mean(dis_k_gi_min_list))
            time_fitting_mean_list.append(np.mean(time_fitting_list))
            time_generating_mean_list.append(np.mean(time_generating_list))
            time_total_mean_list.append(np.mean(time_total_list))
            sod_sm2gm_mean = getRelations(np.sign(sod_gm_mean_list[-1] - sod_sm_mean_list[-1]))
            dis_k_sm2gm_mean = getRelations(np.sign(dis_k_gm_mean_list[-1] - dis_k_sm_mean_list[-1]))
            dis_k_gi2sm_mean = getRelations(np.sign(dis_k_sm_mean_list[-1] - dis_k_gi_min_mean_list[-1]))
            dis_k_gi2gm_mean = getRelations(np.sign(dis_k_gm_mean_list[-1] - dis_k_gi_min_mean_list[-1]))
            if save_results:
                f_summary = open(dir_output + fn_output_summary, 'a')
                csv.writer(f_summary).writerow([ds_name, gkernel, 
                          edit_cost_name, ged_method, attr_distance,
                          fit_method, k, y,
                          sod_sm_mean_list[-1], sod_gm_mean_list[-1], 
                          dis_k_sm_mean_list[-1], dis_k_gm_mean_list[-1],
                          dis_k_gi_min_mean_list[-1], sod_sm2gm_mean, dis_k_sm2gm_mean, 
                          dis_k_gi2sm_mean, dis_k_gi2gm_mean, 
                          time_fitting_mean_list[-1], time_generating_mean_list[-1],
                          time_total_mean_list[-1], nb_sod_sm2gm, 
                          nb_dis_k_sm2gm, nb_dis_k_gi2sm, nb_dis_k_gi2gm, 
                          repeats_better_sod_sm2gm, repeats_better_dis_k_sm2gm, 
                          repeats_better_dis_k_gi2sm, repeats_better_dis_k_gi2gm])
                f_summary.close()
            

        # write result summary for each letter. 
        sod_sm_mean = np.mean(sod_sm_mean_list)
        sod_gm_mean = np.mean(sod_gm_mean_list)
        dis_k_sm_mean = np.mean(dis_k_sm_mean_list)
        dis_k_gm_mean = np.mean(dis_k_gm_mean_list)
        dis_k_gi_min_mean = np.mean(dis_k_gi_min_list)
        time_fitting_mean = np.mean(time_fitting_list)
        time_generating_mean = np.mean(time_generating_list)
        time_total_mean = np.mean(time_total_list)
        sod_sm2gm_mean = getRelations(np.sign(sod_gm_mean - sod_sm_mean))
        dis_k_sm2gm_mean = getRelations(np.sign(dis_k_gm_mean - dis_k_sm_mean))
        dis_k_gi2sm_mean = getRelations(np.sign(dis_k_sm_mean - dis_k_gi_min_mean))
        dis_k_gi2gm_mean = getRelations(np.sign(dis_k_gm_mean - dis_k_gi_min_mean))
        if save_results:
            f_summary = open(dir_output + fn_output_summary, 'a')
            csv.writer(f_summary).writerow([ds_name, gkernel, 
                      edit_cost_name, ged_method, attr_distance,
                      fit_method, k, 'all',
                      sod_sm_mean, sod_gm_mean, dis_k_sm_mean, dis_k_gm_mean,
                      dis_k_gi_min_mean, sod_sm2gm_mean, dis_k_sm2gm_mean, 
                      dis_k_gi2sm_mean, dis_k_gi2gm_mean,
                      time_fitting_mean, time_generating_mean, time_total_mean])
            f_summary.close()
        
    print('\ncomplete.')
    
    
#Dessin median courrant
def draw_Letter_graph(graph, file_prefix):
    plt.figure()
    pos = {}
    for n in graph.nodes:
        pos[n] = np.array([float(graph.node[n]['x']),float(graph.node[n]['y'])])
    nx.draw_networkx(graph, pos)
    plt.savefig(file_prefix + '.eps', format='eps', dpi=300)
#    plt.show()
    plt.clf()
        

if __name__ == "__main__":
#    #### xp 1: Letter-high, spkernel.
#    # load dataset.
#    print('getting dataset and computing kernel distance matrix first...')
#    ds_name = 'Letter-high'
#    gkernel = 'spkernel'
#    Gn, y_all, graph_dir = get_dataset(ds_name)
#    # remove graphs without edges.
#    Gn = [(idx, G) for idx, G in enumerate(Gn) if nx.number_of_edges(G) != 0]
#    idx = [G[0] for G in Gn]
#    Gn = [G[1] for G in Gn]
#    y_all = [y_all[i] for i in idx]
##    Gn = Gn[0:50]
##    y_all = y_all[0:50]
#    # compute pair distances.
#    dis_mat, dis_max, dis_min, dis_mean = kernel_distance_matrix(Gn, None, None, 
#        Kmatrix=None, gkernel=gkernel, verbose=True)
##    dis_mat, dis_max, dis_min, dis_mean = 0, 0, 0, 0
#    # fitting and computing.
#    fit_methods = ['random', 'expert', 'k-graphs']
#    for fit_method in fit_methods:
#        print('\n-------------------------------------')
#        print('fit method:', fit_method)
#        parameters = {'ds_name': ds_name,
#                      'gkernel': gkernel,
#                      'edit_cost_name': 'LETTER2',
#                      'ged_method': 'mIPFP',
#                      'attr_distance': 'euclidean',
#                      'fit_method': fit_method}
#        xp_fit_method_for_non_symbolic(parameters, save_results=True, 
#                                       initial_solutions=40,
#                                       Gn_data = [Gn, y_all, graph_dir],
#                                       k_dis_data = [dis_mat, dis_max, dis_min, dis_mean])
        
        
#    #### xp 2: Letter-high, sspkernel.
#    # load dataset.
#    print('getting dataset and computing kernel distance matrix first...')
#    ds_name = 'Letter-high'
#    gkernel = 'structuralspkernel'
#    Gn, y_all, graph_dir = get_dataset(ds_name)
##    Gn = Gn[0:50]
##    y_all = y_all[0:50]
#    # compute pair distances.
#    dis_mat, dis_max, dis_min, dis_mean = kernel_distance_matrix(Gn, None, None, 
#        Kmatrix=None, gkernel=gkernel, verbose=True)
##    dis_mat, dis_max, dis_min, dis_mean = 0, 0, 0, 0
#    # fitting and computing.
#    fit_methods = ['random', 'expert', 'k-graphs']
#    for fit_method in fit_methods:
#        print('\n-------------------------------------')
#        print('fit method:', fit_method)
#        parameters = {'ds_name': ds_name,
#                      'gkernel': gkernel,
#                      'edit_cost_name': 'LETTER2',
#                      'ged_method': 'mIPFP',
#                      'attr_distance': 'euclidean',
#                      'fit_method': fit_method}
#        print('parameters: ', parameters)
#        xp_fit_method_for_non_symbolic(parameters, save_results=True, 
#                                       initial_solutions=40,
#                                       Gn_data = [Gn, y_all, graph_dir],
#                                       k_dis_data = [dis_mat, dis_max, dis_min, dis_mean])
    
    
#    #### xp 3: Fingerprint, sspkernel, using LETTER2.
#    # load dataset.
#    print('getting dataset and computing kernel distance matrix first...')
#    ds_name = 'Fingerprint'
#    gkernel = 'structuralspkernel'
#    Gn, y_all, graph_dir = get_dataset(ds_name)
#    # remove graphs without nodes and edges.
#    Gn = [(idx, G) for idx, G in enumerate(Gn) if (nx.number_of_edges(G) != 0
#          and nx.number_of_edges(G) != 0)]
#    idx = [G[0] for G in Gn]
#    Gn = [G[1] for G in Gn]
#    y_all = [y_all[i] for i in idx]
##    Gn = Gn[0:50]
##    y_all = y_all[0:50]
#    # compute pair distances.
##    dis_mat, dis_max, dis_min, dis_mean = kernel_distance_matrix(Gn, None, None, 
##        Kmatrix=None, gkernel=gkernel, verbose=True)
#    dis_mat, dis_max, dis_min, dis_mean = 0, 0, 0, 0
#    # fitting and computing.
#    fit_methods = ['k-graphs', 'expert', 'random', 'random', 'random']
#    for fit_method in fit_methods:
#        print('\n-------------------------------------')
#        print('fit method:', fit_method)
#        parameters = {'ds_name': ds_name,
#                      'gkernel': gkernel,
#                      'edit_cost_name': 'LETTER2',
#                      'ged_method': 'mIPFP',
#                      'attr_distance': 'euclidean',
#                      'fit_method': fit_method}
#        xp_fit_method_for_non_symbolic(parameters, save_results=True, 
#                                       initial_solutions=40,
#                                       Gn_data = [Gn, y_all, graph_dir],
#                                       k_dis_data = [dis_mat, dis_max, dis_min, dis_mean])
        
        
#    #### xp 4: SYNTHETICnew, sspkernel, using NON_SYMBOLIC.
#    # load dataset.
#    print('getting dataset and computing kernel distance matrix first...')
#    ds_name = 'SYNTHETICnew'
#    gkernel = 'structuralspkernel'
#    Gn, y_all, graph_dir = get_dataset(ds_name)
#    # remove graphs without nodes and edges.
#    Gn = [(idx, G) for idx, G in enumerate(Gn) if (nx.number_of_edges(G) != 0
#          and nx.number_of_edges(G) != 0)]
#    idx = [G[0] for G in Gn]
#    Gn = [G[1] for G in Gn]
#    y_all = [y_all[i] for i in idx]
#    Gn = Gn[0:10]
#    y_all = y_all[0:10]
#    for G in Gn:
#        G.graph['filename'] = 'graph' + str(G.graph['name']) + '.gxl'
#    # compute pair distances.
#    dis_mat, dis_max, dis_min, dis_mean = kernel_distance_matrix(Gn, None, None, 
#        Kmatrix=None, gkernel=gkernel, verbose=True)
##    dis_mat, dis_max, dis_min, dis_mean = 0, 0, 0, 0
#    # fitting and computing.
#    fit_methods = ['k-graphs', 'random', 'random', 'random']
#    for fit_method in fit_methods:
#        print('\n-------------------------------------')
#        print('fit method:', fit_method)
#        parameters = {'ds_name': ds_name,
#                      'gkernel': gkernel,
#                      'edit_cost_name': 'NON_SYMBOLIC',
#                      'ged_method': 'mIPFP',
#                      'attr_distance': 'euclidean',
#                      'fit_method': fit_method}
#        xp_fit_method_for_non_symbolic(parameters, save_results=True, 
#                                       initial_solutions=40,
#                                       Gn_data = [Gn, y_all, graph_dir],
#                                       k_dis_data = [dis_mat, dis_max, dis_min, dis_mean])
        
        
    ### xp 5: SYNTHETICnew, spkernel, using NON_SYMBOLIC.
    gmfile = np.load('results/xp_fit_method/Kmatrix.SYNTHETICnew.spkernel.gm.npz')
    Kmatrix = gmfile['Kmatrix']
    # normalization
    Kmatrix_diag = Kmatrix.diagonal().copy()
    for i in range(len(Kmatrix)):
        for j in range(i, len(Kmatrix)):
            Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
            Kmatrix[j][i] = Kmatrix[i][j]
    run_time = 21821.35
    np.savez('results/xp_fit_method/Kmatrix.SYNTHETICnew.spkernel.gm',
             Kmatrix=Kmatrix, run_time=run_time)
    
    # load dataset.
    print('getting dataset and computing kernel distance matrix first...')
    ds_name = 'SYNTHETICnew'
    gkernel = 'spkernel'
    Gn, y_all, graph_dir = get_dataset(ds_name)
#    # remove graphs without nodes and edges.
#    Gn = [(idx, G) for idx, G in enumerate(Gn) if (nx.number_of_edges(G) != 0
#          and nx.number_of_edges(G) != 0)]
#    idx = [G[0] for G in Gn]
#    Gn = [G[1] for G in Gn]
#    y_all = [y_all[i] for i in idx]
#    Gn = Gn[0:5]
#    y_all = y_all[0:5]
    for G in Gn:
        G.graph['filename'] = 'graph' + str(G.graph['name']) + '.gxl'
    
    # compute/read Gram matrix and pair distances.
#    Kmatrix = compute_kernel(Gn, gkernel, None, None, True)
#    np.savez('results/xp_fit_method/Kmatrix.' + ds_name + '.' + gkernel + '.gm', 
#         Kmatrix=Kmatrix)
    gmfile = np.load('results/xp_fit_method/Kmatrix.' + ds_name + '.' + gkernel + '.gm.npz')
    Kmatrix = gmfile['Kmatrix']
    run_time = gmfile['run_time']
#    Kmatrix = Kmatrix[[0,1,2,3,4],:]
#    Kmatrix = Kmatrix[:,[0,1,2,3,4]]
    print('\nTime to compute Gram matrix for the whole dataset: ', run_time)
    dis_mat, dis_max, dis_min, dis_mean = kernel_distance_matrix(Gn, None, None, 
        Kmatrix=Kmatrix, gkernel=gkernel, verbose=True)
#    Kmatrix = np.zeros((len(Gn), len(Gn)))
#    dis_mat, dis_max, dis_min, dis_mean = 0, 0, 0, 0
    
    # fitting and computing.
    fit_methods = ['k-graphs', 'random', 'random', 'random']
    for fit_method in fit_methods:
        print('\n-------------------------------------')
        print('fit method:', fit_method)
        parameters = {'ds_name': ds_name,
                      'gkernel': gkernel,
                      'edit_cost_name': 'NON_SYMBOLIC',
                      'ged_method': 'mIPFP',
                      'attr_distance': 'euclidean',
                      'fit_method': fit_method}
        xp_fit_method_for_non_symbolic(parameters, save_results=True, 
                                       initial_solutions=1,
                                       Gn_data=[Gn, y_all, graph_dir],
                                       k_dis_data=[dis_mat, dis_max, dis_min, dis_mean],
                                       Kmatrix=Kmatrix)