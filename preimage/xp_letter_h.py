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
from pygraph.utils.graphfiles import loadDataset, loadGXL, saveGXL
from preimage.test_k_closest_graphs import median_on_k_closest_graphs, reform_attributes
from preimage.utils import get_same_item_indices
from preimage.find_best_k import getRelations

def xp_letter_h():
    ds = {'name': 'Letter-high', 
          'dataset': '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/collections/Letter.xml',
          'graph_dir': '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/datasets/Letter/HIGH/'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['graph_dir'])
#    ds = {'name': 'Letter-high', 
#          'dataset': '../datasets/Letter-high/Letter-high_A.txt'}  # node/edge symb
#    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
    gkernel = 'structuralspkernel'
    node_label = None
    edge_label = None
    ds_name = 'letter-h'
    dir_output = 'results/xp_letter_h/'
    
    repeats = 1
#    k_list = range(2, 11)
    k_list = [150]
    fit_method = 'precomputed'
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    
    # create result files.
    fn_output_detail = 'results_detail.' + fit_method + '.csv'
    f_detail = open(dir_output + fn_output_detail, 'a')
    csv.writer(f_detail).writerow(['dataset', 'graph kernel', 'fit method', 'k', 
              'target', 'repeat', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
              'min dis_k gi', 'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
              'dis_k gi -> GM', 'median set'])
    f_detail.close()
    fn_output_summary = 'results_summary.csv'
    f_summary = open(dir_output + fn_output_summary, 'a')
    csv.writer(f_summary).writerow(['dataset', 'graph kernel', 'fit method', 'k', 
              'target', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
              'min dis_k gi', 'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
              'dis_k gi -> GM', '# SOD SM -> GM', '# dis_k SM -> GM', 
              '# dis_k gi -> SM', '# dis_k gi -> GM', 'repeats better SOD SM -> GM', 
              'repeats better dis_k SM -> GM', 'repeats better dis_k gi -> SM', 
              'repeats better dis_k gi -> GM'])
    f_summary.close()
    
    random.seed(1)
    rdn_seed_list = random.sample(range(0, repeats * 100), repeats)
    
    for k in k_list:
        print('\n--------- k =', k, '----------')
        
        sod_sm_mean_list = []
        sod_gm_mean_list = []
        dis_k_sm_mean_list = []
        dis_k_gm_mean_list = []
        dis_k_gi_min_mean_list = []
#        nb_sod_sm2gm = [0, 0, 0]
#        nb_dis_k_sm2gm = [0, 0, 0]
#        nb_dis_k_gi2sm = [0, 0, 0]
#        nb_dis_k_gi2gm = [0, 0, 0]
#        repeats_better_sod_sm2gm = []
#        repeats_better_dis_k_sm2gm = []
#        repeats_better_dis_k_gi2sm = []
#        repeats_better_dis_k_gi2gm = []
        
        for i, (y, values) in enumerate(y_idx.items()):
            print('\ny =', y)
#            y = 'I'
#            values = y_idx[y]
            
#            k = len(values)
#            k = kkk
            
            sod_sm_list = []
            sod_gm_list = []
            dis_k_sm_list = []
            dis_k_gm_list = []
            dis_k_gi_min_list = []
            nb_sod_sm2gm = [0, 0, 0]
            nb_dis_k_sm2gm = [0, 0, 0]
            nb_dis_k_gi2sm = [0, 0, 0]
            nb_dis_k_gi2gm = [0, 0, 0]
            repeats_better_sod_sm2gm = []
            repeats_better_dis_k_sm2gm = []
            repeats_better_dis_k_gi2sm = []
            repeats_better_dis_k_gi2gm = []
            
            for repeat in range(repeats):
                print('\nrepeat =', repeat)
                random.seed(rdn_seed_list[repeat])
                median_set_idx_idx = random.sample(range(0, len(values)), k)
                median_set_idx = [values[idx] for idx in median_set_idx_idx]
                print('median set: ', median_set_idx)
                Gn_median = [Gn[g] for g in values]
        
                sod_sm, sod_gm, dis_k_sm, dis_k_gm, dis_k_gi, dis_k_gi_min, idx_dis_k_gi_min \
                    = median_on_k_closest_graphs(Gn_median, node_label, edge_label, 
                        gkernel, k, fit_method=fit_method, graph_dir=ds['graph_dir'],
                        edit_costs=None, group_min=median_set_idx_idx, 
                        dataset='Letter', parallel=False)
                    
                # write result detail.
                sod_sm2gm = getRelations(np.sign(sod_gm - sod_sm))
                dis_k_sm2gm = getRelations(np.sign(dis_k_gm - dis_k_sm))
                dis_k_gi2sm = getRelations(np.sign(dis_k_sm - dis_k_gi_min))
                dis_k_gi2gm = getRelations(np.sign(dis_k_gm - dis_k_gi_min))
                f_detail = open(dir_output + fn_output_detail, 'a')
                csv.writer(f_detail).writerow([ds_name, gkernel, fit_method, k, 
                          y, repeat,
                          sod_sm, sod_gm, dis_k_sm, dis_k_gm, 
                          dis_k_gi_min, sod_sm2gm, dis_k_sm2gm, dis_k_gi2sm,
                          dis_k_gi2gm, median_set_idx])
                f_detail.close()
                
                # compute result summary.
                sod_sm_list.append(sod_sm)
                sod_gm_list.append(sod_gm)
                dis_k_sm_list.append(dis_k_sm)
                dis_k_gm_list.append(dis_k_gm)
                dis_k_gi_min_list.append(dis_k_gi_min)
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
                fn_pre_sm_new = dir_output + 'medians/set_median.k' + str(int(k)) + '.y' + y + '.repeat' + str(repeat)
                copyfile(fname_sm, fn_pre_sm_new + '.gxl')
                fname_gm = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/output/tmp_ged/gen_median.gxl'
                fn_pre_gm_new = dir_output + 'medians/gen_median.k' + str(int(k)) + '.y' + y + '.repeat' + str(repeat)
                copyfile(fname_gm, fn_pre_gm_new + '.gxl')
                G_best_kernel = Gn_median[idx_dis_k_gi_min].copy()
                reform_attributes(G_best_kernel)
                fn_pre_g_best_kernel = dir_output + 'medians/g_best_kernel.k' + str(int(k)) + '.y' + y + '.repeat' + str(repeat)
                saveGXL(G_best_kernel, fn_pre_g_best_kernel + '.gxl', method='gedlib-letter')
                
                # plot median graphs.
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
            sod_sm2gm_mean = getRelations(np.sign(sod_gm_mean_list[-1] - sod_sm_mean_list[-1]))
            dis_k_sm2gm_mean = getRelations(np.sign(dis_k_gm_mean_list[-1] - dis_k_sm_mean_list[-1]))
            dis_k_gi2sm_mean = getRelations(np.sign(dis_k_sm_mean_list[-1] - dis_k_gi_min_mean_list[-1]))
            dis_k_gi2gm_mean = getRelations(np.sign(dis_k_gm_mean_list[-1] - dis_k_gi_min_mean_list[-1]))
            f_summary = open(dir_output + fn_output_summary, 'a')
            csv.writer(f_summary).writerow([ds_name, gkernel, fit_method, k, y,
                      sod_sm_mean_list[-1], sod_gm_mean_list[-1], 
                      dis_k_sm_mean_list[-1], dis_k_gm_mean_list[-1],
                      dis_k_gi_min_mean_list[-1], sod_sm2gm_mean, dis_k_sm2gm_mean, 
                      dis_k_gi2sm_mean, dis_k_gi2gm_mean, nb_sod_sm2gm, 
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
        sod_sm2gm_mean = getRelations(np.sign(sod_gm_mean - sod_sm_mean))
        dis_k_sm2gm_mean = getRelations(np.sign(dis_k_gm_mean - dis_k_sm_mean))
        dis_k_gi2sm_mean = getRelations(np.sign(dis_k_sm_mean - dis_k_gi_min_mean))
        dis_k_gi2gm_mean = getRelations(np.sign(dis_k_gm_mean - dis_k_gi_min_mean))
        f_summary = open(dir_output + fn_output_summary, 'a')
        csv.writer(f_summary).writerow([ds_name, gkernel, fit_method, k, 'all',
                  sod_sm_mean, sod_gm_mean, dis_k_sm_mean, dis_k_gm_mean,
                  dis_k_gi_min_mean, sod_sm2gm_mean, dis_k_sm2gm_mean, 
                  dis_k_gi2sm_mean, dis_k_gi2gm_mean])
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
    xp_letter_h()