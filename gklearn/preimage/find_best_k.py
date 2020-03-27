#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:54:32 2020

@author: ljia
"""
import numpy as np
import random
import csv

from gklearn.utils.graphfiles import loadDataset
from gklearn.preimage.test_k_closest_graphs import median_on_k_closest_graphs

def find_best_k():
    ds = {'name': 'monoterpenoides', 
          'dataset': '../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
    gkernel = 'treeletkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
    ds_name = 'mono'
    dir_output = 'results/test_find_best_k/'
    
    repeats = 50
    k_list = range(2, 11)
    fit_method = 'k-graphs'
    # fitted on the whole dataset - treelet - mono
    edit_costs = [0.1268873773592978, 0.004084633224249829, 0.0897581955378986, 0.15328856114451297, 0.3109956881625734, 0.0]
    
    # create result files.
    fn_output_detail = 'results_detail.' + fit_method + '.csv'
    f_detail = open(dir_output + fn_output_detail, 'a')
    csv.writer(f_detail).writerow(['dataset', 'graph kernel', 'fit method', 'k', 
              'repeat', 'median set', 'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
              'min dis_k gi', 'SOD SM -> GM', 'dis_k SM -> GM', 'dis_k gi -> SM', 
              'dis_k gi -> GM'])
    f_detail.close()
    fn_output_summary = 'results_summary.csv'
    f_summary = open(dir_output + fn_output_summary, 'a')
    csv.writer(f_summary).writerow(['dataset', 'graph kernel', 'fit method', 'k', 
              'SOD SM', 'SOD GM', 'dis_k SM', 'dis_k GM',
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
            median_set_idx = random.sample(range(0, len(Gn)), k)
            print('median set: ', median_set_idx)
            
            sod_sm, sod_gm, dis_k_sm, dis_k_gm, dis_k_gi, dis_k_gi_min \
                = median_on_k_closest_graphs(Gn, node_label, edge_label, gkernel, k, 
                                             fit_method='k-graphs', 
                                             edit_costs=edit_costs,
                                             group_min=median_set_idx,
                                             parallel=False)
                
            # write result detail.
            sod_sm2gm = getRelations(np.sign(sod_gm - sod_sm))
            dis_k_sm2gm = getRelations(np.sign(dis_k_gm - dis_k_sm))
            dis_k_gi2sm = getRelations(np.sign(dis_k_sm - dis_k_gi_min))
            dis_k_gi2gm = getRelations(np.sign(dis_k_gm - dis_k_gi_min))
            f_detail = open(dir_output + fn_output_detail, 'a')
            csv.writer(f_detail).writerow([ds_name, gkernel, fit_method, k, repeat,
                      median_set_idx, sod_sm, sod_gm, dis_k_sm, dis_k_gm, 
                      dis_k_gi_min, sod_sm2gm, dis_k_sm2gm, dis_k_gi2sm,
                      dis_k_gi2gm])
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
            
        # write result summary. 
        sod_sm_mean = np.mean(sod_sm_list)
        sod_gm_mean = np.mean(sod_gm_list)
        dis_k_sm_mean = np.mean(dis_k_sm_list)
        dis_k_gm_mean = np.mean(dis_k_gm_list)
        dis_k_gi_min_mean = np.mean(dis_k_gi_min_list)
        sod_sm2gm_mean = getRelations(np.sign(sod_gm_mean - sod_sm_mean))
        dis_k_sm2gm_mean = getRelations(np.sign(dis_k_gm_mean - dis_k_sm_mean))
        dis_k_gi2sm_mean = getRelations(np.sign(dis_k_sm_mean - dis_k_gi_min_mean))
        dis_k_gi2gm_mean = getRelations(np.sign(dis_k_gm_mean - dis_k_gi_min_mean))
        f_summary = open(dir_output + fn_output_summary, 'a')
        csv.writer(f_summary).writerow([ds_name, gkernel, fit_method, k, 
                  sod_sm_mean, sod_gm_mean, dis_k_sm_mean, dis_k_gm_mean,
                  dis_k_gi_min_mean, sod_sm2gm_mean, dis_k_sm2gm_mean, 
                  dis_k_gi2sm_mean, dis_k_gi2gm_mean, nb_sod_sm2gm, 
                  nb_dis_k_sm2gm, nb_dis_k_gi2sm, nb_dis_k_gi2gm, 
                  repeats_better_sod_sm2gm, repeats_better_dis_k_sm2gm, 
                  repeats_better_dis_k_gi2sm, repeats_better_dis_k_gi2gm])
        f_summary.close()
        
    print('\ncomplete.')
    return


def getRelations(sign):
    if sign == -1:
        return 'better'
    elif sign == 0:
        return 'same'
    elif sign == 1:
        return 'worse'


if __name__ == '__main__':
    find_best_k()