#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:22:04 2020

@author: ljia
"""
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
import random
#import csv
from shutil import copyfile


import sys
sys.path.insert(0, "../")
from preimage.iam import iam_bash
from gklearn.utils.graphfiles import loadDataset, loadGXL
from preimage.ged import GED
from preimage.utils import get_same_item_indices

def test_knn():
    ds = {'name': 'monoterpenoides', 
          'dataset': '../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
#    gkernel = 'treeletkernel'
#    node_label = 'atom'
#    edge_label = 'bond_type'
#    ds_name = 'mono'
    dir_output = 'results/knn/'
    graph_dir='/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/datasets/monoterpenoides/'
    
    k_nn = 1
    percent = 0.1
    repeats = 50
    edit_cost_constant = [3, 3, 1, 3, 3, 1]
    
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    sod_sm_list_list
    for repeat in range(0, repeats):
        print('\n---------------------------------')
        print('repeat =', repeat)
        accuracy_sm_list = []
        accuracy_gm_list = []
        sod_sm_list = []
        sod_gm_list = []
        
        random.seed(repeat)
        set_median_list = []
        gen_median_list = []
        train_y_set = []
        for y, values in y_idx.items():
            print('\ny =', y)
            size_median_set = int(len(values) * percent)
            median_set_idx = random.sample(values, size_median_set)
            print('median set: ', median_set_idx)
            
            # compute set median and gen median using IAM (C++ through bash).
    #        Gn_median = [Gn[idx] for idx in median_set_idx]
            group_fnames = [Gn[g].graph['filename'] for g in median_set_idx]
            sod_sm, sod_gm, fname_sm, fname_gm = iam_bash(group_fnames, edit_cost_constant,
                                                          graph_dir=graph_dir)
            print('sod_sm, sod_gm:', sod_sm, sod_gm)
            sod_sm_list.append(sod_sm)
            sod_gm_list.append(sod_gm)
            fname_sm_new = dir_output + 'medians/set_median.y' + str(int(y)) + '.repeat' + str(repeat) + '.gxl'
            copyfile(fname_sm, fname_sm_new)
            fname_gm_new = dir_output + 'medians/gen_median.y' + str(int(y)) + '.repeat' + str(repeat) + '.gxl'
            copyfile(fname_gm, fname_gm_new)
            set_median_list.append(loadGXL(fname_sm_new))
            gen_median_list.append(loadGXL(fname_gm_new))
            train_y_set.append(int(y))
        
        print(sod_sm, sod_gm)
        
        # do 1-nn.
        test_y_set = [int(y) for y in y_all]
        accuracy_sm = knn(set_median_list, train_y_set, Gn, test_y_set, k=k_nn, distance='ged')
        accuracy_gm = knn(set_median_list, train_y_set, Gn, test_y_set, k=k_nn, distance='ged')
        accuracy_sm_list.append(accuracy_sm)
        accuracy_gm_list.append(accuracy_gm)
        print('current accuracy sm and gm:', accuracy_sm, accuracy_gm)
        
    # output
    accuracy_sm_mean = np.mean(accuracy_sm_list)
    accuracy_gm_mean = np.mean(accuracy_gm_list)
    print('\ntotal average accuracy sm and gm:', accuracy_sm_mean, accuracy_gm_mean)

        
def knn(train_set, train_y_set, test_set, test_y_set, k=1, distance='ged'):
    if k == 1 and distance == 'ged':
        algo_options = '--threads 8 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
        params_ged = {'lib': 'gedlibpy', 'cost': 'CONSTANT', 'method': 'IPFP', 
                    'algo_options': algo_options, 'stabilizer': None}
        accuracy = 0
        for idx_test, g_test in tqdm(enumerate(test_set), desc='computing 1-nn', 
                                     file=sys.stdout):
            dis = np.inf
            for idx_train, g_train in enumerate(train_set):
                dis_cur, _, _ = GED(g_test, g_train, **params_ged)
                if dis_cur < dis:
                    dis = dis_cur
                    test_y_cur = train_y_set[idx_train]
            if test_y_cur == test_y_set[idx_test]:
                accuracy += 1
        accuracy = accuracy / len(test_set)
        
    return accuracy

    

if __name__ == '__main__':
    test_knn()