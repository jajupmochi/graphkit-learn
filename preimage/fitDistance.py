#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:20:06 2019

@author: ljia
"""
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "../")
from pygraph.utils.graphfiles import loadDataset
from ged import GED, get_nb_edit_operations
from utils import kernel_distance_matrix

def fit_GED_to_kernel_distance(Gn, gkernel, itr_max):
    c_vi = 1
    c_vr = 1
    c_vs = 1
    c_ei = 1
    c_er = 1
    c_es = 1
    
    # compute distances in feature space.
    dis_k_mat, _, _, _ = kernel_distance_matrix(Gn, gkernel=gkernel)
    dis_k_vec = []
    for i in range(len(dis_k_mat)):
        for j in range(i, len(dis_k_mat)):
            dis_k_vec.append(dis_k_mat[i, j])
    dis_k_vec = np.array(dis_k_vec)
    
    residual_list = []
    edit_cost_list = []
    
    for itr in range(itr_max):
        print('iteration', itr)
        ged_all = []
        n_vi_all = []
        n_vr_all = []
        n_vs_all = []
        n_ei_all = []
        n_er_all = []
        n_es_all = []
        # compute GEDs and numbers of edit operations.
        edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
        edit_cost_list.append(edit_cost_constant)
        for i in tqdm(range(len(Gn)), desc='computing GEDs', file=sys.stdout):
#        for i in range(len(Gn)):
            for j in range(i, len(Gn)):
                dis, pi_forward, pi_backward = GED(Gn[i], Gn[j], lib='gedlibpy', 
                    cost='CONSTANT', method='IPFP', 
                    edit_cost_constant=edit_cost_constant, stabilizer='min', 
                    repeat=30)
                ged_all.append(dis)
                n_vi, n_vr, n_vs, n_ei, n_er, n_es = get_nb_edit_operations(Gn[i], 
                    Gn[j], pi_forward, pi_backward)
                n_vi_all.append(n_vi) 
                n_vr_all.append(n_vr)
                n_vs_all.append(n_vs) 
                n_ei_all.append(n_ei) 
                n_er_all.append(n_er)
                n_es_all.append(n_es)
                
        residual = np.sqrt(np.sum(np.square(np.array(ged_all) - dis_k_vec)))
        residual_list.append(residual)
        
        # "fit" geds to distances in feature space by tuning edit costs using the
        # Least Squares Method.
        nb_cost_mat = np.column_stack((np.array(n_vi_all), np.array(n_vr_all),
                                       np.array(n_vs_all), np.array(n_ei_all),
                                       np.array(n_er_all), np.array(n_es_all)))
        edit_costs, residual, _, _ = np.linalg.lstsq(nb_cost_mat, dis_k_vec,
                                                     rcond=None)
        for i in range(len(edit_costs)):
            if edit_costs[i] < 0:
                if edit_costs[i] > -1e-3:
                    edit_costs[i] = 0
#                else:
#                    raise ValueError('The edit cost is negative.')
            
        c_vi = edit_costs[0]
        c_vr = edit_costs[1]
        c_vs = edit_costs[2]
        c_ei = edit_costs[3]
        c_er = edit_costs[4]
        c_es = edit_costs[5]
    
    return c_vi, c_vr, c_vs, c_ei, c_er, c_es, residual_list, edit_cost_list



if __name__ == '__main__':
    from utils import remove_edges
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    Gn = Gn[0:10]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    itr_max = 10
    c_vi, c_vr, c_vs, c_ei, c_er, c_es, residual_list, edit_cost_list = \
        fit_GED_to_kernel_distance(Gn, gkernel, itr_max)