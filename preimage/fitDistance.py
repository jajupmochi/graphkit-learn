#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:20:06 2019

@author: ljia
"""
import numpy as np
from tqdm import tqdm
from itertools import combinations_with_replacement
import multiprocessing
from multiprocessing import Pool
from functools import partial
import time
import random

from scipy import optimize
import cvxpy as cp

import sys
sys.path.insert(0, "../")
from ged import GED, get_nb_edit_operations
from utils import kernel_distance_matrix

def fit_GED_to_kernel_distance(Gn, gkernel, itr_max):
    # c_vi, c_vr, c_vs, c_ei, c_er, c_es or parts of them.
    random.seed(1)
    cost_rdm = random.sample(range(1, 10), 5)
    edit_costs = cost_rdm + [0]
#    edit_costs = [0.2, 0.2, 0.2, 0.2, 0.2, 0]
#    edit_costs = [0, 0, 0.9544, 0.026, 0.0196, 0]
#    edit_costs = [0.008429912251810438, 0.025461055985319694, 0.2047320869225948, 0.004148727085832133, 0.0, 0]
    idx_nonzeros = [i for i, item in enumerate(edit_costs) if item != 0]
    
    # compute distances in feature space.
    dis_k_mat, _, _, _ = kernel_distance_matrix(Gn, gkernel=gkernel)
    dis_k_vec = []
    for i in range(len(dis_k_mat)):
        for j in range(i, len(dis_k_mat)):
            dis_k_vec.append(dis_k_mat[i, j])
    dis_k_vec = np.array(dis_k_vec)
    
    residual_list = []
    edit_cost_list = []
    time_list = []
    
    for itr in range(itr_max):
        print('\niteration', itr)
        time0 = time.time()
        # compute GEDs and numbers of edit operations.
        edit_cost_constant = [i for i in edit_costs]
        edit_cost_list.append(edit_cost_constant)
        
        ged_all, ged_mat, n_edit_operations = compute_geds(Gn, edit_cost_constant, 
            idx_nonzeros, parallel=True)
                
        residual = np.sqrt(np.sum(np.square(np.array(ged_all) - dis_k_vec)))
        residual_list.append(residual)
        
        # "fit" geds to distances in feature space by tuning edit costs using the
        # Least Squares Method.
        nb_cost_mat = np.array(n_edit_operations).T
        edit_costs_new, residual = compute_better_costs(nb_cost_mat, dis_k_vec)

        print('pseudo residual:', residual)
        for i in range(len(edit_costs_new)):
            if edit_costs_new[i] < 0:
                if edit_costs_new[i] > -1e-9:
                    edit_costs_new[i] = 0
                else:
                    raise ValueError('The edit cost is negative.')
        
        for idx, item in enumerate(idx_nonzeros):
            edit_costs[item] = edit_costs_new[idx]
        
        time_list.append(time.time() - time0)
            
        print('edit_costs:', edit_costs)
        print('residual_list:', residual_list)
        
        
    edit_cost_list.append(edit_costs)
    ged_all, ged_mat, n_edit_operations = compute_geds(Gn, edit_costs, 
            idx_nonzeros, parallel=True)
    residual = np.sqrt(np.sum(np.square(np.array(ged_all) - dis_k_vec)))
    residual_list.append(residual)
    
    return edit_costs, residual_list, edit_cost_list, dis_k_mat, ged_mat, time_list


def compute_geds(Gn, edit_cost_constant, idx_nonzeros, parallel=False):
    ged_mat = np.zeros((len(Gn), len(Gn)))
    if parallel:
#        print('parallel')
        len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
        ged_all = [0 for i in range(len_itr)]
        n_edit_operations = [[0 for i in range(len_itr)] for j in 
                              range(len(idx_nonzeros))]
               
        itr = combinations_with_replacement(range(0, len(Gn)), 2)
        n_jobs = multiprocessing.cpu_count()
        if len_itr < 100 * n_jobs:
            chunksize = int(len_itr / n_jobs) + 1
        else:
            chunksize = 100
        def init_worker(gn_toshare):
            global G_gn
            G_gn = gn_toshare
        do_partial = partial(_wrapper_compute_ged_parallel, edit_cost_constant, 
                             idx_nonzeros)
        pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(Gn,))
        iterator = tqdm(pool.imap_unordered(do_partial, itr, chunksize),
                        desc='computing GEDs', file=sys.stdout)
#        iterator = pool.imap_unordered(do_partial, itr, chunksize)
        for i, j, dis, n_eo_tmp in iterator:
            idx_itr = int(len(Gn) * i + j - i * (i + 1) / 2)
            ged_all[idx_itr] = dis
            ged_mat[i][j] = dis
            ged_mat[j][i] = dis
            for idx, item in enumerate(idx_nonzeros):
                n_edit_operations[idx][idx_itr] = n_eo_tmp[item]
#            print('\n-------------------------------------------')
#            print(i, j, idx_itr, dis)
        pool.close()
        pool.join()
        
    else:
        ged_all = []
        n_edit_operations = [[] for i in range(len(idx_nonzeros))]
        for i in tqdm(range(len(Gn)), desc='computing GEDs', file=sys.stdout):
#        for i in range(len(Gn)):
            for j in range(i, len(Gn)):
#                time0 = time.time()
                dis, pi_forward, pi_backward = GED(Gn[i], Gn[j], lib='gedlibpy', 
                    cost='CONSTANT', method='IPFP', 
                    edit_cost_constant=edit_cost_constant, stabilizer='min', 
                    repeat=50)
#                time1 = time.time() - time0
#                time0 = time.time()
                ged_all.append(dis)
                ged_mat[i][j] = dis
                ged_mat[j][i] = dis
                n_eo_tmp = get_nb_edit_operations(Gn[i], Gn[j], pi_forward, pi_backward)
                for idx, item in enumerate(idx_nonzeros):
                    n_edit_operations[idx].append(n_eo_tmp[item])
#                time2 = time.time() - time0
#                print(time1, time2, time1 / time2)
                    
    return ged_all, ged_mat, n_edit_operations
                    

def _wrapper_compute_ged_parallel(edit_cost_constant, idx_nonzeros, itr):
    i = itr[0]
    j = itr[1]
    dis, n_eo_tmp = _compute_ged_parallel(G_gn[i], G_gn[j], edit_cost_constant, 
                                          idx_nonzeros)
    return i, j, dis, n_eo_tmp


def _compute_ged_parallel(g1, g2, edit_cost_constant, idx_nonzeros):
    dis, pi_forward, pi_backward = GED(g1, g2, lib='gedlibpy', 
        cost='CONSTANT', method='IPFP', 
        edit_cost_constant=edit_cost_constant, stabilizer='min', 
        repeat=50)
    n_eo_tmp = get_nb_edit_operations(g1, g2, pi_forward, pi_backward)
        
    return dis, n_eo_tmp


def compute_better_costs(nb_cost_mat, dis_k_vec):
#    # method 1: simple least square method.
#    edit_costs_new, residual, _, _ = np.linalg.lstsq(nb_cost_mat, dis_k_vec,
#                                                     rcond=None)
    
#    # method 2: least square method with x_i >= 0.
#    edit_costs_new, residual = optimize.nnls(nb_cost_mat, dis_k_vec)
    
    # method 3: solve as a quadratic program with constraints: x_i >= 0, sum(x) = 1.
#    P = np.dot(nb_cost_mat.T, nb_cost_mat)
#    q_T = -2 * np.dot(dis_k_vec.T, nb_cost_mat)
#    G = -1 * np.identity(nb_cost_mat.shape[1])
#    h = np.array([0 for i in range(nb_cost_mat.shape[1])])
#    A = np.array([1 for i in range(nb_cost_mat.shape[1])])
#    b = 1
#    x = cp.Variable(nb_cost_mat.shape[1])
#    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q_T@x),
#                      [G@x <= h])
#    prob.solve()
#    edit_costs_new = x.value
#    residual = prob.value - np.dot(dis_k_vec.T, dis_k_vec)
    
#    G = -1 * np.identity(nb_cost_mat.shape[1])
#    h = np.array([0 for i in range(nb_cost_mat.shape[1])])
    x = cp.Variable(nb_cost_mat.shape[1])
    cost = cp.sum_squares(nb_cost_mat * x - dis_k_vec)
    constraints = [x >= [0 for i in range(nb_cost_mat.shape[1])]]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    edit_costs_new = x.value
    residual = np.sqrt(prob.value)
    
    # method 4: 
    
    return edit_costs_new, residual


if __name__ == '__main__':
    print('check test_fitDistance.py')