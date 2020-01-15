#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:20:06 2019

@author: ljia
"""
import numpy as np
from tqdm import tqdm
from itertools import combinations_with_replacement, combinations
import multiprocessing
from multiprocessing import Pool
from functools import partial
import time
import random

from scipy import optimize
import cvxpy as cp

import sys
#sys.path.insert(0, "../")
from ged import GED, get_nb_edit_operations
from utils import kernel_distance_matrix

def fit_GED_to_kernel_distance(Gn, node_label, edge_label, gkernel, itr_max, k=4,
                               params_ged={'lib': 'gedlibpy', 'cost': 'CONSTANT', 
                                           'method': 'IPFP', 'stabilizer': None},
                               init_costs=[3, 3, 1, 3, 3, 1],
                               parallel=True):
    # c_vi, c_vr, c_vs, c_ei, c_er, c_es or parts of them.
#    random.seed(1)
#    cost_rdm = random.sample(range(1, 10), 6)
#    init_costs = cost_rdm + [0]
#    init_costs = cost_rdm
    init_costs = [3, 3, 1, 3, 3, 1]
#    init_costs = [i * 0.01 for i in cost_rdm] + [0]
#    init_costs = [0.2, 0.2, 0.2, 0.2, 0.2, 0]
#    init_costs = [0, 0, 0.9544, 0.026, 0.0196, 0]
#    init_costs = [0.008429912251810438, 0.025461055985319694, 0.2047320869225948, 0.004148727085832133, 0.0, 0]
#    idx_cost_nonzeros = [i for i, item in enumerate(edit_costs) if item != 0]
    
    # compute distances in feature space.
    dis_k_mat, _, _, _ = kernel_distance_matrix(Gn, node_label, edge_label, gkernel=gkernel)
    dis_k_vec = []
    for i in range(len(dis_k_mat)):
#        for j in range(i, len(dis_k_mat)):
        for j in range(i + 1, len(dis_k_mat)):
            dis_k_vec.append(dis_k_mat[i, j])
    dis_k_vec = np.array(dis_k_vec)
    
    # init ged.
    print('\ninitial:')
    time0 = time.time()
    params_ged['edit_cost_constant'] = init_costs
    ged_vec_init, ged_mat, n_edit_operations = compute_geds(Gn, params_ged, 
                                                            parallel=parallel)
    residual_list = [np.sqrt(np.sum(np.square(np.array(ged_vec_init) - dis_k_vec)))]    
    time_list = [time.time() - time0]
    edit_cost_list = [init_costs]  
    nb_cost_mat = np.array(n_edit_operations)
    nb_cost_mat_list = [nb_cost_mat]
    print('edit_costs:', init_costs)
    print('residual_list:', residual_list)
    
    for itr in range(itr_max):
        print('\niteration', itr)
        time0 = time.time()
        # "fit" geds to distances in feature space by tuning edit costs using the
        # Least Squares Method.
        edit_costs_new, residual = update_costs(nb_cost_mat, dis_k_vec)
        for i in range(len(edit_costs_new)):
            if edit_costs_new[i] < 0:
                if edit_costs_new[i] > -1e-9:
                    edit_costs_new[i] = 0
                else:
                    raise ValueError('The edit cost is negative.')
#        for i in range(len(edit_costs_new)):
#            if edit_costs_new[i] < 0:
#                edit_costs_new[i] = 0

        # compute new GEDs and numbers of edit operations.
        params_ged['edit_cost_constant'] = edit_costs_new
        ged_vec, ged_mat, n_edit_operations = compute_geds(Gn, params_ged, 
                                                           parallel=parallel)
        residual_list.append(np.sqrt(np.sum(np.square(np.array(ged_vec) - dis_k_vec))))
        time_list.append(time.time() - time0)
        edit_cost_list.append(edit_costs_new)
        nb_cost_mat = np.array(n_edit_operations)
        nb_cost_mat_list.append(nb_cost_mat)                        
        print('edit_costs:', edit_costs_new)
        print('residual_list:', residual_list)
    
    return edit_costs_new, residual_list, edit_cost_list, dis_k_mat, ged_mat, \
        time_list, nb_cost_mat_list


def compute_geds(Gn, params_ged, parallel=False):
    ged_mat = np.zeros((len(Gn), len(Gn)))
    if parallel:
#        print('parallel')
#        len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
        len_itr = int(len(Gn) * (len(Gn) - 1) / 2)
        ged_vec = [0 for i in range(len_itr)]
        n_edit_operations = [0 for i in range(len_itr)]
#        itr = combinations_with_replacement(range(0, len(Gn)), 2)
        itr = combinations(range(0, len(Gn)), 2)
        n_jobs = multiprocessing.cpu_count()
        if len_itr < 100 * n_jobs:
            chunksize = int(len_itr / n_jobs) + 1
        else:
            chunksize = 100
        def init_worker(gn_toshare):
            global G_gn
            G_gn = gn_toshare
        do_partial = partial(_wrapper_compute_ged_parallel, params_ged)
        pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(Gn,))
        iterator = tqdm(pool.imap_unordered(do_partial, itr, chunksize),
                        desc='computing GEDs', file=sys.stdout)
#        iterator = pool.imap_unordered(do_partial, itr, chunksize)
        for i, j, dis, n_eo_tmp in iterator:
            idx_itr = int(len(Gn) * i + j - (i + 1) * (i + 2) / 2)
            ged_vec[idx_itr] = dis
            ged_mat[i][j] = dis
            ged_mat[j][i] = dis
            n_edit_operations[idx_itr] = n_eo_tmp
#            print('\n-------------------------------------------')
#            print(i, j, idx_itr, dis)
        pool.close()
        pool.join()
        
    else:
        ged_vec = []
        n_edit_operations = []
        for i in tqdm(range(len(Gn)), desc='computing GEDs', file=sys.stdout):
#        for i in range(len(Gn)):
            for j in range(i + 1, len(Gn)):
                dis, pi_forward, pi_backward = GED(Gn[i], Gn[j], **params_ged)
                ged_vec.append(dis)
                ged_mat[i][j] = dis
                ged_mat[j][i] = dis
                n_eo_tmp = get_nb_edit_operations(Gn[i], Gn[j], pi_forward, pi_backward)
                n_edit_operations.append(n_eo_tmp)
                    
    return ged_vec, ged_mat, n_edit_operations
                    

def _wrapper_compute_ged_parallel(params_ged, itr):
    i = itr[0]
    j = itr[1]
    dis, n_eo_tmp = _compute_ged_parallel(G_gn[i], G_gn[j], params_ged)
    return i, j, dis, n_eo_tmp


def _compute_ged_parallel(g1, g2, params_ged):
    dis, pi_forward, pi_backward = GED(g1, g2, **params_ged)
    n_eo_tmp = get_nb_edit_operations(g1, g2, pi_forward, pi_backward)       
    return dis, n_eo_tmp


def update_costs(nb_cost_mat, dis_k_vec):
#    # method 1: simple least square method.
#    edit_costs_new, residual, _, _ = np.linalg.lstsq(nb_cost_mat, dis_k_vec,
#                                                     rcond=None)
    
#    # method 2: least square method with x_i >= 0.
#    edit_costs_new, residual = optimize.nnls(nb_cost_mat, dis_k_vec)
    
    # method 3: solve as a quadratic program with constraints.
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
    constraints = [x >= [0.0001 for i in range(nb_cost_mat.shape[1])],
#                   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
                   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
                   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    edit_costs_new = x.value
    residual = np.sqrt(prob.value)
    
    # method 4: 
    
    return edit_costs_new, residual


if __name__ == '__main__':
    print('check test_fitDistance.py')