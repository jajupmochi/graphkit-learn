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
import sys

from scipy import optimize
from scipy.optimize import minimize
import cvxpy as cp

from gklearn.preimage.ged import GED, get_nb_edit_operations, get_nb_edit_operations_letter, get_nb_edit_operations_nonsymbolic
from gklearn.preimage.utils import kernel_distance_matrix

def fit_GED_to_kernel_distance(Gn, node_label, edge_label, gkernel, itr_max,
                               params_ged={'lib': 'gedlibpy', 'cost': 'CONSTANT', 
                                           'method': 'IPFP', 'stabilizer': None},
                               init_costs=[3, 3, 1, 3, 3, 1],
                               dataset='monoterpenoides', Kmatrix=None,
                               parallel=True):
#    dataset = dataset.lower()
    
    # c_vi, c_vr, c_vs, c_ei, c_er, c_es or parts of them.
#    random.seed(1)
#    cost_rdm = random.sample(range(1, 10), 6)
#    init_costs = cost_rdm + [0]
#    init_costs = cost_rdm
#    init_costs = [3, 3, 1, 3, 3, 1]
#    init_costs = [i * 0.01 for i in cost_rdm] + [0]
#    init_costs = [0.2, 0.2, 0.2, 0.2, 0.2, 0]
#    init_costs = [0, 0, 0.9544, 0.026, 0.0196, 0]
#    init_costs = [0.008429912251810438, 0.025461055985319694, 0.2047320869225948, 0.004148727085832133, 0.0, 0]
#    idx_cost_nonzeros = [i for i, item in enumerate(edit_costs) if item != 0]
    
    # compute distances in feature space.
    dis_k_mat, _, _, _ = kernel_distance_matrix(Gn, node_label, edge_label, 
                                                Kmatrix=Kmatrix, gkernel=gkernel)
    dis_k_vec = []
    for i in range(len(dis_k_mat)):
#        for j in range(i, len(dis_k_mat)):
        for j in range(i + 1, len(dis_k_mat)):
            dis_k_vec.append(dis_k_mat[i, j])
    dis_k_vec = np.array(dis_k_vec)
    
    # init ged.
    print('\ninitial:')
    time0 = time.time()
    params_ged['dataset'] = dataset
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
        np.savez('results/xp_fit_method/fit_data_debug' + str(itr) + '.gm', 
                 nb_cost_mat=nb_cost_mat, dis_k_vec=dis_k_vec, 
                 n_edit_operations=n_edit_operations, ged_vec_init=ged_vec_init,
                 ged_mat=ged_mat)
        edit_costs_new, residual = update_costs(nb_cost_mat, dis_k_vec, 
                                                dataset=dataset, cost=params_ged['cost'])
        for i in range(len(edit_costs_new)):
            if -1e-9 <= edit_costs_new[i] <= 1e-9:
                edit_costs_new[i] = 0
            if edit_costs_new[i] < 0:
                raise ValueError('The edit cost is negative.')
#        for i in range(len(edit_costs_new)):
#            if edit_costs_new[i] < 0:
#                edit_costs_new[i] = 0

        # compute new GEDs and numbers of edit operations.
        params_ged['edit_cost_constant'] = edit_costs_new # np.array([edit_costs_new[0], edit_costs_new[1], 0.75])
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
    edit_cost_name = params_ged['cost']
    if edit_cost_name == 'LETTER' or edit_cost_name == 'LETTER2':
        get_nb_eo = get_nb_edit_operations_letter
    elif edit_cost_name == 'NON_SYMBOLIC':
        get_nb_eo = get_nb_edit_operations_nonsymbolic
    else: 
        get_nb_eo = get_nb_edit_operations
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
        do_partial = partial(_wrapper_compute_ged_parallel, params_ged, get_nb_eo)
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
                n_eo_tmp = get_nb_eo(Gn[i], Gn[j], pi_forward, pi_backward)
                n_edit_operations.append(n_eo_tmp)
                    
    return ged_vec, ged_mat, n_edit_operations
                    

def _wrapper_compute_ged_parallel(params_ged, get_nb_eo, itr):
    i = itr[0]
    j = itr[1]
    dis, n_eo_tmp = _compute_ged_parallel(G_gn[i], G_gn[j], params_ged, get_nb_eo)
    return i, j, dis, n_eo_tmp


def _compute_ged_parallel(g1, g2, params_ged, get_nb_eo):
    dis, pi_forward, pi_backward = GED(g1, g2, **params_ged)
    n_eo_tmp = get_nb_eo(g1, g2, pi_forward, pi_backward) # [0,0,0,0,0,0]
    return dis, n_eo_tmp


def update_costs(nb_cost_mat, dis_k_vec, dataset='monoterpenoides', 
                 cost='CONSTANT', rw_constraints='inequality'):
#    if dataset == 'Letter-high':
    if cost == 'LETTER':            
        pass
#        # method 1: set alpha automatically, just tune c_vir and c_eir by 
#        # LMS using cvxpy.
#        alpha = 0.5
#        coeff = 100 # np.max(alpha * nb_cost_mat[:,4] / dis_k_vec)
##        if np.count_nonzero(nb_cost_mat[:,4]) == 0:
##            alpha = 0.75
##        else:
##            alpha = np.min([dis_k_vec / c_vs for c_vs in nb_cost_mat[:,4] if c_vs != 0])
##        alpha = alpha * 0.99
#        param_vir = alpha * (nb_cost_mat[:,0] + nb_cost_mat[:,1])
#        param_eir = (1 - alpha) * (nb_cost_mat[:,4] + nb_cost_mat[:,5])
#        nb_cost_mat_new = np.column_stack((param_vir, param_eir))
#        dis_new = coeff * dis_k_vec - alpha * nb_cost_mat[:,3]
#        
#        x = cp.Variable(nb_cost_mat_new.shape[1])
#        cost = cp.sum_squares(nb_cost_mat_new * x - dis_new)
#        constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])]]
#        prob = cp.Problem(cp.Minimize(cost), constraints)
#        prob.solve()
#        edit_costs_new = x.value
#        edit_costs_new = np.array([edit_costs_new[0], edit_costs_new[1], alpha])
#        residual = np.sqrt(prob.value)
    
#        # method 2: tune c_vir, c_eir and alpha by nonlinear programming by 
#        # scipy.optimize.minimize.
#        w0 = nb_cost_mat[:,0] + nb_cost_mat[:,1]
#        w1 = nb_cost_mat[:,4] + nb_cost_mat[:,5]
#        w2 = nb_cost_mat[:,3]
#        w3 = dis_k_vec
#        func_min = lambda x: np.sum((w0 * x[0] * x[3] + w1 * x[1] * (1 - x[2]) \
#                             + w2 * x[2] - w3 * x[3]) ** 2)
#        bounds = ((0, None), (0., None), (0.5, 0.5), (0, None))
#        res = minimize(func_min, [0.9, 1.7, 0.75, 10], bounds=bounds)
#        edit_costs_new = res.x[0:3]
#        residual = res.fun
    
    # method 3: tune c_vir, c_eir and alpha by nonlinear programming using cvxpy.
    
    
#        # method 4: tune c_vir, c_eir and alpha by QP function
#        # scipy.optimize.least_squares. An initial guess is required.
#        w0 = nb_cost_mat[:,0] + nb_cost_mat[:,1]
#        w1 = nb_cost_mat[:,4] + nb_cost_mat[:,5]
#        w2 = nb_cost_mat[:,3]
#        w3 = dis_k_vec
#        func = lambda x: (w0 * x[0] * x[3] + w1 * x[1] * (1 - x[2]) \
#                             + w2 * x[2] - w3 * x[3]) ** 2
#        res = optimize.root(func, [0.9, 1.7, 0.75, 100])
#        edit_costs_new = res.x
#        residual = None
    elif cost == 'LETTER2':
#            # 1. if c_vi != c_vr, c_ei != c_er.
#            nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
#            x = cp.Variable(nb_cost_mat_new.shape[1])
#            cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
##            # 1.1 no constraints.
##            constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])]]
#            # 1.2 c_vs <= c_vi + c_vr.
#            constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
#                           np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]            
##            # 2. if c_vi == c_vr, c_ei == c_er.
##            nb_cost_mat_new = nb_cost_mat[:,[0,3,4]]
##            nb_cost_mat_new[:,0] += nb_cost_mat[:,1]
##            nb_cost_mat_new[:,2] += nb_cost_mat[:,5]
##            x = cp.Variable(nb_cost_mat_new.shape[1])
##            cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
##            # 2.1 no constraints.
##            constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])]]
###            # 2.2 c_vs <= c_vi + c_vr.
###            constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
###                           np.array([2.0, -1.0, 0.0]).T@x >= 0.0]     
#            
#            prob = cp.Problem(cp.Minimize(cost_fun), constraints)
#            prob.solve()
#            edit_costs_new = [x.value[0], x.value[0], x.value[1], x.value[2], x.value[2]]
#            edit_costs_new = np.array(edit_costs_new)
#            residual = np.sqrt(prob.value)
        if rw_constraints == 'inequality':
            # c_vs <= c_vi + c_vr.
            nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
            x = cp.Variable(nb_cost_mat_new.shape[1])
            cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
            constraints = [x >= [0.001 for i in range(nb_cost_mat_new.shape[1])],
                           np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
            prob = cp.Problem(cp.Minimize(cost_fun), constraints)
            try:
                prob.solve(verbose=True)
            except MemoryError as error0:
                print('\nUsing solver "OSQP" caused a memory error.')
                print('the original error message is\n', error0)
                print('solver status: ', prob.status)
                print('trying solver "CVXOPT" instead...\n')
                try:
                    prob.solve(solver=cp.CVXOPT, verbose=True)
                except Exception as error1:
                    print('\nAn error occured when using solver "CVXOPT".')
                    print('the original error message is\n', error1)
                    print('solver status: ', prob.status)
                    print('trying solver "MOSEK" instead. Notice this solver is commercial and a lisence is required.\n')
                    prob.solve(solver=cp.MOSEK, verbose=True)
                else:
                    print('solver status: ', prob.status)                    
            else:
                print('solver status: ', prob.status)
            print()
            edit_costs_new = x.value
            residual = np.sqrt(prob.value)
        elif rw_constraints == '2constraints':
            # c_vs <= c_vi + c_vr and c_vi == c_vr, c_ei == c_er.
            nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
            x = cp.Variable(nb_cost_mat_new.shape[1])
            cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
            constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
                           np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0,
                           np.array([1.0, -1.0, 0.0, 0.0, 0.0]).T@x == 0.0,
                           np.array([0.0, 0.0, 0.0, 1.0, -1.0]).T@x == 0.0]
            prob = cp.Problem(cp.Minimize(cost_fun), constraints)
            prob.solve()
            edit_costs_new = x.value
            residual = np.sqrt(prob.value)
        elif rw_constraints == 'no-constraint':
            # no constraint.
            nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
            x = cp.Variable(nb_cost_mat_new.shape[1])
            cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
            constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
            prob = cp.Problem(cp.Minimize(cost_fun), constraints)
            prob.solve()
            edit_costs_new = x.value
            residual = np.sqrt(prob.value)
#            elif method == 'inequality_modified':
#                # c_vs <= c_vi + c_vr.
#                nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
#                x = cp.Variable(nb_cost_mat_new.shape[1])
#                cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
#                constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
#                               np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
#                prob = cp.Problem(cp.Minimize(cost_fun), constraints)
#                prob.solve()
#                # use same costs for insertion and removal rather than the fitted costs.
#                edit_costs_new = [x.value[0], x.value[0], x.value[1], x.value[2], x.value[2]]
#                edit_costs_new = np.array(edit_costs_new)
#                residual = np.sqrt(prob.value)
    elif cost == 'NON_SYMBOLIC':
        is_n_attr = np.count_nonzero(nb_cost_mat[:,2])
        is_e_attr = np.count_nonzero(nb_cost_mat[:,5])
        
        if dataset == 'SYNTHETICnew':
#            nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
            nb_cost_mat_new = nb_cost_mat[:,[2,3,4]]
            x = cp.Variable(nb_cost_mat_new.shape[1])
            cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
#            constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
#                           np.array([0.0, 0.0, 0.0, 1.0, -1.0]).T@x == 0.0]
#            constraints = [x >= [0.0001 for i in range(nb_cost_mat_new.shape[1])]]
            constraints = [x >= [0.0001 for i in range(nb_cost_mat_new.shape[1])],
                   np.array([0.0, 1.0, -1.0]).T@x == 0.0]
            prob = cp.Problem(cp.Minimize(cost_fun), constraints)
            prob.solve()
#            print(x.value)
            edit_costs_new = np.concatenate((np.array([0.0, 0.0]), x.value, 
                                             np.array([0.0])))
            residual = np.sqrt(prob.value)
            
        elif rw_constraints == 'inequality':
            # c_vs <= c_vi + c_vr.
            if is_n_attr and is_e_attr:
                nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4,5]]
                x = cp.Variable(nb_cost_mat_new.shape[1])
                cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
                constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
                               np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
                               np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
                prob = cp.Problem(cp.Minimize(cost_fun), constraints)
                prob.solve()
                edit_costs_new = x.value
                residual = np.sqrt(prob.value)
            elif is_n_attr and not is_e_attr:
                nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
                x = cp.Variable(nb_cost_mat_new.shape[1])
                cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
                constraints = [x >= [0.001 for i in range(nb_cost_mat_new.shape[1])],
                               np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
                prob = cp.Problem(cp.Minimize(cost_fun), constraints)
                prob.solve()
                print(x.value)
                edit_costs_new = np.concatenate((x.value, np.array([0.0])))
                residual = np.sqrt(prob.value)
            elif not is_n_attr and is_e_attr:
                nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
                x = cp.Variable(nb_cost_mat_new.shape[1])
                cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
                constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
                               np.array([0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
                prob = cp.Problem(cp.Minimize(cost_fun), constraints)
                prob.solve()
                edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
                residual = np.sqrt(prob.value)
            else:
                nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
                x = cp.Variable(nb_cost_mat_new.shape[1])
                cost_fun = cp.sum_squares(nb_cost_mat_new * x - dis_k_vec)
                constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
                prob = cp.Problem(cp.Minimize(cost_fun), constraints)
                prob.solve()
                edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
                                                 x.value[2:], np.array([0.0])))
                residual = np.sqrt(prob.value)
    else:
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
        cost_fun = cp.sum_squares(nb_cost_mat * x - dis_k_vec)
        constraints = [x >= [0.0 for i in range(nb_cost_mat.shape[1])],
    #                   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
                       np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
                       np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
        prob = cp.Problem(cp.Minimize(cost_fun), constraints)
        prob.solve()
        edit_costs_new = x.value
        residual = np.sqrt(prob.value)
    
    # method 4: 
    
    return edit_costs_new, residual


if __name__ == '__main__':
    print('check test_fitDistance.py')