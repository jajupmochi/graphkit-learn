#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:20:06 2019

@author: ljia
"""
import numpy as np
from tqdm import tqdm

from scipy import optimize
import cvxpy as cp

import sys
sys.path.insert(0, "../")
from pygraph.utils.graphfiles import loadDataset
from ged import GED, get_nb_edit_operations
from utils import kernel_distance_matrix

def fit_GED_to_kernel_distance(Gn, gkernel, itr_max):
    # c_vi, c_vr, c_vs, c_ei, c_er, c_es or parts of them.
    edit_costs = [0.2, 0.2, 0.2, 0.2, 0.2, 0]
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
    
    for itr in range(itr_max):
        print('iteration', itr)
        ged_all = []
        n_edit_operations = [[] for i in range(len(idx_nonzeros))]
        # compute GEDs and numbers of edit operations.
        edit_cost_constant = [i for i in edit_costs]
        edit_cost_list.append(edit_cost_constant)
        for i in tqdm(range(len(Gn)), desc='computing GEDs', file=sys.stdout):
#        for i in range(len(Gn)):
            for j in range(i, len(Gn)):
                dis, pi_forward, pi_backward = GED(Gn[i], Gn[j], lib='gedlibpy', 
                    cost='CONSTANT', method='IPFP', 
                    edit_cost_constant=edit_cost_constant, stabilizer='min', 
                    repeat=30)
                ged_all.append(dis)
                n_eo_tmp = get_nb_edit_operations(Gn[i], 
                    Gn[j], pi_forward, pi_backward)
                for idx, item in enumerate(idx_nonzeros):
                    n_edit_operations[idx].append(n_eo_tmp[item])
                
        residual = np.sqrt(np.sum(np.square(np.array(ged_all) - dis_k_vec)))
        residual_list.append(residual)
        
        # "fit" geds to distances in feature space by tuning edit costs using the
        # Least Squares Method.
        nb_cost_mat = np.array(n_edit_operations).T
        edit_costs_new, residual = get_better_costs(nb_cost_mat, dis_k_vec)

        print(residual)
        for i in range(len(edit_costs_new)):
            if edit_costs_new[i] < 0:
                if edit_costs_new[i] > -1e-6:
                    edit_costs_new[i] = 0
                else:
                    raise ValueError('The edit cost is negative.')
        
        for idx, item in enumerate(idx_nonzeros):
            edit_costs[item] = edit_costs_new[idx]
    
    return edit_costs, residual_list, edit_cost_list


def get_better_costs(nb_cost_mat, dis_k_vec):
#    # method 1: simple least square method.
#    edit_costs_new, residual, _, _ = np.linalg.lstsq(nb_cost_mat, dis_k_vec,
#                                                     rcond=None)
    
#    # method 2: least square method with x_i >= 0.
#    edit_costs_new, residual = optimize.nnls(nb_cost_mat, dis_k_vec)
    
    # method 3: solve as a quadratic program with constraints: x_i >= 0, sum(x) = 1.
    P = np.dot(nb_cost_mat.T, nb_cost_mat)
    q_T = -2 * np.dot(dis_k_vec.T, nb_cost_mat)
    G = -1 * np.identity(nb_cost_mat.shape[1])
    h = np.array([0 for i in range(nb_cost_mat.shape[1])])
    A = np.array([1 for i in range(nb_cost_mat.shape[1])])
    b = 1
    x = cp.Variable(nb_cost_mat.shape[1])
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q_T@x),
                      [G@x <= h,
                       A@x == b])
    prob.solve()
    edit_costs_new = x.value
    residual = prob.value - np.dot(dis_k_vec.T, dis_k_vec)
    
#    p = program(minimize(norm2(nb_cost_mat*x-dis_k_vec)),[equals(sum(x),1),geq(x,0)])
    return edit_costs_new, residual


if __name__ == '__main__':
    from utils import remove_edges
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:10]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    itr_max = 10
    edit_costs, residual_list, edit_cost_list = \
        fit_GED_to_kernel_distance(Gn, gkernel, itr_max)