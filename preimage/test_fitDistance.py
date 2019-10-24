#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:50:56 2019

@author: ljia
"""
from matplotlib import pyplot as plt
import numpy as np

from pygraph.utils.graphfiles import loadDataset
from utils import remove_edges
from fitDistance import fit_GED_to_kernel_distance
from utils import normalize_distance_matrix

def test_anycosts():
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:10]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    itr_max = 10
    edit_costs, residual_list, edit_cost_list, dis_k_mat, ged_mat, time_list = \
        fit_GED_to_kernel_distance(Gn, gkernel, itr_max)
    total_time = np.sum(time_list)
    print('\nedit_costs:', edit_costs)
    print('\nresidual_list:', residual_list)
    print('\nedit_cost_list:', edit_cost_list)
    print('\ndistance matrix in kernel space:', dis_k_mat)
    print('\nged matrix:', ged_mat)
    print('total time:', total_time)
    np.savez('results/fit_distance.any_costs.gm', edit_costs=edit_costs, 
             residual_list=residual_list, edit_cost_list=edit_cost_list,
             dis_k_mat=dis_k_mat, ged_mat=ged_mat, time_list=time_list, 
             total_time=total_time)
    
    # normalized distance matrices.
#    gmfile = np.load('results/fit_distance.any_costs.gm.npz')
#    edit_costs = gmfile['edit_costs']
#    residual_list = gmfile['residual_list']
#    edit_cost_list = gmfile['edit_cost_list']
#    dis_k_mat = gmfile['dis_k_mat']
#    ged_mat = gmfile['ged_mat']
#    total_time = gmfile['total_time']
    
    norm_dis_k_mat = normalize_distance_matrix(dis_k_mat)
    plt.imshow(norm_dis_k_mat)
    plt.colorbar()
    plt.savefig('results/norm_dis_k_mat.any_costs' + '.eps', format='eps', dpi=300)
#    plt.savefig('results/norm_dis_k_mat.any_costs' + '.jpg', format='jpg')
#    plt.show()
    plt.clf()
    norm_ged_mat = normalize_distance_matrix(ged_mat)
    plt.imshow(norm_ged_mat)
    plt.colorbar()
    plt.savefig('results/norm_ged_mat.any_costs' + '.eps', format='eps', dpi=300)
#    plt.savefig('results/norm_ged_mat.any_costs' + '.jpg', format='jpg')
#    plt.show()
    plt.clf()
    

def test_cs_leq_ci_plus_cr():
    """c_vs <= c_vi + c_vr, c_es <= c_ei + c_er
    """
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#    Gn = Gn[0:10]
    remove_edges(Gn)
    gkernel = 'marginalizedkernel'
    itr_max = 10
    edit_costs, residual_list, edit_cost_list, dis_k_mat, ged_mat, time_list = \
        fit_GED_to_kernel_distance(Gn, gkernel, itr_max)
    total_time = np.sum(time_list)
    print('\nedit_costs:', edit_costs)
    print('\nresidual_list:', residual_list)
    print('\nedit_cost_list:', edit_cost_list)
    print('\ndistance matrix in kernel space:', dis_k_mat)
    print('\nged matrix:', ged_mat)
    print('total time:', total_time)
    np.savez('results/fit_distance.cs_leq_ci_plus_cr.gm', edit_costs=edit_costs, 
             residual_list=residual_list, edit_cost_list=edit_cost_list,
             dis_k_mat=dis_k_mat, ged_mat=ged_mat, time_list=time_list, 
             total_time=total_time)
    
    # normalized distance matrices.
#    gmfile = np.load('results/fit_distance.cs_leq_ci_plus_cr.gm.npz')
#    edit_costs = gmfile['edit_costs']
#    residual_list = gmfile['residual_list']
#    edit_cost_list = gmfile['edit_cost_list']
#    dis_k_mat = gmfile['dis_k_mat']
#    ged_mat = gmfile['ged_mat']
#    total_time = gmfile['total_time']
    
    norm_dis_k_mat = normalize_distance_matrix(dis_k_mat)
    plt.imshow(norm_dis_k_mat)
    plt.colorbar()
    plt.savefig('results/norm_dis_k_mat.cs_leq_ci_plus_cr' + '.eps', format='eps', dpi=300)
#    plt.savefig('results/norm_dis_k_mat.cs_leq_ci_plus_cr' + '.jpg', format='jpg')
#    plt.show()
    plt.clf()
    norm_ged_mat = normalize_distance_matrix(ged_mat)
    plt.imshow(norm_ged_mat)
    plt.colorbar()
    plt.savefig('results/norm_ged_mat.cs_leq_ci_plus_cr' + '.eps', format='eps', dpi=300)
#    plt.savefig('results/norm_ged_mat.cs_leq_ci_plus_cr' + '.jpg', format='jpg')
#    plt.show()
    plt.clf()
    
    
if __name__ == '__main__':
    test_anycosts()
    test_cs_leq_ci_plus_cr()