#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:50:56 2019

@author: ljia
"""
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "../")
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
    edit_costs, residual_list, edit_cost_list, dis_k_mat, ged_mat, time_list, \
        nb_cost_mat_list, coef_dk = fit_GED_to_kernel_distance(Gn, gkernel, itr_max)
    total_time = np.sum(time_list)
    print('\nedit_costs:', edit_costs)
    print('\nresidual_list:', residual_list)
    print('\nedit_cost_list:', edit_cost_list)
    print('\ndistance matrix in kernel space:', dis_k_mat)
    print('\nged matrix:', ged_mat)
    print('\ntotal time:', total_time)
    print('\nnb_cost_mat:', nb_cost_mat_list[-1])
    np.savez('results/fit_distance.any_costs.gm', edit_costs=edit_costs, 
             residual_list=residual_list, edit_cost_list=edit_cost_list,
             dis_k_mat=dis_k_mat, ged_mat=ged_mat, time_list=time_list, 
             total_time=total_time, nb_cost_mat_list=nb_cost_mat_list)
    
#    # normalized distance matrices.
#    gmfile = np.load('results/fit_distance.any_costs.gm.npz')
#    edit_costs = gmfile['edit_costs']
#    residual_list = gmfile['residual_list']
#    edit_cost_list = gmfile['edit_cost_list']
#    dis_k_mat = gmfile['dis_k_mat']
#    ged_mat = gmfile['ged_mat']
#    total_time = gmfile['total_time']
##    nb_cost_mat_list = gmfile['nb_cost_mat_list']
    
    norm_dis_k_mat = normalize_distance_matrix(dis_k_mat)
    plt.imshow(norm_dis_k_mat)
    plt.colorbar()
    plt.savefig('results/norm_dis_k_mat.any_costs' + '.eps', format='eps', dpi=300)
#    plt.savefig('results/norm_dis_k_mat.any_costs' + '.png', format='png')
#    plt.show()
    plt.clf()
    
    norm_ged_mat = normalize_distance_matrix(ged_mat)
    plt.imshow(norm_ged_mat)
    plt.colorbar()
    plt.savefig('results/norm_ged_mat.any_costs' + '.eps', format='eps', dpi=300)
#    plt.savefig('results/norm_ged_mat.any_costs' + '.png', format='png')
#    plt.show()
    plt.clf()
    
    norm_diff = norm_ged_mat - norm_dis_k_mat
    plt.imshow(norm_diff)
    plt.colorbar()
    plt.savefig('results/diff_mat_norm_ged_dis_k.any_costs' + '.eps', format='eps', dpi=300)
#    plt.savefig('results/diff_mat_norm_ged_dis_k.any_costs' + '.png', format='png')
#    plt.show()
    plt.clf()
#    draw_count_bar(norm_diff)
    

def test_cs_leq_ci_plus_cr():
    """c_vs <= c_vi + c_vr, c_es <= c_ei + c_er
    """
    ds = {'name': 'monoterpenoides', 
          'dataset': '../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:10]
    gkernel = 'untilhpathkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
    itr_max = 10
    edit_costs, residual_list, edit_cost_list, dis_k_mat, ged_mat, time_list, \
        nb_cost_mat_list, coef_dk = fit_GED_to_kernel_distance(Gn, node_label, edge_label, 
                                                      gkernel, itr_max,
                                                      fitkernel='gaussian')
    total_time = np.sum(time_list)
    print('\nedit_costs:', edit_costs)
    print('\nresidual_list:', residual_list)
    print('\nedit_cost_list:', edit_cost_list)
    print('\ndistance matrix in kernel space:', dis_k_mat)
    print('\nged matrix:', ged_mat)
    print('\ntotal time:', total_time)
    print('\nnb_cost_mat:', nb_cost_mat_list[-1])
    np.savez('results/fit_distance.cs_leq_ci_plus_cr.gaussian.cost_leq_1en2.monot.elabeled.uhpkernel.gm', 
             edit_costs=edit_costs, 
             residual_list=residual_list, edit_cost_list=edit_cost_list,
             dis_k_mat=dis_k_mat, ged_mat=ged_mat, time_list=time_list, 
             total_time=total_time, nb_cost_mat_list=nb_cost_mat_list, 
             coef_dk=coef_dk)
    
#    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
#          'extra_params': {}}  # node/edge symb
#    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
##    Gn = Gn[0:10]
##    remove_edges(Gn)
#    gkernel = 'untilhpathkernel'
#    node_label = 'atom'
#    edge_label = 'bond_type'
#    itr_max = 10
#    edit_costs, residual_list, edit_cost_list, dis_k_mat, ged_mat, time_list, \
#        nb_cost_mat_list, coef_dk = fit_GED_to_kernel_distance(Gn, node_label, edge_label, 
#                                                      gkernel, itr_max)
#    total_time = np.sum(time_list)
#    print('\nedit_costs:', edit_costs)
#    print('\nresidual_list:', residual_list)
#    print('\nedit_cost_list:', edit_cost_list)
#    print('\ndistance matrix in kernel space:', dis_k_mat)
#    print('\nged matrix:', ged_mat)
#    print('\ntotal time:', total_time)
#    print('\nnb_cost_mat:', nb_cost_mat_list[-1])
#    np.savez('results/fit_distance.cs_leq_ci_plus_cr.cost_leq_1en2.mutag.elabeled.uhpkernel.gm', 
#             edit_costs=edit_costs, 
#             residual_list=residual_list, edit_cost_list=edit_cost_list,
#             dis_k_mat=dis_k_mat, ged_mat=ged_mat, time_list=time_list, 
#             total_time=total_time, nb_cost_mat_list=nb_cost_mat_list, coef_dk)
    
    
#    # normalized distance matrices.
#    gmfile = np.load('results/fit_distance.cs_leq_ci_plus_cr.cost_leq_1en2.monot.elabeled.uhpkernel.gm.npz')
#    edit_costs = gmfile['edit_costs']
#    residual_list = gmfile['residual_list']
#    edit_cost_list = gmfile['edit_cost_list']
#    dis_k_mat = gmfile['dis_k_mat']
#    ged_mat = gmfile['ged_mat']
#    total_time = gmfile['total_time']
#    nb_cost_mat_list = gmfile['nb_cost_mat_list']
#    coef_dk = gmfile['coef_dk']
    
    nb_consistent, nb_inconsistent, ratio_consistent = pairwise_substitution_consistence(dis_k_mat, ged_mat)
    print(nb_consistent, nb_inconsistent, ratio_consistent)
    
#    dis_k_sub = pairwise_substitution(dis_k_mat)
#    ged_sub = pairwise_substitution(ged_mat)    
#    np.savez('results/sub_dis_mat.cs_leq_ci_plus_cr.cost_leq_1en2.gm', 
#             dis_k_sub=dis_k_sub, ged_sub=ged_sub)
    
    
    norm_dis_k_mat = normalize_distance_matrix(dis_k_mat)
    plt.imshow(norm_dis_k_mat)
    plt.colorbar()
    plt.savefig('results/norm_dis_k_mat.cs_leq_ci_plus_cr.gaussian.cost_leq_1en2.monot.elabeled.uhpkernel' 
                + '.eps', format='eps', dpi=300)
    plt.savefig('results/norm_dis_k_mat.cs_leq_ci_plus_cr.gaussian.cost_leq_1en2.monot.elabeled.uhpkernel' 
                + '.png', format='png')
#    plt.show()
    plt.clf()
    
    norm_ged_mat = normalize_distance_matrix(ged_mat)
    plt.imshow(norm_ged_mat)
    plt.colorbar()
    plt.savefig('results/norm_ged_mat.cs_leq_ci_plus_cr.gaussian.cost_leq_1en2.monot.elabeled.uhpkernel' 
                + '.eps', format='eps', dpi=300)
    plt.savefig('results/norm_ged_mat.cs_leq_ci_plus_cr.gaussian.cost_leq_1en2.monot.elabeled.uhpkernel' 
                + '.png', format='png')
#    plt.show()
    plt.clf()
    
    norm_diff = norm_ged_mat - norm_dis_k_mat
    plt.imshow(norm_diff)
    plt.colorbar()
    plt.savefig('results/diff_mat_norm_ged_dis_k.cs_leq_ci_plus_cr.gaussian.cost_leq_1en2.monot.elabeled.uhpkernel' 
                + '.eps', format='eps', dpi=300)
    plt.savefig('results/diff_mat_norm_ged_dis_k.cs_leq_ci_plus_cr.gaussian.cost_leq_1en2.monot.elabeled.uhpkernel' 
                + '.png', format='png')
#    plt.show()
    plt.clf()
#    draw_count_bar(norm_diff)
    
    
def test_unfitted():
    """unfitted.
    """  
    from fitDistance import compute_geds
    from utils import kernel_distance_matrix
    ds = {'name': 'monoterpenoides', 
          'dataset': '../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:10]
    gkernel = 'untilhpathkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
        

#    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
#          'extra_params': {}}  # node/edge symb
#    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
##    Gn = Gn[0:10]
##    remove_edges(Gn)
#    gkernel = 'marginalizedkernel'

    dis_k_mat, _, _, _ = kernel_distance_matrix(Gn, node_label, edge_label, gkernel=gkernel)
    ged_all, ged_mat, n_edit_operations = compute_geds(Gn, [3, 3, 1, 3, 3, 1], 
            [0, 1, 2, 3, 4, 5], parallel=True)
    print('\ndistance matrix in kernel space:', dis_k_mat)
    print('\nged matrix:', ged_mat)
#    np.savez('results/fit_distance.cs_leq_ci_plus_cr.cost_leq_1en2.gm', edit_costs=edit_costs, 
#             residual_list=residual_list, edit_cost_list=edit_cost_list,
#             dis_k_mat=dis_k_mat, ged_mat=ged_mat, time_list=time_list, 
#             total_time=total_time, nb_cost_mat_list=nb_cost_mat_list) 
    
    # normalized distance matrices.
#    gmfile = np.load('results/fit_distance.cs_leq_ci_plus_cr.cost_leq_1en3.gm.npz')
#    edit_costs = gmfile['edit_costs']
#    residual_list = gmfile['residual_list']
#    edit_cost_list = gmfile['edit_cost_list']
#    dis_k_mat = gmfile['dis_k_mat']
#    ged_mat = gmfile['ged_mat']
#    total_time = gmfile['total_time']
#    nb_cost_mat_list = gmfile['nb_cost_mat_list']
    
    nb_consistent, nb_inconsistent, ratio_consistent = pairwise_substitution_consistence(dis_k_mat, ged_mat)
    print(nb_consistent, nb_inconsistent, ratio_consistent)
    
    norm_dis_k_mat = normalize_distance_matrix(dis_k_mat)
    plt.imshow(norm_dis_k_mat)
    plt.colorbar()
    plt.savefig('results/norm_dis_k_mat.unfitted.MUTAG' + '.eps', format='eps', dpi=300)
    plt.savefig('results/norm_dis_k_mat.unfitted.MUTAG' + '.png', format='png')
#    plt.show()
    plt.clf()
    
    norm_ged_mat = normalize_distance_matrix(ged_mat)
    plt.imshow(norm_ged_mat)
    plt.colorbar()
    plt.savefig('results/norm_ged_mat.unfitted.MUTAG' + '.eps', format='eps', dpi=300)
    plt.savefig('results/norm_ged_mat.unfitted.MUTAG' + '.png', format='png')
#    plt.show()
    plt.clf()
    
    norm_diff = norm_ged_mat - norm_dis_k_mat
    plt.imshow(norm_diff)
    plt.colorbar()
    plt.savefig('results/diff_mat_norm_ged_dis_k.unfitted.MUTAG' + '.eps', format='eps', dpi=300)
    plt.savefig('results/diff_mat_norm_ged_dis_k.unfitted.MUTAG' + '.png', format='png')
#    plt.show()
    plt.clf()
    draw_count_bar(norm_diff)
    
    
def pairwise_substitution_consistence(mat1, mat2):
    """
    """
    nb_consistent = 0
    nb_inconsistent = 0
    # the matrix is considered symmetric.
    upper_tri1 = mat1[np.triu_indices_from(mat1)]
    upper_tri2 = mat2[np.tril_indices_from(mat2)]
    for i in tqdm(range(len(upper_tri1)), desc='computing consistence', file=sys.stdout):
        for j in range(i, len(upper_tri1)):
            if np.sign(upper_tri1[i] - upper_tri1[j]) == np.sign(upper_tri2[i] - upper_tri2[j]):
                nb_consistent += 1
            else:
                nb_inconsistent += 1
    return nb_consistent, nb_inconsistent, nb_consistent / (nb_consistent + nb_inconsistent)


def pairwise_substitution(mat):
    # the matrix is considered symmetric.
    upper_tri = mat[np.triu_indices_from(mat)]
    sub_list = []
    for i in tqdm(range(len(upper_tri)), desc='computing', file=sys.stdout):
        for j in range(i, len(upper_tri)):
            sub_list.append(upper_tri[i] - upper_tri[j])
    return sub_list
    
    
def draw_count_bar(norm_diff):
    import pandas
    from collections import Counter, OrderedDict
    norm_diff_cnt = norm_diff.flatten()
    norm_diff_cnt = norm_diff_cnt * 10
    norm_diff_cnt = np.floor(norm_diff_cnt)
    norm_diff_cnt = Counter(norm_diff_cnt)
    norm_diff_cnt = OrderedDict(sorted(norm_diff_cnt.items()))
    df = pandas.DataFrame.from_dict(norm_diff_cnt, orient='index')
    df.plot(kind='bar')
    
    
if __name__ == '__main__':
#    test_anycosts()
    test_cs_leq_ci_plus_cr()
#    test_unfitted()
    
#    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
#    xx = pairwise_substitution(x)