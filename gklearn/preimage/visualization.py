#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:16:23 2019

@author: ljia
"""
import numpy as np
from sklearn.manifold import TSNE, Isomap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from tqdm import tqdm

from gklearn.utils.graphfiles import loadDataset, loadGXL
from gklearn.preimage.utils import kernel_distance_matrix, compute_kernel, dis_gstar, get_same_item_indices


def visualize_graph_dataset(dis_measure, visual_method, draw_figure, 
                            draw_params={}, dis_mat=None, Gn=None, 
                            median_set=None):
    
    
    def draw_zoomed_axes(Gn_embedded, ax):
        margin = 0.01
        if dis_measure == 'graph-kernel':
            index = -2
        elif dis_measure == 'ged':
            index = -1
        x1 = np.min(Gn_embedded[median_set + [index], 0]) - margin * np.max(Gn_embedded)
        x2 = np.max(Gn_embedded[median_set + [index], 0]) + margin * np.max(Gn_embedded)
        y1 = np.min(Gn_embedded[median_set + [index], 1]) - margin * np.max(Gn_embedded)
        y2 = np.max(Gn_embedded[median_set + [index], 1]) + margin * np.max(Gn_embedded)
        if (x1 < 0 and y1 < 0) or ((x1 > 0 and y1 > 0)):
            loc = 2
        else:
            loc = 3
        axins = zoomed_inset_axes(ax, 4, loc=loc) # zoom-factor: 2.5, location: upper-left
        draw_figure(axins, Gn_embedded, dis_measure=dis_measure, 
                    median_set=median_set, **draw_params)
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        loc1 = 1 if loc == 2 else 3
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")  
        
        
    if dis_mat is None:
        if dis_measure == 'graph-kernel':
            gkernel = 'untilhpathkernel'
            node_label = 'atom'
            edge_label = 'bond_type'
            dis_mat, _, _, _ = kernel_distance_matrix(Gn, node_label, edge_label, 
                                                      Kmatrix=None, gkernel=gkernel)
        elif dis_measure == 'ged':
            pass
        
    if visual_method == 'tsne':
        Gn_embedded = TSNE(n_components=2, metric='precomputed').fit_transform(dis_mat)
    elif visual_method == 'isomap':
        Gn_embedded = Isomap(n_components=2, metric='precomputed').fit_transform(dis_mat)
    print(Gn_embedded.shape)
    fig, ax = plt.subplots()
    draw_figure(plt, Gn_embedded, dis_measure=dis_measure, legend=True, 
                median_set=median_set, **draw_params)        
#    draw_zoomed_axes(Gn_embedded, ax)
    plt.show()
    plt.clf()
    
    return


def draw_figure(ax, Gn_embedded, dis_measure=None, y_idx=None, legend=False,
                median_set=None):
    from matplotlib import colors as mcolors
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
#    colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
#              '#c6dbef', '#deebf7']
#    for i, values in enumerate(y_idx.values()):
#        for item in values:
##            ax.scatter(Gn_embedded[item,0], Gn_embedded[item,1], c=colors[i]) # , c='b')        
#            ax.scatter(Gn_embedded[item,0], Gn_embedded[item,1], c='b')
#    ax.scatter(Gn_embedded[:,0], Gn_embedded[:,1], c='b')        
    h1 = ax.scatter(Gn_embedded[median_set, 0], Gn_embedded[median_set, 1], c='b')
    if dis_measure == 'graph-kernel':
        h2 = ax.scatter(Gn_embedded[-1, 0], Gn_embedded[-1, 1], c='darkorchid') # \psi
        h3 = ax.scatter(Gn_embedded[-2, 0], Gn_embedded[-2, 1], c='gold') # gen median
        h4 = ax.scatter(Gn_embedded[-3, 0], Gn_embedded[-3, 1], c='r') #c='g', marker='+') # set median
    elif dis_measure == 'ged':
        h3 = ax.scatter(Gn_embedded[-1, 0], Gn_embedded[-1, 1], c='gold') # gen median
        h4 = ax.scatter(Gn_embedded[-2, 0], Gn_embedded[-2, 1], c='r') #c='g', marker='+') # set median        
    if legend:
#    fig.subplots_adjust(bottom=0.17)
        if dis_measure == 'graph-kernel':
            ax.legend([h1, h2, h3, h4], 
                      ['k closest graphs', 'true median', 'gen median', 'set median'])
        elif dis_measure == 'ged':       
            ax.legend([h1, h3, h4], ['k closest graphs', 'gen median', 'set median'])
#    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
#    plt.savefig('symbolic_and_non_comparison_vertical_short.eps', format='eps', dpi=300, transparent=True,
#            bbox_inches='tight')
#    plt.show()
            
    
###############################################################################
    
def visualize_distances_in_kernel():
    
    ds = {'name': 'monoterpenoides', 
          'dataset': '../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
    fname_medians = 'expert.treelet'
    # add set median.
    fname_sm = 'results/test_k_closest_graphs/set_median.' + fname_medians + '.gxl'
    set_median = loadGXL(fname_sm)
    Gn.append(set_median)
    # add generalized median (estimated pre-image.)
    fname_gm = 'results/test_k_closest_graphs/gen_median.' + fname_medians + '.gxl'
    gen_median = loadGXL(fname_gm)
    Gn.append(gen_median)
    
    # compute distance matrix
    median_set = [22, 29, 54, 74]
    gkernel = 'treeletkernel'
    node_label = 'atom'
    edge_label = 'bond_type'
    Gn_median_set = [Gn[i].copy() for i in median_set]
    Kmatrix_median = compute_kernel(Gn + Gn_median_set, gkernel, node_label, 
                                    edge_label, True)
    Kmatrix = Kmatrix_median[0:len(Gn), 0:len(Gn)]
    dis_mat, _, _, _ = kernel_distance_matrix(Gn, node_label, edge_label, 
                                              Kmatrix=Kmatrix, gkernel=gkernel)
    print('average distances: ', np.mean(np.mean(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))
    print('min distances: ', np.min(np.min(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))
    print('max distances: ', np.max(np.max(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))

    # add distances for the image of exact median \psi.
    dis_k_median_list = []
    for idx, g in enumerate(Gn):
        dis_k_median_list.append(dis_gstar(idx, range(len(Gn), len(Gn) + len(Gn_median_set)), 
                                           [1 / len(Gn_median_set)] * len(Gn_median_set),
                                           Kmatrix_median, withterm3=False))
    dis_mat_median = np.zeros((len(Gn) + 1, len(Gn) + 1))
    for i in range(len(Gn)):
        for j in range(i, len(Gn)):
            dis_mat_median[i, j] = dis_mat[i, j]
            dis_mat_median[j, i] = dis_mat_median[i, j]
    for i in range(len(Gn)):
        dis_mat_median[i, -1] = dis_k_median_list[i]
        dis_mat_median[-1, i] = dis_k_median_list[i]
    
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    
    # visualization.
#    visualize_graph_dataset('graph-kernel', 'tsne', Gn)
#    visualize_graph_dataset('graph-kernel', 'tsne', draw_figure, 
#                            draw_params={'y_idx': y_idx}, dis_mat=dis_mat_median)
    visualize_graph_dataset('graph-kernel', 'tsne', draw_figure, 
                            draw_params={'y_idx': y_idx}, dis_mat=dis_mat_median,
                            median_set=median_set)
        
    
def visualize_distances_in_ged():
    from gklearn.preimage.fitDistance import compute_geds
    from gklearn.preimage.ged import GED
    ds = {'name': 'monoterpenoides', 
          'dataset': '../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
    # add set median.
    fname_medians = 'expert.treelet'
    fname_sm = 'preimage/results/test_k_closest_graphs/set_median.' + fname_medians + '.gxl'
    set_median = loadGXL(fname_sm)
    Gn.append(set_median)
    # add generalized median (estimated pre-image.)
    fname_gm = 'preimage/results/test_k_closest_graphs/gen_median.' + fname_medians + '.gxl'
    gen_median = loadGXL(fname_gm)
    Gn.append(gen_median)
    
    # compute/load ged matrix.
#    # compute.
##    k = 4
##    edit_costs = [0.16229209837639536, 0.06612870523413916, 0.04030113378793905, 0.20723547009415202, 0.3338607220394598, 0.27054392518077297]
#    edit_costs = [3, 3, 1, 3, 3, 1]
##    edit_costs = [7, 3, 5, 9, 2, 6]
#    algo_options = '--threads 8 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
#    params_ged = {'lib': 'gedlibpy', 'cost': 'CONSTANT', 'method': 'IPFP', 
#                'algo_options': algo_options, 'stabilizer': None, 
#                'edit_cost_constant': edit_costs}    
#    _, ged_mat, _ = compute_geds(Gn, params_ged=params_ged, parallel=True)
#    np.savez('results/test_k_closest_graphs/ged_mat.' + fname_medians + '.with_medians.gm', ged_mat=ged_mat)
    # load from file.
    gmfile = np.load('results/test_k_closest_graphs/ged_mat.' + fname_medians + '.with_medians.gm.npz')
    ged_mat = gmfile['ged_mat']
#    # change medians.
#    edit_costs = [3, 3, 1, 3, 3, 1]
#    algo_options = '--threads 8 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
#    params_ged = {'lib': 'gedlibpy', 'cost': 'CONSTANT', 'method': 'IPFP', 
#                'algo_options': algo_options, 'stabilizer': None, 
#                'edit_cost_constant': edit_costs}
#    for idx in tqdm(range(len(Gn) - 2), desc='computing GEDs', file=sys.stdout):
#        dis, _, _ = GED(Gn[idx], set_median, **params_ged)
#        ged_mat[idx, -2] = dis
#        ged_mat[-2, idx] = dis
#        dis, _, _ = GED(Gn[idx], gen_median, **params_ged)
#        ged_mat[idx, -1] = dis
#        ged_mat[-1, idx] = dis
#    np.savez('results/test_k_closest_graphs/ged_mat.' + fname_medians + '.with_medians.gm', 
#             ged_mat=ged_mat)

    
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)

    # visualization.
    median_set = [22, 29, 54, 74]
    visualize_graph_dataset('ged', 'tsne', draw_figure, 
                            draw_params={'y_idx': y_idx}, dis_mat=ged_mat,
                            median_set=median_set)
    
###############################################################################
    
    
def visualize_distances_in_kernel_monoterpenoides():
    import os

    ds = {'dataset': '../datasets/monoterpenoides/dataset_10+.ds',
          'graph_dir': os.path.dirname(os.path.realpath(__file__))  + '../../datasets/monoterpenoides/'}  # node/edge symb
    Gn_original, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
    
    # compute distance matrix
#    median_set = [22, 29, 54, 74]
    gkernel = 'treeletkernel'
    fit_method = 'expert'
    node_label = 'atom'
    edge_label = 'bond_type'
    ds_name = 'monoterpenoides'
    fname_medians = fit_method + '.' + gkernel
    dir_output = 'results/xp_monoterpenoides/'
    repeat = 0
    
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    for i, (y, values) in enumerate(y_idx.items()):
        print('\ny =', y)
        k = len(values)
        
        Gn = [Gn_original[g].copy() for g in values]
        # add set median.
        fname_sm = dir_output + 'medians/' + str(int(y)) + '/set_median.k' + str(int(k)) \
            + '.y' + str(int(y)) + '.repeat' + str(repeat) + '.gxl'
        set_median = loadGXL(fname_sm)
        Gn.append(set_median)
        # add generalized median (estimated pre-image.)
        fname_gm = dir_output + 'medians/' + str(int(y)) + '/gen_median.k' + str(int(k)) \
            + '.y' + str(int(y)) + '.repeat' + str(repeat) + '.gxl'
        gen_median = loadGXL(fname_gm)
        Gn.append(gen_median)
    
        # compute distance matrix
        median_set = range(0, len(values))
    
        Gn_median_set = [Gn[i].copy() for i in median_set]
        Kmatrix_median = compute_kernel(Gn + Gn_median_set, gkernel, node_label, 
                                        edge_label, False)
        Kmatrix = Kmatrix_median[0:len(Gn), 0:len(Gn)]
        dis_mat, _, _, _ = kernel_distance_matrix(Gn, node_label, edge_label, 
                                                  Kmatrix=Kmatrix, gkernel=gkernel)
        print('average distances: ', np.mean(np.mean(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))
        print('min distances: ', np.min(np.min(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))
        print('max distances: ', np.max(np.max(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))

        # add distances for the image of exact median \psi.
        dis_k_median_list = []
        for idx, g in enumerate(Gn):
            dis_k_median_list.append(dis_gstar(idx, range(len(Gn), len(Gn) + len(Gn_median_set)), 
                                               [1 / len(Gn_median_set)] * len(Gn_median_set),
                                               Kmatrix_median, withterm3=False))
        dis_mat_median = np.zeros((len(Gn) + 1, len(Gn) + 1))
        for i in range(len(Gn)):
            for j in range(i, len(Gn)):
                dis_mat_median[i, j] = dis_mat[i, j]
                dis_mat_median[j, i] = dis_mat_median[i, j]
        for i in range(len(Gn)):
            dis_mat_median[i, -1] = dis_k_median_list[i]
            dis_mat_median[-1, i] = dis_k_median_list[i]
            
    
        # visualization.
#    visualize_graph_dataset('graph-kernel', 'tsne', Gn)
#    visualize_graph_dataset('graph-kernel', 'tsne', draw_figure, 
#                            draw_params={'y_idx': y_idx}, dis_mat=dis_mat_median)
        visualize_graph_dataset('graph-kernel', 'tsne', draw_figure, 
                                draw_params={'y_idx': y_idx}, dis_mat=dis_mat_median,
                                median_set=median_set)
        
    
def visualize_distances_in_ged_monoterpenoides():
    from gklearn.preimage.fitDistance import compute_geds
    from gklearn.preimage.ged import GED
    import os
    
    ds = {'dataset': '../datasets/monoterpenoides/dataset_10+.ds',
          'graph_dir': os.path.dirname(os.path.realpath(__file__)) + '../../datasets/monoterpenoides/'}  # node/edge symb
    Gn_original, y_all = loadDataset(ds['dataset'])
#    Gn = Gn[0:50]
    
    # compute distance matrix
#    median_set = [22, 29, 54, 74]
    gkernel = 'treeletkernel'
    fit_method = 'expert'
    ds_name = 'monoterpenoides'
    fname_medians = fit_method + '.' + gkernel
    dir_output = 'results/xp_monoterpenoides/'
    repeat = 0
#    edit_costs = [0.16229209837639536, 0.06612870523413916, 0.04030113378793905, 0.20723547009415202, 0.3338607220394598, 0.27054392518077297]
    edit_costs = [3, 3, 1, 3, 3, 1]
#    edit_costs = [7, 3, 5, 9, 2, 6]
    
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    for i, (y, values) in enumerate(y_idx.items()):
        print('\ny =', y)
        k = len(values)
        
        Gn = [Gn_original[g].copy() for g in values]
        # add set median.
        fname_sm = dir_output + 'medians/' + str(int(y)) + '/set_median.k' + str(int(k)) \
            + '.y' + str(int(y)) + '.repeat' + str(repeat) + '.gxl'
        set_median = loadGXL(fname_sm)
        Gn.append(set_median)
        # add generalized median (estimated pre-image.)
        fname_gm = dir_output + 'medians/' + str(int(y)) + '/gen_median.k' + str(int(k)) \
            + '.y' + str(int(y)) + '.repeat' + str(repeat) + '.gxl'
        gen_median = loadGXL(fname_gm)
        Gn.append(gen_median)
    
    
        # compute/load ged matrix.
        # compute.
        algo_options = '--threads 1 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
        params_ged = {'dataset': ds_name, 'lib': 'gedlibpy', 'cost': 'CONSTANT', 
                      'method': 'IPFP', 'algo_options': algo_options, 
                      'stabilizer': None, 'edit_cost_constant': edit_costs}    
        _, ged_mat, _ = compute_geds(Gn, params_ged=params_ged, parallel=True)
        np.savez(dir_output + 'ged_mat.' + fname_medians + '.y' + str(int(y)) \
            + '.with_medians.gm', ged_mat=ged_mat)
#        # load from file.
#        gmfile = np.load('dir_output + 'ged_mat.' + fname_medians + '.y' + str(int(y)) + '.with_medians.gm.npz')
#        ged_mat = gmfile['ged_mat']
#        # change medians.
#        algo_options = '--threads 1 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
#        params_ged = {'lib': 'gedlibpy', 'cost': 'CONSTANT', 'method': 'IPFP', 
#                    'algo_options': algo_options, 'stabilizer': None, 
#                    'edit_cost_constant': edit_costs}
#        for idx in tqdm(range(len(Gn) - 2), desc='computing GEDs', file=sys.stdout):
#            dis, _, _ = GED(Gn[idx], set_median, **params_ged)
#            ged_mat[idx, -2] = dis
#            ged_mat[-2, idx] = dis
#            dis, _, _ = GED(Gn[idx], gen_median, **params_ged)
#            ged_mat[idx, -1] = dis
#            ged_mat[-1, idx] = dis
#        np.savez(dir_output + 'ged_mat.' + fname_medians + '.y' + str(int(y)) + '.with_medians.gm', 
#                 ged_mat=ged_mat)

        # visualization.
        median_set = range(0, len(values))
        visualize_graph_dataset('ged', 'tsne', draw_figure, 
                                draw_params={'y_idx': y_idx}, dis_mat=ged_mat,
                                median_set=median_set)
        
        
###############################################################################
    
    
def visualize_distances_in_kernel_letter_h():
    
    ds = {'dataset': 'cpp_ext/data/collections/Letter.xml',
          'graph_dir': os.path.dirname(os.path.realpath(__file__)) + '/cpp_ext/data/datasets/Letter/HIGH/'}  # node/edge symb
    Gn_original, y_all = loadDataset(ds['dataset'], extra_params=ds['graph_dir'])
#    Gn = Gn[0:50]
    
    # compute distance matrix
#    median_set = [22, 29, 54, 74]
    gkernel = 'structuralspkernel'
    fit_method = 'expert'
    node_label = None
    edge_label = None
    ds_name = 'letter-h'
    fname_medians = fit_method + '.' + gkernel
    dir_output = 'results/xp_letter_h/'
    k = 150
    repeat = 0
    
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    for i, (y, values) in enumerate(y_idx.items()):
        print('\ny =', y)
        
        Gn = [Gn_original[g].copy() for g in values]
        # add set median.
        fname_sm = dir_output + 'medians/' + y + '/set_median.k' + str(int(k)) \
            + '.y' + y + '.repeat' + str(repeat) + '.gxl'
        set_median = loadGXL(fname_sm)
        Gn.append(set_median)
        # add generalized median (estimated pre-image.)
        fname_gm = dir_output + 'medians/' + y + '/gen_median.k' + str(int(k)) \
            + '.y' + y + '.repeat' + str(repeat) + '.gxl'
        gen_median = loadGXL(fname_gm)
        Gn.append(gen_median)
    
        # compute distance matrix
        median_set = range(0, len(values))
    
        Gn_median_set = [Gn[i].copy() for i in median_set]
        Kmatrix_median = compute_kernel(Gn + Gn_median_set, gkernel, node_label, 
                                        edge_label, False)
        Kmatrix = Kmatrix_median[0:len(Gn), 0:len(Gn)]
        dis_mat, _, _, _ = kernel_distance_matrix(Gn, node_label, edge_label, 
                                                  Kmatrix=Kmatrix, gkernel=gkernel)
        print('average distances: ', np.mean(np.mean(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))
        print('min distances: ', np.min(np.min(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))
        print('max distances: ', np.max(np.max(dis_mat[0:len(Gn)-2, 0:len(Gn)-2])))

        # add distances for the image of exact median \psi.
        dis_k_median_list = []
        for idx, g in enumerate(Gn):
            dis_k_median_list.append(dis_gstar(idx, range(len(Gn), len(Gn) + len(Gn_median_set)), 
                                               [1 / len(Gn_median_set)] * len(Gn_median_set),
                                               Kmatrix_median, withterm3=False))
        dis_mat_median = np.zeros((len(Gn) + 1, len(Gn) + 1))
        for i in range(len(Gn)):
            for j in range(i, len(Gn)):
                dis_mat_median[i, j] = dis_mat[i, j]
                dis_mat_median[j, i] = dis_mat_median[i, j]
        for i in range(len(Gn)):
            dis_mat_median[i, -1] = dis_k_median_list[i]
            dis_mat_median[-1, i] = dis_k_median_list[i]
            
    
        # visualization.
#    visualize_graph_dataset('graph-kernel', 'tsne', Gn)
#    visualize_graph_dataset('graph-kernel', 'tsne', draw_figure, 
#                            draw_params={'y_idx': y_idx}, dis_mat=dis_mat_median)
        visualize_graph_dataset('graph-kernel', 'tsne', draw_figure, 
                                draw_params={'y_idx': y_idx}, dis_mat=dis_mat_median,
                                median_set=median_set)
        
    
def visualize_distances_in_ged_letter_h():
    from fitDistance import compute_geds
    from preimage.test_k_closest_graphs import reform_attributes
    
    ds = {'dataset': 'cpp_ext/data/collections/Letter.xml',
          'graph_dir': os.path.dirname(os.path.realpath(__file__)) + '/cpp_ext/data/datasets/Letter/HIGH/'}  # node/edge symb
    Gn_original, y_all = loadDataset(ds['dataset'], extra_params=ds['graph_dir'])
#    Gn = Gn[0:50]
    
    # compute distance matrix
#    median_set = [22, 29, 54, 74]
    gkernel = 'structuralspkernel'
    fit_method = 'expert'
    ds_name = 'letter-h'
    fname_medians = fit_method + '.' + gkernel
    dir_output = 'results/xp_letter_h/'
    k = 150
    repeat = 0
#    edit_costs = [0.16229209837639536, 0.06612870523413916, 0.04030113378793905, 0.20723547009415202, 0.3338607220394598, 0.27054392518077297]
    edit_costs = [3, 3, 1, 3, 3, 1]
#    edit_costs = [7, 3, 5, 9, 2, 6]
    
    # get indices by classes.
    y_idx = get_same_item_indices(y_all)
    for i, (y, values) in enumerate(y_idx.items()):
        print('\ny =', y)
        
        Gn = [Gn_original[g].copy() for g in values]
        # add set median.
        fname_sm = dir_output + 'medians/' + y + '/set_median.k' + str(int(k)) \
            + '.y' + y + '.repeat' + str(repeat) + '.gxl'
        set_median = loadGXL(fname_sm)
        Gn.append(set_median)
        # add generalized median (estimated pre-image.)
        fname_gm = dir_output + 'medians/' + y + '/gen_median.k' + str(int(k)) \
            + '.y' + y + '.repeat' + str(repeat) + '.gxl'
        gen_median = loadGXL(fname_gm)
        Gn.append(gen_median)
    
    
        # compute/load ged matrix.
        # compute.
        algo_options = '--threads 1 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
        params_ged = {'dataset': 'Letter', 'lib': 'gedlibpy', 'cost': 'CONSTANT', 
                      'method': 'IPFP', 'algo_options': algo_options, 
                      'stabilizer': None, 'edit_cost_constant': edit_costs}    
        for g in Gn:
            reform_attributes(g)
        _, ged_mat, _ = compute_geds(Gn, params_ged=params_ged, parallel=True)
        np.savez(dir_output + 'ged_mat.' + fname_medians + '.y' + y + '.with_medians.gm', ged_mat=ged_mat)
#        # load from file.
#        gmfile = np.load('dir_output + 'ged_mat.' + fname_medians + '.y' + y + '.with_medians.gm.npz')
#        ged_mat = gmfile['ged_mat']
#        # change medians.
#        algo_options = '--threads 1 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'
#        params_ged = {'lib': 'gedlibpy', 'cost': 'CONSTANT', 'method': 'IPFP', 
#                    'algo_options': algo_options, 'stabilizer': None, 
#                    'edit_cost_constant': edit_costs}
#        for idx in tqdm(range(len(Gn) - 2), desc='computing GEDs', file=sys.stdout):
#            dis, _, _ = GED(Gn[idx], set_median, **params_ged)
#            ged_mat[idx, -2] = dis
#            ged_mat[-2, idx] = dis
#            dis, _, _ = GED(Gn[idx], gen_median, **params_ged)
#            ged_mat[idx, -1] = dis
#            ged_mat[-1, idx] = dis
#        np.savez(dir_output + 'ged_mat.' + fname_medians + '.y' + y + '.with_medians.gm', 
#                 ged_mat=ged_mat)

    
        # visualization.
        median_set = range(0, len(values))
        visualize_graph_dataset('ged', 'tsne', draw_figure, 
                                draw_params={'y_idx': y_idx}, dis_mat=ged_mat,
                                median_set=median_set)


if __name__ == '__main__':
    visualize_distances_in_kernel_letter_h()
#    visualize_distances_in_ged_letter_h()
#    visualize_distances_in_kernel_monoterpenoides()
#    visualize_distances_in_kernel_monoterpenoides()
#    visualize_distances_in_kernel()
#    visualize_distances_in_ged()
    
    
    
    
    
    
    
#def draw_figure_dis_k(ax, Gn_embedded, y_idx=None, legend=False):
#    from matplotlib import colors as mcolors
#    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
##    colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
##              '#c6dbef', '#deebf7']
#    for i, values in enumerate(y_idx.values()):
#        for item in values:
##            ax.scatter(Gn_embedded[item,0], Gn_embedded[item,1], c=colors[i]) # , c='b')        
#            ax.scatter(Gn_embedded[item,0], Gn_embedded[item,1], c='b')        
#    h1 = ax.scatter(Gn_embedded[[12, 13, 22, 29], 0], Gn_embedded[[12, 13, 22, 29], 1], c='r')
#    h2 = ax.scatter(Gn_embedded[-1, 0], Gn_embedded[-1, 1], c='darkorchid') # \psi
#    h3 = ax.scatter(Gn_embedded[-2, 0], Gn_embedded[-2, 1], c='gold') # gen median
#    h4 = ax.scatter(Gn_embedded[-3, 0], Gn_embedded[-3, 1], c='r', marker='+') # set median
#    if legend:
##    fig.subplots_adjust(bottom=0.17)
#        ax.legend([h1, h2, h3, h4], ['k clostest graphs', 'true median', 'gen median', 'set median'])
##    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
##    plt.savefig('symbolic_and_non_comparison_vertical_short.eps', format='eps', dpi=300, transparent=True,
##            bbox_inches='tight')
##    plt.show()
    
    
    
#def draw_figure_ged(ax, Gn_embedded, y_idx=None, legend=False):
#    from matplotlib import colors as mcolors
#    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
##    colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
##              '#c6dbef', '#deebf7']
#    for i, values in enumerate(y_idx.values()):
#        for item in values:
##            ax.scatter(Gn_embedded[item,0], Gn_embedded[item,1], c=colors[i]) # , c='b')        
#            ax.scatter(Gn_embedded[item,0], Gn_embedded[item,1], c='b')        
#    h1 = ax.scatter(Gn_embedded[[12, 13, 22, 29], 0], Gn_embedded[[12, 13, 22, 29], 1], c='r')
##    h2 = ax.scatter(Gn_embedded[-1, 0], Gn_embedded[-1, 1], c='darkorchid') # \psi
#    h3 = ax.scatter(Gn_embedded[-1, 0], Gn_embedded[-1, 1], c='gold') # gen median
#    h4 = ax.scatter(Gn_embedded[-2, 0], Gn_embedded[-2, 1], c='r', marker='+') # set median
#    if legend:
##    fig.subplots_adjust(bottom=0.17)
#        ax.legend([h1, h3, h4], ['k clostest graphs', 'gen median', 'set median'])
##    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
##    plt.savefig('symbolic_and_non_comparison_vertical_short.eps', format='eps', dpi=300, transparent=True,
##            bbox_inches='tight')
##    plt.show()