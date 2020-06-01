#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:37:57 2020

@author: ljia
"""
import multiprocessing
import numpy as np
import networkx as nx
import os
from gklearn.preimage import RandomPreimageGenerator
from gklearn.utils import Dataset


dir_root = '../results/xp_random_preimage_generation/'


def xp_random_preimage_generation():
	"""
	Experiment similar to the one in Bakir's paper. A test to check if RandomPreimageGenerator class works correctly.

	Returns
	-------
	None.

	"""
	alpha1_list = np.linspace(0, 1, 11)
	k_dis_datasets = []
	k_dis_preimages = []
	preimages = []
	bests_from_dataset = []
	for alpha1 in alpha1_list:
		print('alpha1 =', alpha1, ':\n')
		# set parameters.
		ds_name = 'MUTAG'
		rpg_options = {'k': 5,
					   'r_max': 10, #
					   'l': 500,
					   'alphas': None,
					   'parallel': True,
					   'verbose': 2}
		kernel_options = {'name': 'PathUpToH',
						  'depth': 2, #
						  'k_func': 'MinMax', #
						  'compute_method': 'trie',
	 					  'parallel': 'imap_unordered', 
	                      # 'parallel': None, 
						  'n_jobs': multiprocessing.cpu_count(),
						  'normalize': True,
						  'verbose': 0}
		edge_required = True
		irrelevant_labels = {'edge_labels': ['label_0']}
		cut_range = None
		
		# create/get Gram matrix.
		dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/'
		if not os.path.exists(dir_save):
			os.makedirs(dir_save)
		gm_fname = dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm.npz'
		gmfile_exist = os.path.isfile(os.path.abspath(gm_fname))
		if gmfile_exist:
			gmfile = np.load(gm_fname, allow_pickle=True) # @todo: may not be safe.
			gram_matrix_unnorm = gmfile['gram_matrix_unnorm']
			time_precompute_gm = gmfile['run_time']
	
		# 1. get dataset.
		print('1. getting dataset...')
		dataset_all = Dataset()
		dataset_all.load_predefined_dataset(ds_name)
		dataset_all.trim_dataset(edge_required=edge_required)
		if irrelevant_labels is not None:
			dataset_all.remove_labels(**irrelevant_labels)
		if cut_range is not None:
			dataset_all.cut_graphs(cut_range)
		
		# add two "random" graphs.
		g1 = nx.Graph()
		g1.add_nodes_from(range(0, 16), label_0='0')
		g1.add_nodes_from(range(16, 25), label_0='1')
		g1.add_node(25, label_0='2')
		g1.add_nodes_from([26, 27], label_0='3')
		g1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (5, 0), (4, 9), (12, 3), (10, 13), (13, 14), (14, 15), (15, 8), (0, 16), (1, 17), (2, 18), (12, 19), (11, 20), (13, 21), (15, 22), (7, 23), (6, 24), (14, 25), (25, 26), (25, 27)])
		g2 = nx.Graph()
		g2.add_nodes_from(range(0, 12), label_0='0')
		g2.add_nodes_from(range(12, 19), label_0='1')
		g2.add_nodes_from([19, 20, 21], label_0='2')
		g2.add_nodes_from([22, 23], label_0='3')
		g2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 19), (19, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 20), (20, 7), (5, 0), (4, 8), (0, 12), (1, 13), (2, 14), (9, 15), (10, 16), (11, 17), (6, 18), (3, 21), (21, 22), (21, 23)])
		dataset_all.load_graphs([g1, g2] + dataset_all.graphs, targets=None)
		
		# 2. initialize rpg and setting parameters.
		print('2. initializing rpg and setting parameters...')
		nb_graphs = len(dataset_all.graphs) - 2
		rpg_options['alphas'] = [alpha1, 1 - alpha1] + [0] * nb_graphs
		if gmfile_exist:
			rpg_options['gram_matrix_unnorm'] = gram_matrix_unnorm
			rpg_options['runtime_precompute_gm'] = time_precompute_gm
		rpg = RandomPreimageGenerator()
		rpg.dataset = dataset_all
		rpg.set_options(**rpg_options.copy())
		rpg.kernel_options = kernel_options.copy()
	
		# 3. compute preimage.
		print('3. computing preimage...')
		rpg.run()
		results = rpg.get_results()
		k_dis_datasets.append(results['k_dis_dataset'])
		k_dis_preimages.append(results['k_dis_preimage'])
		bests_from_dataset.append(rpg.best_from_dataset)
		preimages.append(rpg.preimage)
		
		# 4. save results.
		# write Gram matrices to file.
		if not gmfile_exist:
			np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm=rpg.gram_matrix_unnorm, run_time=results['runtime_precompute_gm'])

	print('\ncomplete.\n')
	
	return k_dis_datasets, k_dis_preimages, bests_from_dataset, preimages


if __name__ == '__main__':
	k_dis_datasets, k_dis_preimages, bests_from_dataset, preimages = xp_random_preimage_generation()