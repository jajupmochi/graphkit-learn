#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:15:11 2020

@author: ljia
"""
import functools
import multiprocessing
import os
import sys
import logging
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from gklearn.preimage import kernel_knn_cv

dir_root = '../results/xp_1nn.init1.no_triangle_rule.allow_zeros/'
num_random = 10
initial_solutions = 1
triangle_rule = False
allow_zeros = True
update_order = False
test_sizes = [0.9, 0.7] # , 0.5, 0.3, 0.1]

def xp_knn_1_1():
	for test_size in test_sizes:
		ds_name = 'Letter-high'
		knn_options = {'n_neighbors': 1,
					   'n_splits': 30,
					   'test_size': test_size,
					   'verbose': True}
		mpg_options = {'fit_method': 'k-graphs',
					   'init_ecc': [0.675, 0.675, 0.75, 0.425, 0.425],
					   'ds_name': ds_name,
					   'parallel': True, # False
					   'time_limit_in_sec': 0,
					   'max_itrs': 100,
					   'max_itrs_without_update': 3,
					   'epsilon_residual': 0.01,
					   'epsilon_ec': 0.1,
					   'allow_zeros': allow_zeros,
					   'triangle_rule': triangle_rule,
					   'verbose': 1}
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		kernel_options = {'name': 'StructuralSP',
						  'edge_weight': None,
						  'node_kernels': sub_kernels,
						  'edge_kernels': sub_kernels, 
						  'compute_method': 'naive',
						  'parallel': 'imap_unordered', 
# 						  'parallel': None, 
						  'n_jobs': multiprocessing.cpu_count(),
						  'normalize': True,
						  'verbose': 0}
		ged_options = {'method': 'IPFP',
					   'initialization_method': 'RANDOM', # 'NODE'
					   'initial_solutions': initial_solutions, # 1
					   'edit_cost': 'LETTER2',
					   'attr_distance': 'euclidean',
					   'ratio_runs_from_initial_solutions': 1,
					   'threads': multiprocessing.cpu_count(),
					   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
		mge_options = {'init_type': 'MEDOID',
					   'random_inits': 10,
					   'time_limit': 0,
					   'verbose': 1,
					   'update_order': update_order,
					   'randomness': 'REAL',
					   'refine': False}
		save_results = True
		dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' + ('update_order/' if update_order else '')
		
		if not os.path.exists(dir_save):
			os.makedirs(dir_save)
		file_output = open(dir_save + 'output.txt', 'a')
# 		sys.stdout = file_output

		# print settings.
		print('parameters:')
		print('dataset name:', ds_name)
		print('mpg_options:', mpg_options)
		print('kernel_options:', kernel_options)
		print('ged_options:', ged_options)
		print('mge_options:', mge_options)
		print('save_results:', save_results)
		
		for train_examples in ['k-graphs', 'expert', 'random', 'best-dataset', 'trainset']:
# 		for train_examples in ['expert']:
			print('\n-------------------------------------')
			print('train examples used:', train_examples, '\n')
			mpg_options['fit_method'] = train_examples
# 			try:
			kernel_knn_cv(ds_name, train_examples, knn_options, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, load_gm='auto', dir_save=dir_save, irrelevant_labels=None, edge_required=False, cut_range=None)
# 			except Exception as exp:
# 				print('An exception occured when running this experiment:')
# 				LOG_FILENAME = dir_save + 'error.txt'
# 				logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
# 				logging.exception('')
# 				print(repr(exp))


if __name__ == '__main__':
	xp_knn_1_1()