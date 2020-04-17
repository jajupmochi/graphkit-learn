#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:39:29 2020

@author: ljia
"""
import multiprocessing
import functools
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from gklearn.preimage.utils import generate_median_preimages_by_class
from gklearn.utils import compute_gram_matrices_by_class


def xp_median_preimage_12_1():
	"""xp 12_1: PAH, StructuralSP, using NON_SYMBOLIC, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 0, 1, 1, 0], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'NON_SYMBOLIC', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.unlabeled/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_12_2():
	"""xp 12_2: PAH, PathUpToH, using CONSTANT, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 1, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.unlabeled/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		

def xp_median_preimage_12_3():
	"""xp 12_3: PAH, Treelet, using CONSTANT, unlabeled.
	"""
	from gklearn.utils.kernels import gaussiankernel
	# set parameters.
	ds_name = 'PAH' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	pkernel = functools.partial(gaussiankernel, gamma=None) # @todo
	kernel_options = {'name': 'Treelet', #
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.unlabeled/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_12_4():
	"""xp 12_4: PAH, WeisfeilerLehman, using CONSTANT, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 14,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.unlabeled/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
# 	# compute gram matrices for each class a priori.
# 	print('Compute gram matrices for each class a priori.')
# 	compute_gram_matrices_by_class(ds_name, kernel_options, save_results=True, dir_save=dir_save, irrelevant_labels=irrelevant_labels)
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_12_5():
	"""xp 12_5: PAH, ShortestPath, using NON_SYMBOLIC, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 0, 1, 1, 0], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'ShortestPath',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'NON_SYMBOLIC', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.unlabeled/' # 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = True #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5: # 
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)


def xp_median_preimage_9_1():
	"""xp 9_1: MAO, StructuralSP, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'MAO' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.symb/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_9_2():
	"""xp 9_2: MAO, PathUpToH, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'MAO' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 9, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.symb/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_9_3():
	"""xp 9_3: MAO, Treelet, using CONSTANT, symbolic only.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'MAO' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	pkernel = functools.partial(polynomialkernel, d=4, c=1e+7)
	kernel_options = {'name': 'Treelet', #
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.symb/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_9_4():
	"""xp 9_4: MAO, WeisfeilerLehman, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'MAO' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 6,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.symb/'
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
# 	# compute gram matrices for each class a priori.
# 	print('Compute gram matrices for each class a priori.')
# 	compute_gram_matrices_by_class(ds_name, kernel_options, save_results=True, dir_save=dir_save, irrelevant_labels=irrelevant_labels)
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)


def xp_median_preimage_8_1():
	"""xp 8_1: Monoterpenoides, StructuralSP, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'Monoterpenoides' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [3, 3, 1, 3, 3, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_8_2():
	"""xp 8_2: Monoterpenoides, PathUpToH, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'Monoterpenoides' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 7, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_8_3():
	"""xp 8_3: Monoterpenoides, Treelet, using CONSTANT.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'Monoterpenoides' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	pkernel = functools.partial(polynomialkernel, d=2, c=1e+5)
	kernel_options = {'name': 'Treelet',
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_8_4():
	"""xp 8_4: Monoterpenoides, WeisfeilerLehman, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'Monoterpenoides' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 4,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		

def xp_median_preimage_7_1():
	"""xp 7_1: MUTAG, StructuralSP, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'MUTAG' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_7_2():
	"""xp 7_2: MUTAG, PathUpToH, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'MUTAG' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 2, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_7_3():
	"""xp 7_3: MUTAG, Treelet, using CONSTANT.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'MUTAG' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	pkernel = functools.partial(polynomialkernel, d=3, c=1e+8)
	kernel_options = {'name': 'Treelet',
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_7_4():
	"""xp 7_4: MUTAG, WeisfeilerLehman, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'MUTAG' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100, # 
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 1,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'CONSTANT', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)


def xp_median_preimage_6_1():
	"""xp 6_1: COIL-RAG, StructuralSP, using NON_SYMBOLIC.
	"""
	# set parameters.
	ds_name = 'COIL-RAG' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [3, 3, 1, 3, 3, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'NON_SYMBOLIC', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_6_2():
	"""xp 6_2: COIL-RAG, ShortestPath, using NON_SYMBOLIC.
	"""
	# set parameters.
	ds_name = 'COIL-RAG' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [3, 3, 1, 3, 3, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'ShortestPath',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'NON_SYMBOLIC', # 
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = True #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)


def xp_median_preimage_5_1():
	"""xp 5_1: FRANKENSTEIN, StructuralSP, using NON_SYMBOLIC.
	"""
	# set parameters.
	ds_name = 'FRANKENSTEIN' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [3, 3, 1, 3, 3, 0], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'NON_SYMBOLIC',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		

def xp_median_preimage_4_1():
	"""xp 4_1: COLORS-3, StructuralSP, using NON_SYMBOLIC.
	"""
	# set parameters.
	ds_name = 'COLORS-3' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [3, 3, 1, 3, 3, 0], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'NON_SYMBOLIC',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		
		
def xp_median_preimage_3_2():
	"""xp 3_2: Fingerprint, ShortestPath, using LETTER2, only node attrs.
	"""
	# set parameters.
	ds_name = 'Fingerprint' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.525, 0.525, 0.001, 0.125, 0.125], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'ShortestPath',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = {'edge_attrs': ['orient', 'angle']} #
	edge_required = True #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)


def xp_median_preimage_3_1():
	"""xp 3_1: Fingerprint, StructuralSP, using LETTER2, only node attrs.
	"""
	# set parameters.
	ds_name = 'Fingerprint' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.525, 0.525, 0.001, 0.125, 0.125], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = {'edge_attrs': ['orient', 'angle']} #
	edge_required = False #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		

def xp_median_preimage_2_1():
	"""xp 2_1: COIL-DEL, StructuralSP, using LETTER2, only node attrs.
	"""
	# set parameters.
	ds_name = 'COIL-DEL' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [3, 3, 1, 3, 3],
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '.node_attrs/'
	irrelevant_labels = {'edge_labels': ['valence']}
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
# 	# compute gram matrices for each class a priori.
# 	print('Compute gram matrices for each class a priori.')
# 	compute_gram_matrices_by_class(ds_name, kernel_options, save_results=True, dir_save=dir_save, irrelevant_labels=irrelevant_labels)
	
	# generate preimages.
	for fit_method in ['k-graphs'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels)


def xp_median_preimage_1_1():
	"""xp 1_1: Letter-high, StructuralSP.
	"""
	# set parameters.
	ds_name = 'Letter-high'
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.675, 0.675, 0.75, 0.425, 0.425],
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
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
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save)
		
		
def xp_median_preimage_1_2():
	"""xp 1_2: Letter-high, ShortestPath.
	"""
	# set parameters.
	ds_name = 'Letter-high'
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.675, 0.675, 0.75, 0.425, 0.425],
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'ShortestPath',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'parallel': 'imap_unordered', 
# 						  'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = True #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
		

def xp_median_preimage_10_1():
	"""xp 10_1: Letter-med, StructuralSP.
	"""
	# set parameters.
	ds_name = 'Letter-med'
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.525, 0.525, 0.75, 0.475, 0.475],
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
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
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save)
		
		
def xp_median_preimage_10_2():
	"""xp 10_2: Letter-med, ShortestPath.
	"""
	# set parameters.
	ds_name = 'Letter-med'
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.525, 0.525, 0.75, 0.475, 0.475],
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'ShortestPath',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'parallel': 'imap_unordered', 
# 						  'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = True #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)


def xp_median_preimage_11_1():
	"""xp 11_1: Letter-low, StructuralSP.
	"""
	# set parameters.
	ds_name = 'Letter-low'
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.075, 0.075, 0.25, 0.075, 0.075],
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
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
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save)
		
		
def xp_median_preimage_11_2():
	"""xp 11_2: Letter-low, ShortestPath.
	"""
	# set parameters.
	ds_name = 'Letter-low'
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [0.075, 0.075, 0.25, 0.075, 0.075],
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_residual': 0.01,
				   'epsilon_ec': 0.1,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'ShortestPath',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'parallel': 'imap_unordered', 
# 						  'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 2}
	ged_options = {'method': 'IPFP',
				   'initialization_method': 'RANDOM', # 'NODE'
				   'initial_solutions': 10, # 1
				   'edit_cost': 'LETTER2',
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mge_options = {'init_type': 'MEDOID',
				   'random_inits': 10,
				   'time_limit': 600,
				   'verbose': 2,
				   'refine': False}
	save_results = True
	dir_save = '../results/xp_median_preimage/' + ds_name + '.' + kernel_options['name'] + '/'
	irrelevant_labels = None #
	edge_required = True #
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('mpg_options:', mpg_options)
	print('kernel_options:', kernel_options)
	print('ged_options:', ged_options)
	print('mge_options:', mge_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	for fit_method in ['k-graphs', 'expert'] + ['random'] * 5:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)


if __name__ == "__main__":
	
# 	#### xp 1_1: Letter-high, StructuralSP.
#  	# xp_median_preimage_1_1()

# 	#### xp 1_2: Letter-high, ShortestPath.
# # 	xp_median_preimage_1_2()

# 	#### xp 10_1: Letter-med, StructuralSP.
#  	# xp_median_preimage_10_1()

# 	#### xp 10_2: Letter-med, ShortestPath.
#  	# xp_median_preimage_10_2()

# 	#### xp 11_1: Letter-low, StructuralSP.
#  	# xp_median_preimage_11_1()

# 	#### xp 11_2: Letter-low, ShortestPath.
# # 	xp_median_preimage_11_2()
# 	
# 	#### xp 2_1: COIL-DEL, StructuralSP, using LETTER2, only node attrs.
# # 	xp_median_preimage_2_1()
# 	
# 	#### xp 3_1: Fingerprint, StructuralSP, using LETTER2, only node attrs.
#  	# xp_median_preimage_3_1()

# 	#### xp 3_2: Fingerprint, ShortestPath, using LETTER2, only node attrs.
 	# xp_median_preimage_3_2()

# 	#### xp 4_1: COLORS-3, StructuralSP, using NON_SYMBOLIC.
# # 	xp_median_preimage_4_1()
# 	
# 	#### xp 5_1: FRANKENSTEIN, StructuralSP, using NON_SYMBOLIC.
# # 	xp_median_preimage_5_1()
# 	
# 	#### xp 6_1: COIL-RAG, StructuralSP, using NON_SYMBOLIC.
#  	# xp_median_preimage_6_1()

# 	#### xp 6_2: COIL-RAG, ShortestPath, using NON_SYMBOLIC.
 	# xp_median_preimage_6_2()

# 	#### xp 7_1: MUTAG, StructuralSP, using CONSTANT.
#  	# xp_median_preimage_7_1()

# 	#### xp 7_2: MUTAG, PathUpToH, using CONSTANT.
#  	# xp_median_preimage_7_2()

# 	#### xp 7_3: MUTAG, Treelet, using CONSTANT.
#  	# xp_median_preimage_7_3()

# 	#### xp 7_4: MUTAG, WeisfeilerLehman, using CONSTANT.
#  	xp_median_preimage_7_4()
# 	 
#     #### xp 8_1: Monoterpenoides, StructuralSP, using CONSTANT.
# # 	xp_median_preimage_8_1()

# 	#### xp 8_2: Monoterpenoides, PathUpToH, using CONSTANT.
#  	# xp_median_preimage_8_2()

# 	#### xp 8_3: Monoterpenoides, Treelet, using CONSTANT.
#  	# xp_median_preimage_8_3()

# 	#### xp 8_4: Monoterpenoides, WeisfeilerLehman, using CONSTANT.
#  	xp_median_preimage_8_4()

# 	#### xp 9_1: MAO, StructuralSP, using CONSTANT, symbolic only.
 	# xp_median_preimage_9_1()

# 	#### xp 9_2: MAO, PathUpToH, using CONSTANT, symbolic only.
 	# xp_median_preimage_9_2()

# 	#### xp 9_3: MAO, Treelet, using CONSTANT, symbolic only.
 	# xp_median_preimage_9_3()

# 	#### xp 9_4: MAO, WeisfeilerLehman, using CONSTANT, symbolic only.
 	# xp_median_preimage_9_4()

	#### xp 12_1: PAH, StructuralSP, using NON_SYMBOLIC, unlabeled.
 	# xp_median_preimage_12_1()

	#### xp 12_2: PAH, PathUpToH, using CONSTANT, unlabeled.
 	# xp_median_preimage_12_2()

	#### xp 12_3: PAH, Treelet, using CONSTANT, unlabeled.
# 	xp_median_preimage_12_3()

	#### xp 12_4: PAH, WeisfeilerLehman, using CONSTANT, unlabeled.
 	# xp_median_preimage_12_4()

	#### xp 12_5: PAH, ShortestPath, using NON_SYMBOLIC, unlabeled.
	xp_median_preimage_12_5()




# 	#### xp 7_4: MUTAG, WeisfeilerLehman, using CONSTANT.
 	# xp_median_preimage_7_4()
 
# 	#### xp 8_4: Monoterpenoides, WeisfeilerLehman, using CONSTANT.
 	# xp_median_preimage_8_4()

# 	#### xp 9_4: MAO, WeisfeilerLehman, using CONSTANT, symbolic only.
 	# xp_median_preimage_9_4()
	 
# 	#### xp 10_1: Letter-med, StructuralSP.
#  	xp_median_preimage_10_1()

# 	#### xp 10_2: Letter-med, ShortestPath.
#  	xp_median_preimage_10_2()

# 	#### xp 11_1: Letter-low, StructuralSP.
#  	xp_median_preimage_11_1()

# 	#### xp 11_2: Letter-low, ShortestPath.
#  	xp_median_preimage_11_2()
# 	 
# 	#### xp 1_1: Letter-high, StructuralSP.
#  	xp_median_preimage_1_1()

# 	#### xp 1_2: Letter-high, ShortestPath.
#  	xp_median_preimage_1_2()

# 	#### xp 3_1: Fingerprint, StructuralSP, using LETTER2, only node attrs.
#  	xp_median_preimage_3_1()
# 	 
 	# #### xp 6_1: COIL-RAG, StructuralSP, using NON_SYMBOLIC.
 	# xp_median_preimage_6_1()

 	# #### xp 6_2: COIL-RAG, ShortestPath, using NON_SYMBOLIC.
 	# xp_median_preimage_6_2()
	 # 
  #   #### xp 3_2: Fingerprint, ShortestPath, using LETTER2, only node attrs.
 	# xp_median_preimage_3_2()

 	#### xp 7_1: MUTAG, StructuralSP, using CONSTANT.
 	# xp_median_preimage_7_1()

  #   #### xp 8_1: Monoterpenoides, StructuralSP, using CONSTANT.
 	# xp_median_preimage_8_1()

 	# #### xp 9_1: MAO, StructuralSP, using CONSTANT, symbolic only.
 	# xp_median_preimage_9_1()

 	# #### xp 2_1: COIL-DEL, StructuralSP, using LETTER2, only node attrs.
 	# xp_median_preimage_2_1()

 	#### xp 5_1: FRANKENSTEIN, StructuralSP, using NON_SYMBOLIC.
 	# xp_median_preimage_5_1()

 	#### xp 4_1: COLORS-3, StructuralSP, using NON_SYMBOLIC.
 	# xp_median_preimage_4_1()
