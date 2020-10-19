#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:39:29 2020

@author: ljia
"""
import multiprocessing
import functools
from gklearn.preimage.utils import generate_median_preimages_by_class

		
def test_median_preimage_generator():
	"""MAO, Treelet, using CONSTANT, symbolic only.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'MAO' #
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [4, 4, 2, 1, 1, 1], #
				   'ds_name': ds_name,
				   'parallel': True, # False
				   'time_limit_in_sec': 0,
				   'max_itrs': 3, # 
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
				   'initial_solutions': 1, # 1
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
	dir_save = ds_name + '.' + kernel_options['name'] + '.symb.pytest/'
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
	for fit_method in ['k-graphs', 'expert', 'random']:
		print('\n-------------------------------------')
		print('fit method:', fit_method, '\n')
		mpg_options['fit_method'] = fit_method
		try:
			generate_median_preimages_by_class(ds_name, mpg_options, kernel_options, ged_options, mge_options, save_results=save_results, save_medians=True, plot_medians=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required, cut_range=range(0, 4))
		except Exception as exception:
			assert False, exception
			
	
if __name__ == '__main__':
	test_median_preimage_generator()