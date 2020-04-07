#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:30:55 2020

@author: ljia
"""
import multiprocessing
import functools
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from gklearn.preimage import MedianPreimageGenerator
from gklearn.utils import Dataset


def test_median_preimage_generator():
	
	# 1. set parameters.
	print('1. setting parameters...')
	ds_name = 'Letter-high'
	mpg = MedianPreimageGenerator()
	mpg_options = {'fit_method': 'k-graphs',
				   'init_ecc': [3, 3, 1, 3, 3],
				   'ds_name': 'Letter-high',
				   'parallel': True,
				   'time_limit_in_sec': 0,
				   'max_itrs': 100,
				   'max_itrs_without_update': 3,
				   'epsilon_ratio': 0.01,
				   'verbose': 2}
	mpg.set_options(**mpg_options)
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	mpg.kernel_options = {'name': 'structuralspkernel',
					      'edge_weight': None,
						  'node_kernels': sub_kernels,
						  'edge_kernels': sub_kernels, 
						  'compute_method': 'naive',
						  'parallel': 'imap_unordered', 
# 						  'parallel': None, 
						  'n_jobs': multiprocessing.cpu_count(),
						  'normalize': True,
						  'verbose': 2}
	mpg.ged_options = {'method': 'IPFP',
					   'initial_solutions': 40,
					   'edit_cost': 'LETTER2',
					   'attr_distance': 'euclidean',
					   'ratio_runs_from_initial_solutions': 1,
					   'threads': multiprocessing.cpu_count(),
					   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'}
	mpg.mge_options = {'init_type': 'MEDOID',
					   'random_inits': 10,
					   'time_limit': 600,
					   'verbose': 2,
					   'refine': False}
	
	
	# 2. get dataset.
	print('2. getting dataset...')
	mpg.dataset = Dataset()
	mpg.dataset.load_predefined_dataset(ds_name)
	mpg.dataset.cut_graphs(range(0, 10))
		
	# 3. compute median preimage.
	print('3. computing median preimage...')
	mpg.run()
	
	
if __name__ == '__main__':
	test_median_preimage_generator()