#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:33:28 2020

@author: ljia
"""
import multiprocessing
import numpy as np
from gklearn.utils import model_selection_for_precomputed_kernel


Graph_Kernel_List = ['PathUpToH', 'WLSubtree', 'SylvesterEquation', 'Marginalized', 'ShortestPath', 'Treelet', 'ConjugateGradient', 'FixedPoint', 'SpectralDecomposition', 'StructuralSP', 'CommonWalk']
# Graph_Kernel_List = ['CommonWalk', 'Marginalized', 'SylvesterEquation', 'ConjugateGradient', 'FixedPoint', 'SpectralDecomposition', 'ShortestPath', 'StructuralSP', 'PathUpToH', 'Treelet', 'WLSubtree']


Graph_Kernel_List_VSym = ['PathUpToH', 'WLSubtree', 'Marginalized', 'ShortestPath', 'Treelet', 'ConjugateGradient', 'FixedPoint', 'StructuralSP', 'CommonWalk'] 


Graph_Kernel_List_ESym = ['PathUpToH', 'Marginalized', 'Treelet', 'ConjugateGradient', 'FixedPoint', 'StructuralSP', 'CommonWalk']


Graph_Kernel_List_VCon = ['ShortestPath', 'ConjugateGradient', 'FixedPoint', 'StructuralSP']


Graph_Kernel_List_ECon = ['ConjugateGradient', 'FixedPoint', 'StructuralSP']


Dataset_List = ['Alkane', 'Acyclic', 'MAO', 'PAH', 'MUTAG', 'Letter-med', 'ENZYMES', 'AIDS', 'NCI1', 'NCI109', 'DD']


def compute_graph_kernel(graphs, kernel_name, n_jobs=multiprocessing.cpu_count(), chunksize=None):
	
	if kernel_name == 'CommonWalk':
		from gklearn.kernels.commonWalkKernel import commonwalkkernel
		estimator = commonwalkkernel
		params = {'compute_method': 'geo', 'weight': 0.1}
		
	elif kernel_name == 'Marginalized':
		from gklearn.kernels.marginalizedKernel import marginalizedkernel
		estimator = marginalizedkernel
		params = {'p_quit': 0.5, 'n_iteration': 5, 'remove_totters': False}
		
	elif kernel_name == 'SylvesterEquation':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		params = {'compute_method': 'sylvester', 'weight': 0.1}
		
	elif kernel_name == 'ConjugateGradient':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		params = {'compute_method': 'conjugate', 'weight': 0.1, 'node_kernels': sub_kernel, 'edge_kernels': sub_kernel}
		
	elif kernel_name == 'FixedPoint':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		params = {'compute_method': 'fp', 'weight': 1e-4, 'node_kernels': sub_kernel, 'edge_kernels': sub_kernel}
		
	elif kernel_name == 'SpectralDecomposition':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		params = {'compute_method': 'spectral', 'sub_kernel': 'geo', 'weight': 0.1}
		
	elif kernel_name == 'ShortestPath':
		from gklearn.kernels.spKernel import spkernel
		estimator = spkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		params = {'node_kernels': sub_kernel}
		
	elif kernel_name == 'StructuralSP':
		from gklearn.kernels.structuralspKernel import structuralspkernel
		estimator = structuralspkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		params = {'node_kernels': sub_kernel, 'edge_kernels': sub_kernel}
		
	elif kernel_name == 'PathUpToH':
		from gklearn.kernels.untilHPathKernel import untilhpathkernel
		estimator = untilhpathkernel
		params = {'depth': 5, 'k_func': 'MinMax', 'compute_method': 'trie'}
		
	elif kernel_name == 'Treelet':
		from gklearn.kernels.treeletKernel import treeletkernel
		estimator = treeletkernel
		from gklearn.utils.kernels import polynomialkernel
		import functools
		sub_kernel = functools.partial(polynomialkernel, d=4, c=1e+8)
		params = {'sub_kernel': sub_kernel}
		
	elif kernel_name == 'WLSubtree':
		from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
		estimator = weisfeilerlehmankernel
		params = {'base_kernel': 'subtree', 'height': 5}
		
# 	params['parallel'] = None
	params['n_jobs'] = n_jobs
	params['chunksize'] = chunksize
	params['verbose'] = True
	results = estimator(graphs, **params)
	
	return results[0], results[1]


def cross_validate(graphs, targets, kernel_name, output_dir='outputs/', ds_name='synthesized', n_jobs=multiprocessing.cpu_count()):
	
	param_grid = None
	
	if kernel_name == 'CommonWalk':
		from gklearn.kernels.commonWalkKernel import commonwalkkernel
		estimator = commonwalkkernel
		param_grid_precomputed = [{'compute_method': ['geo'], 
							 'weight': np.linspace(0.01, 0.15, 15)}]
		
	elif kernel_name == 'Marginalized':
		from gklearn.kernels.marginalizedKernel import marginalizedkernel
		estimator = marginalizedkernel
		param_grid_precomputed = {'p_quit': np.linspace(0.1, 0.9, 9),
                          'n_iteration': np.linspace(1, 19, 7), 
                          'remove_totters': [False]}
		
	elif kernel_name == 'SylvesterEquation':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		param_grid_precomputed = {'compute_method': ['sylvester'],
#                          'weight': np.linspace(0.01, 0.10, 10)}
                          'weight': np.logspace(-1, -10, num=10, base=10)}
		
	elif kernel_name == 'ConjugateGradient':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		param_grid_precomputed = {'compute_method': ['conjugate'], 
                          'node_kernels': [sub_kernel], 'edge_kernels': [sub_kernel],
                          'weight': np.logspace(-1, -10, num=10, base=10)}
		
	elif kernel_name == 'FixedPoint':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		param_grid_precomputed = {'compute_method': ['fp'], 
                          'node_kernels': [sub_kernel], 'edge_kernels': [sub_kernel],
                          'weight': np.logspace(-4, -10, num=7, base=10)}
		
	elif kernel_name == 'SpectralDecomposition':
		from gklearn.kernels.randomWalkKernel import randomwalkkernel
		estimator = randomwalkkernel
		param_grid_precomputed = {'compute_method': ['spectral'],
                          'weight': np.logspace(-1, -10, num=10, base=10),
                          'sub_kernel': ['geo', 'exp']}
		
	elif kernel_name == 'ShortestPath':
		from gklearn.kernels.spKernel import spkernel
		estimator = spkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		param_grid_precomputed = {'node_kernels': [sub_kernel]}
		
	elif kernel_name == 'StructuralSP':
		from gklearn.kernels.structuralspKernel import structuralspkernel
		estimator = structuralspkernel
		from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
		import functools
		mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
		sub_kernel = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
		param_grid_precomputed = {'node_kernels': [sub_kernel], 'edge_kernels': [sub_kernel],
                          'compute_method': ['naive']}
		
	elif kernel_name == 'PathUpToH':
		from gklearn.kernels.untilHPathKernel import untilhpathkernel
		estimator = untilhpathkernel
		param_grid_precomputed = {'depth': np.linspace(1, 10, 10),   # [2], 
                          'k_func': ['MinMax', 'tanimoto'], # ['MinMax'], # 
                          'compute_method': ['trie']} # ['MinMax']}
		
	elif kernel_name == 'Treelet':
		from gklearn.kernels.treeletKernel import treeletkernel
		estimator = treeletkernel
		from gklearn.utils.kernels import gaussiankernel, polynomialkernel
		import functools
		gkernels = [functools.partial(gaussiankernel, gamma=1 / ga) 
		#            for ga in np.linspace(1, 10, 10)]
			  for ga in np.logspace(0, 10, num=11, base=10)]
		pkernels = [functools.partial(polynomialkernel, d=d, c=c) for d in range(1, 11) 
			  for c in np.logspace(0, 10, num=11, base=10)]
# 		pkernels = [functools.partial(polynomialkernel, d=1, c=1)]

		param_grid_precomputed = {'sub_kernel': pkernels + gkernels}
# 							'parallel': [None]}
		
	elif kernel_name == 'WLSubtree':
		from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
		estimator = weisfeilerlehmankernel
		param_grid_precomputed = {'base_kernel': ['subtree'], 
                          'height': np.linspace(0, 10, 11)}
		param_grid = {'C': np.logspace(-10, 4, num=29, base=10)}
		
	if param_grid is None:
		param_grid = {'C': np.logspace(-10, 10, num=41, base=10)}
	
	results = model_selection_for_precomputed_kernel(
        graphs,
        estimator,
        param_grid_precomputed,
        param_grid,
        'classification',
        NUM_TRIALS=28,
        datafile_y=targets,
        extra_params=None,
        ds_name=ds_name,
		output_dir=output_dir,
        n_jobs=n_jobs,
        read_gm_from_file=False,
        verbose=True)
	
	return results[0], results[1]