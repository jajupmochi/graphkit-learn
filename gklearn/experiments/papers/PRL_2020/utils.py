#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:33:28 2020

@author: ljia
"""
import multiprocessing


Graph_Kernel_List = ['PathUpToH', 'WLSubtree', 'SylvesterEquation', 'Marginalized', 'ShortestPath', 'Treelet', 'ConjugateGradient', 'FixedPoint', 'SpectralDecomposition', 'StructuralSP', 'CommonWalk']
# Graph_Kernel_List = ['CommonWalk', 'Marginalized', 'SylvesterEquation', 'ConjugateGradient', 'FixedPoint', 'SpectralDecomposition', 'ShortestPath', 'StructuralSP', 'PathUpToH', 'Treelet', 'WLSubtree']


Graph_Kernel_List_VSym = ['PathUpToH', 'WLSubtree', 'Marginalized', 'ShortestPath', 'Treelet', 'ConjugateGradient', 'FixedPoint', 'StructuralSP', 'CommonWalk'] 


Graph_Kernel_List_ESym = ['PathUpToH', 'Marginalized', 'Treelet', 'ConjugateGradient', 'FixedPoint', 'StructuralSP', 'CommonWalk']


Graph_Kernel_List_VCon = ['ShortestPath', 'ConjugateGradient', 'FixedPoint', 'StructuralSP']


Graph_Kernel_List_ECon = ['ConjugateGradient', 'FixedPoint', 'StructuralSP']


Dataset_List = ['Alkane', 'Acyclic', 'MAO', 'PAH', 'MUTAG', 'Letter-med', 'ENZYMES', 'AIDS', 'NCI1', 'NCI109', 'DD']


def compute_graph_kernel(graphs, kernel_name, n_jobs=multiprocessing.cpu_count()):
	
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
		params = {'compute_method': 'fp', 'weight': 1e-3, 'node_kernels': sub_kernel, 'edge_kernels': sub_kernel}
		
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
	params['verbose'] = True
	results = estimator(graphs, **params)
	
	return results[0], results[1]