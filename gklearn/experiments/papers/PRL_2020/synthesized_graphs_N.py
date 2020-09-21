#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:34:26 2020

@author: ljia
"""
Graph_Kernel_List = ['PathUpToH', 'WLSubtree', 'SylvesterEquation', 'Marginalized', 'ShortestPath', 'Treelet', 'ConjugateGradient', 'FixedPoint', 'SpectralDecomposition', 'CommonWalk'] 
# Graph_Kernel_List = ['CommonWalk', 'Marginalized', 'SylvesterEquation', 'ConjugateGradient', 'FixedPoint', 'SpectralDecomposition', 'ShortestPath', 'StructuralSP', 'PathUpToH', 'Treelet', 'WLSubtree']

def generate_graphs():
	from gklearn.utils.graph_synthesizer import GraphSynthesizer
	gsyzer = GraphSynthesizer()
	graphs = gsyzer.unified_graphs(num_graphs=1000, num_nodes=20, num_edges=40, num_node_labels=0, num_edge_labels=0, seed=None, directed=False)
	return graphs


def compute_graph_kernel(graphs, kernel_name):
	import multiprocessing
	
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
		
	params['n_jobs'] = multiprocessing.cpu_count()
	params['verbose'] = True
	results = estimator(graphs, **params)
	
	return results[0], results[1]


def xp_synthesied_graphs_dataset_size():
	
	# Generate graphs.
	graphs = generate_graphs()
	
	# Run and save.
	import pickle
	import os
	save_dir = 'outputs/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	run_times = {}
	
	for kernel_name in Graph_Kernel_List:
		print()
		print('Kernel:', kernel_name)
		
		run_times[kernel_name] = []
		for num_graphs in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
			print()
			print('Number of graphs:', num_graphs)
			
			sub_graphs = [g.copy() for g in graphs[0:num_graphs]]
			gram_matrix, run_time = compute_graph_kernel(sub_graphs, kernel_name)
			run_times[kernel_name].append(run_time)
			
			pickle.dump(run_times, open(save_dir + 'run_time.' + kernel_name + '.' + str(num_graphs) + '.pkl', 'wb'))
		
	# Save all.	
	pickle.dump(run_times, open(save_dir + 'run_times.pkl', 'wb'))	
	
	return


if __name__ == '__main__':
	xp_synthesied_graphs_dataset_size()