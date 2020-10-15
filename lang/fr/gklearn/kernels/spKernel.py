"""
@author: linlin

@references: 
	
	[1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData 
	Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
"""

import sys
import time
from itertools import product
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import networkx as nx
import numpy as np

from gklearn.utils.utils import getSPGraph
from gklearn.utils.graphdataset import get_dataset_attributes
from gklearn.utils.parallel import parallel_gm

def spkernel(*args,
			 node_label='atom',
			 edge_weight=None,
			 node_kernels=None,
			 parallel='imap_unordered',
			 n_jobs=None,
			 chunksize=None,
			 verbose=True):
	"""Compute shortest-path kernels between graphs.

	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.
	
	G1, G2 : NetworkX graphs
		Two graphs between which the kernel is computed.

	node_label : string
		Node attribute used as label. The default node label is atom.

	edge_weight : string
		Edge attribute name corresponding to the edge weight.

	node_kernels : dict
		A dictionary of kernel functions for nodes, including 3 items: 'symb' 
		for symbolic node labels, 'nsymb' for non-symbolic node labels, 'mix' 
		for both labels. The first 2 functions take two node labels as 
		parameters, and the 'mix' function takes 4 parameters, a symbolic and a
		non-symbolic label for each the two nodes. Each label is in form of 2-D
		dimension array (n_samples, n_features). Each function returns an 
		number as the kernel value. Ignored when nodes are unlabeled.

	n_jobs : int
		Number of jobs for parallelization.

	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the sp kernel between 2 praphs.
	"""
	# pre-process
	Gn = args[0] if len(args) == 1 else [args[0], args[1]]
	Gn = [g.copy() for g in Gn]
	weight = None
	if edge_weight is None:
		if verbose:
			print('\n None edge weight specified. Set all weight to 1.\n')
	else:
		try:
			some_weight = list(
				nx.get_edge_attributes(Gn[0], edge_weight).values())[0]
			if isinstance(some_weight, (float, int)):
				weight = edge_weight
			else:
				if verbose:
					print(
							'\n Edge weight with name %s is not float or integer. Set all weight to 1.\n'
							% edge_weight)
		except:
			if verbose:
				print(
						'\n Edge weight with name "%s" is not found in the edge attributes. Set all weight to 1.\n'
						% edge_weight)
	ds_attrs = get_dataset_attributes(
		Gn,
		attr_names=['node_labeled', 'node_attr_dim', 'is_directed'],
		node_label=node_label)

	# remove graphs with no edges, as no sp can be found in their structures, 
	# so the kernel between such a graph and itself will be zero.
	len_gn = len(Gn)
	Gn = [(idx, G) for idx, G in enumerate(Gn) if nx.number_of_edges(G) != 0]
	idx = [G[0] for G in Gn]
	Gn = [G[1] for G in Gn]
	if len(Gn) != len_gn:
		if verbose:
			print('\n %d graphs are removed as they don\'t contain edges.\n' %
				  (len_gn - len(Gn)))

	start_time = time.time()

	if parallel == 'imap_unordered':
		pool = Pool(n_jobs)
		# get shortest path graphs of Gn
		getsp_partial = partial(wrapper_getSPGraph, weight)
		itr = zip(Gn, range(0, len(Gn)))
		if chunksize is None:
			if len(Gn) < 100 * n_jobs:
		#		# use default chunksize as pool.map when iterable is less than 100
		#		chunksize, extra = divmod(len(Gn), n_jobs * 4)
		#		if extra:
		#			chunksize += 1
				chunksize = int(len(Gn) / n_jobs) + 1
			else:
				chunksize = 100
		if verbose:
			iterator = tqdm(pool.imap_unordered(getsp_partial, itr, chunksize),
							desc='getting sp graphs', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(getsp_partial, itr, chunksize)
		for i, g in iterator:
			Gn[i] = g
		pool.close()
		pool.join()
		
	elif parallel is None:
		pass
#	# ---- direct running, normally use single CPU core. ----
#	for i in tqdm(range(len(Gn)), desc='getting sp graphs', file=sys.stdout):
#		i, Gn[i] = wrapper_getSPGraph(weight, (Gn[i], i))

	# # ---- use pool.map to parallel ----
	# result_sp = pool.map(getsp_partial, range(0, len(Gn)))
	# for i in result_sp:
	#	 Gn[i[0]] = i[1]
	# or
	# getsp_partial = partial(wrap_getSPGraph, Gn, weight)
	# for i, g in tqdm(
	#		 pool.map(getsp_partial, range(0, len(Gn))),
	#		 desc='getting sp graphs',
	#		 file=sys.stdout):
	#	 Gn[i] = g

	# # ---- only for the Fast Computation of Shortest Path Kernel (FCSP)
	# sp_ml = [0] * len(Gn)  # shortest path matrices
	# for i in result_sp:
	#	 sp_ml[i[0]] = i[1]
	# edge_x_g = [[] for i in range(len(sp_ml))]
	# edge_y_g = [[] for i in range(len(sp_ml))]
	# edge_w_g = [[] for i in range(len(sp_ml))]
	# for idx, item in enumerate(sp_ml):
	#	 for i1 in range(len(item)):
	#		 for i2 in range(i1 + 1, len(item)):
	#			 if item[i1, i2] != np.inf:
	#				 edge_x_g[idx].append(i1)
	#				 edge_y_g[idx].append(i2)
	#				 edge_w_g[idx].append(item[i1, i2])
	# print(len(edge_x_g[0]))
	# print(len(edge_y_g[0]))
	# print(len(edge_w_g[0]))

	Kmatrix = np.zeros((len(Gn), len(Gn)))

	# ---- use pool.imap_unordered to parallel and track progress. ----
	def init_worker(gn_toshare):
		global G_gn
		G_gn = gn_toshare
	do_partial = partial(wrapper_sp_do, ds_attrs, node_label, node_kernels)   
	parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
				glbv=(Gn,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


	# # ---- use pool.map to parallel. ----
	# # result_perf = pool.map(do_partial, itr)
	# do_partial = partial(spkernel_do, Gn, ds_attrs, node_label, node_kernels)
	# itr = combinations_with_replacement(range(0, len(Gn)), 2)
	# for i, j, kernel in tqdm(
	#		 pool.map(do_partial, itr), desc='Computing kernels',
	#		 file=sys.stdout):
	#	 Kmatrix[i][j] = kernel
	#	 Kmatrix[j][i] = kernel
	# pool.close()
	# pool.join()

	# # ---- use joblib.Parallel to parallel and track progress. ----
	# result_perf = Parallel(
	#	 n_jobs=n_jobs, verbose=10)(
	#		 delayed(do_partial)(ij)
	#		 for ij in combinations_with_replacement(range(0, len(Gn)), 2))
	# result_perf = [
	#	 do_partial(ij)
	#	 for ij in combinations_with_replacement(range(0, len(Gn)), 2)
	# ]
	# for i in result_perf:
	#	 Kmatrix[i[0]][i[1]] = i[2]
	#	 Kmatrix[i[1]][i[0]] = i[2]

#	# ---- direct running, normally use single CPU core. ----
#	from itertools import combinations_with_replacement
#	itr = combinations_with_replacement(range(0, len(Gn)), 2)
#	for i, j in tqdm(itr, desc='Computing kernels', file=sys.stdout):
#		kernel = spkernel_do(Gn[i], Gn[j], ds_attrs, node_label, node_kernels)
#		Kmatrix[i][j] = kernel
#		Kmatrix[j][i] = kernel

	run_time = time.time() - start_time
	if verbose:
		print(
				"\n --- shortest path kernel matrix of size %d built in %s seconds ---"
				% (len(Gn), run_time))

	return Kmatrix, run_time, idx


def spkernel_do(g1, g2, ds_attrs, node_label, node_kernels):
	
	kernel = 0

	# compute shortest path matrices first, method borrowed from FCSP.
	vk_dict = {}  # shortest path matrices dict
	if ds_attrs['node_labeled']:
		# node symb and non-synb labeled
		if ds_attrs['node_attr_dim'] > 0:
			kn = node_kernels['mix']
			for n1, n2 in product(
					g1.nodes(data=True), g2.nodes(data=True)):
				vk_dict[(n1[0], n2[0])] = kn(
					n1[1][node_label], n2[1][node_label],
					n1[1]['attributes'], n2[1]['attributes'])
		# node symb labeled
		else:
			kn = node_kernels['symb']
			for n1 in g1.nodes(data=True):
				for n2 in g2.nodes(data=True):
					vk_dict[(n1[0], n2[0])] = kn(n1[1][node_label],
												 n2[1][node_label])
	else:
		# node non-synb labeled
		if ds_attrs['node_attr_dim'] > 0:
			kn = node_kernels['nsymb']
			for n1 in g1.nodes(data=True):
				for n2 in g2.nodes(data=True):
					vk_dict[(n1[0], n2[0])] = kn(n1[1]['attributes'],
												 n2[1]['attributes'])
		# node unlabeled
		else:
			for e1, e2 in product(
					g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					kernel += 1
			return kernel

	# compute graph kernels
	if ds_attrs['is_directed']:
		for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
			if e1[2]['cost'] == e2[2]['cost']:
				nk11, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(e1[1],
															   e2[1])]
				kn1 = nk11 * nk22
				kernel += kn1
	else:
		for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
			if e1[2]['cost'] == e2[2]['cost']:
				# each edge walk is counted twice, starting from both its extreme nodes.
				nk11, nk12, nk21, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(
					e1[0], e2[1])], vk_dict[(e1[1],
											 e2[0])], vk_dict[(e1[1],
															   e2[1])]
				kn1 = nk11 * nk22
				kn2 = nk12 * nk21
				kernel += kn1 + kn2

		# # ---- exact implementation of the Fast Computation of Shortest Path Kernel (FCSP), reference [2], sadly it is slower than the current implementation
		# # compute vertex kernels
		# try:
		#	 vk_mat = np.zeros((nx.number_of_nodes(g1),
		#						nx.number_of_nodes(g2)))
		#	 g1nl = enumerate(g1.nodes(data=True))
		#	 g2nl = enumerate(g2.nodes(data=True))
		#	 for i1, n1 in g1nl:
		#		 for i2, n2 in g2nl:
		#			 vk_mat[i1][i2] = kn(
		#				 n1[1][node_label], n2[1][node_label],
		#				 [n1[1]['attributes']], [n2[1]['attributes']])

		#	 range1 = range(0, len(edge_w_g[i]))
		#	 range2 = range(0, len(edge_w_g[j]))
		#	 for i1 in range1:
		#		 x1 = edge_x_g[i][i1]
		#		 y1 = edge_y_g[i][i1]
		#		 w1 = edge_w_g[i][i1]
		#		 for i2 in range2:
		#			 x2 = edge_x_g[j][i2]
		#			 y2 = edge_y_g[j][i2]
		#			 w2 = edge_w_g[j][i2]
		#			 ke = (w1 == w2)
		#			 if ke > 0:
		#				 kn1 = vk_mat[x1][x2] * vk_mat[y1][y2]
		#				 kn2 = vk_mat[x1][y2] * vk_mat[y1][x2]
		#				 kernel += kn1 + kn2

	return kernel


def wrapper_sp_do(ds_attrs, node_label, node_kernels, itr):
	i = itr[0]
	j = itr[1]
	return i, j, spkernel_do(G_gn[i], G_gn[j], ds_attrs, node_label, node_kernels)

#def wrapper_sp_do(ds_attrs, node_label, node_kernels, itr_item):
#	g1 = itr_item[0][0]
#	g2 = itr_item[0][1]
#	i = itr_item[1][0]
#	j = itr_item[1][1]
#	return i, j, spkernel_do(g1, g2, ds_attrs, node_label, node_kernels)


def wrapper_getSPGraph(weight, itr_item):
	g = itr_item[0]
	i = itr_item[1]
	return i, getSPGraph(g, edge_weight=weight)
	# return i, nx.floyd_warshall_numpy(g, weight=weight)
