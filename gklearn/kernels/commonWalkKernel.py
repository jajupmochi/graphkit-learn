"""
@author: linlin

@references: 

	[1] Thomas Gärtner, Peter Flach, and Stefan Wrobel. On graph kernels: 
	Hardness results and efficient alternatives. Learning Theory and Kernel
	Machines, pages 129–143, 2003.
"""

import sys
import time
from collections import Counter
from functools import partial

import networkx as nx
import numpy as np

from gklearn.utils.utils import direct_product
from gklearn.utils.graphdataset import get_dataset_attributes
from gklearn.utils.parallel import parallel_gm


def commonwalkkernel(*args,
					 node_label='atom',
					 edge_label='bond_type',
#					 n=None,
					 weight=1,
					 compute_method=None,
					 n_jobs=None,
					 chunksize=None,
					 verbose=True):
	"""Compute common walk graph kernels between graphs.

	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.
	
	G1, G2 : NetworkX graphs
		Two graphs between which the kernel is computed.
	node_label : string
		Node attribute used as symbolic label. The default node label is 'atom'.
	edge_label : string
		Edge attribute used as symbolic label. The default edge label is 'bond_type'.
	weight: integer
		Weight coefficient of different lengths of walks, which represents beta
		in 'exp' method and gamma in 'geo'.
	compute_method : string
		Method used to compute walk kernel. The Following choices are 
		available:

		'exp': method based on exponential serials applied on the direct 
		product graph, as shown in reference [1]. The time complexity is O(n^6) 
		for graphs with n vertices.

		'geo': method based on geometric serials applied on the direct product 
		graph, as shown in reference [1]. The time complexity is O(n^6) for 
		graphs with n vertices.

	n_jobs : int
		Number of jobs for parallelization. 

	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is a common walk kernel between 2 
		graphs.
	"""
#	n : integer
#		Longest length of walks. Only useful when applying the 'brute' method.
#		'brute': brute force, simply search for all walks and compare them.
	compute_method = compute_method.lower()
	# arrange all graphs in a list
	Gn = args[0] if len(args) == 1 else [args[0], args[1]]
	
	# remove graphs with only 1 node, as they do not have adjacency matrices 
	len_gn = len(Gn)
	Gn = [(idx, G) for idx, G in enumerate(Gn) if nx.number_of_nodes(G) != 1]
	idx = [G[0] for G in Gn]
	Gn = [G[1] for G in Gn]
	if len(Gn) != len_gn:
		if verbose:
			print('\n %d graphs are removed as they have only 1 node.\n' %
				  (len_gn - len(Gn)))
		
	ds_attrs = get_dataset_attributes(
		Gn,
		attr_names=['node_labeled', 'edge_labeled', 'is_directed'],
		node_label=node_label, edge_label=edge_label)
	if not ds_attrs['node_labeled']:
		for G in Gn:
			nx.set_node_attributes(G, '0', 'atom')
	if not ds_attrs['edge_labeled']:
		for G in Gn:
			nx.set_edge_attributes(G, '0', 'bond_type')
	if not ds_attrs['is_directed']:  #  convert
		Gn = [G.to_directed() for G in Gn]

	start_time = time.time()
	
	Kmatrix = np.zeros((len(Gn), len(Gn)))

	# ---- use pool.imap_unordered to parallel and track progress. ----
	def init_worker(gn_toshare):
		global G_gn
		G_gn = gn_toshare
	# direct product graph method - exponential
	if compute_method == 'exp':
		do_partial = partial(wrapper_cw_exp, node_label, edge_label, weight)
	# direct product graph method - geometric
	elif compute_method == 'geo':
		do_partial = partial(wrapper_cw_geo, node_label, edge_label, weight)  
	parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
				glbv=(Gn,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)  
	
	
#	pool = Pool(n_jobs)
#	itr = zip(combinations_with_replacement(Gn, 2),
#			  combinations_with_replacement(range(0, len(Gn)), 2))
#	len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
#	if len_itr < 1000 * n_jobs:
#		chunksize = int(len_itr / n_jobs) + 1
#	else:
#		chunksize = 1000
#
#	# direct product graph method - exponential
#	if compute_method == 'exp':
#		do_partial = partial(wrapper_cw_exp, node_label, edge_label, weight)
#	# direct product graph method - geometric
#	elif compute_method == 'geo':
#		do_partial = partial(wrapper_cw_geo, node_label, edge_label, weight)
#
#	for i, j, kernel in tqdm(
#			pool.imap_unordered(do_partial, itr, chunksize),
#			desc='computing kernels',
#			file=sys.stdout):
#		Kmatrix[i][j] = kernel
#		Kmatrix[j][i] = kernel
#	pool.close()
#	pool.join()


#	# ---- direct running, normally use single CPU core. ----
#	# direct product graph method - exponential
#	itr = combinations_with_replacement(range(0, len(Gn)), 2)
#	if compute_method == 'exp':
#		for i, j in tqdm(itr, desc='Computing kernels', file=sys.stdout):
#			Kmatrix[i][j] = _commonwalkkernel_exp(Gn[i], Gn[j], node_label,
#													  edge_label, weight)
#			Kmatrix[j][i] = Kmatrix[i][j]
#
#	# direct product graph method - geometric
#	elif compute_method == 'geo':
#		for i, j in tqdm(itr, desc='Computing kernels', file=sys.stdout):
#			Kmatrix[i][j] = _commonwalkkernel_geo(Gn[i], Gn[j], node_label,
#													  edge_label, weight)
#			Kmatrix[j][i] = Kmatrix[i][j]


#	# search all paths use brute force.
#	elif compute_method == 'brute':
#		n = int(n)
#		# get all paths of all graphs before computing kernels to save time, but this may cost a lot of memory for large dataset.
#		all_walks = [
#			find_all_walks_until_length(Gn[i], n, node_label, edge_label)
#				for i in range(0, len(Gn))
#		]
#
#		for i in range(0, len(Gn)):
#			for j in range(i, len(Gn)):
#				Kmatrix[i][j] = _commonwalkkernel_brute(
#					all_walks[i],
#					all_walks[j],
#					node_label=node_label,
#					edge_label=edge_label)
#				Kmatrix[j][i] = Kmatrix[i][j]

	run_time = time.time() - start_time
	if verbose:
		print("\n --- kernel matrix of common walk kernel of size %d built in %s seconds ---"
			  % (len(Gn), run_time))

	return Kmatrix, run_time, idx


def _commonwalkkernel_exp(g1, g2, node_label, edge_label, beta):
	"""Compute walk graph kernels up to n between 2 graphs using exponential 
	series.

	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.
	node_label : string
		Node attribute used as label.
	edge_label : string
		Edge attribute used as label.
	beta : integer
		Weight.
	ij : tuple of integer
		Index of graphs between which the kernel is computed.

	Return
	------
	kernel : float
		The common walk Kernel between 2 graphs.
	"""

	# get tensor product / direct product
	gp = direct_product(g1, g2, node_label, edge_label)
	# return 0 if the direct product graph have no more than 1 node.
	if nx.number_of_nodes(gp) < 2:
		return 0
	A = nx.adjacency_matrix(gp).todense()
	# print(A)

	# from matplotlib import pyplot as plt
	# nx.draw_networkx(G1)
	# plt.show()
	# nx.draw_networkx(G2)
	# plt.show()
	# nx.draw_networkx(gp)
	# plt.show()
	# print(G1.nodes(data=True))
	# print(G2.nodes(data=True))
	# print(gp.nodes(data=True))
	# print(gp.edges(data=True))

	ew, ev = np.linalg.eig(A)
	# print('ew: ', ew)
	# print(ev)
	# T = np.matrix(ev)
	# print('T: ', T)
	# T = ev.I
	D = np.zeros((len(ew), len(ew)))
	for i in range(len(ew)):
		D[i][i] = np.exp(beta * ew[i])
		# print('D: ', D)
	# print('hshs: ', T.I * D * T)

	# print(np.exp(-2))
	# print(D)
	# print(np.exp(weight * D))
	# print(ev)
	# print(np.linalg.inv(ev))
	exp_D = ev * D * ev.T
	# print(exp_D)
	# print(np.exp(weight * A))
	# print('-------')

	return exp_D.sum()


def wrapper_cw_exp(node_label, edge_label, beta, itr):
	i = itr[0]
	j = itr[1]
	return i, j, _commonwalkkernel_exp(G_gn[i], G_gn[j], node_label, edge_label, beta)


def _commonwalkkernel_geo(g1, g2, node_label, edge_label, gamma):
	"""Compute common walk graph kernels up to n between 2 graphs using 
	geometric series.

	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.
	node_label : string
		Node attribute used as label.
	edge_label : string
		Edge attribute used as label.
	gamma: integer
		Weight.
	ij : tuple of integer
		Index of graphs between which the kernel is computed.

	Return
	------
	kernel : float
		The common walk Kernel between 2 graphs.
	"""
	# get tensor product / direct product
	gp = direct_product(g1, g2, node_label, edge_label)
	# return 0 if the direct product graph have no more than 1 node.
	if nx.number_of_nodes(gp) < 2:
		return 0
	A = nx.adjacency_matrix(gp).todense()
	mat = np.identity(len(A)) - gamma * A
#	try:
	return mat.I.sum()
#	except np.linalg.LinAlgError:
#		return np.nan
	
	
def wrapper_cw_geo(node_label, edge_label, gama, itr):
	i = itr[0]
	j = itr[1]
	return i, j, _commonwalkkernel_geo(G_gn[i], G_gn[j], node_label, edge_label, gama)


def _commonwalkkernel_brute(walks1,
							walks2,
							node_label='atom',
							edge_label='bond_type',
							labeled=True):
	"""Compute walk graph kernels up to n between 2 graphs.

	Parameters
	----------
	walks1, walks2 : list
		List of walks in 2 graphs, where for unlabeled graphs, each walk is 
		represented by a list of nodes; while for labeled graphs, each walk is 
		represented by a string consists of labels of nodes and edges on that 
		walk.
	node_label : string
		node attribute used as label. The default node label is atom.
	edge_label : string
		edge attribute used as label. The default edge label is bond_type.
	labeled : boolean
		Whether the graphs are labeled. The default is True.

	Return
	------
	kernel : float
		Treelet Kernel between 2 graphs.
	"""
	counts_walks1 = dict(Counter(walks1))
	counts_walks2 = dict(Counter(walks2))
	all_walks = list(set(walks1 + walks2))

	vector1 = [(counts_walks1[walk] if walk in walks1 else 0)
			   for walk in all_walks]
	vector2 = [(counts_walks2[walk] if walk in walks2 else 0)
			   for walk in all_walks]
	kernel = np.dot(vector1, vector2)

	return kernel


# this method find walks repetively, it could be faster.
def find_all_walks_until_length(G,
								length,
								node_label='atom',
								edge_label='bond_type',
								labeled=True):
	"""Find all walks with a certain maximum length in a graph. 
	A recursive depth first search is applied.

	Parameters
	----------
	G : NetworkX graphs
		The graph in which walks are searched.
	length : integer
		The maximum length of walks.
	node_label : string
		node attribute used as label. The default node label is atom.
	edge_label : string
		edge attribute used as label. The default edge label is bond_type.
	labeled : boolean
		Whether the graphs are labeled. The default is True.

	Return
	------
	walk : list
		List of walks retrieved, where for unlabeled graphs, each walk is 
		represented by a list of nodes; while for labeled graphs, each walk 
		is represented by a string consists of labels of nodes and edges on 
		that walk.
	"""
	all_walks = []
	# @todo: in this way, the time complexity is close to N(d^n+d^(n+1)+...+1), which could be optimized to O(Nd^n)
	for i in range(0, length + 1):
		new_walks = find_all_walks(G, i)
		if new_walks == []:
			break
		all_walks.extend(new_walks)

	if labeled == True:  # convert paths to strings
		walk_strs = []
		for walk in all_walks:
			strlist = [
				G.node[node][node_label] +
				G[node][walk[walk.index(node) + 1]][edge_label]
				for node in walk[:-1]
			]
			walk_strs.append(''.join(strlist) + G.node[walk[-1]][node_label])

		return walk_strs

	return all_walks


def find_walks(G, source_node, length):
	"""Find all walks with a certain length those start from a source node. A 
	recursive depth first search is applied.

	Parameters
	----------
	G : NetworkX graphs
		The graph in which walks are searched.
	source_node : integer
		The number of the node from where all walks start.
	length : integer
		The length of walks.

	Return
	------
	walk : list of list
		List of walks retrieved, where each walk is represented by a list of 
		nodes.
	"""
	return [[source_node]] if length == 0 else \
		[[source_node] + walk for neighbor in G[source_node]
		 for walk in find_walks(G, neighbor, length - 1)]


def find_all_walks(G, length):
	"""Find all walks with a certain length in a graph. A recursive depth first
	search is applied.

	Parameters
	----------
	G : NetworkX graphs
		The graph in which walks are searched.
	length : integer
		The length of walks.

	Return
	------
	walk : list of list
		List of walks retrieved, where each walk is represented by a list of 
		nodes.
	"""
	all_walks = []
	for node in G:
		all_walks.extend(find_walks(G, node, length))

	# The following process is not carried out according to the original article
	# all_paths_r = [ path[::-1] for path in all_paths ]

	# # For each path, two presentation are retrieved from its two extremities. Remove one of them.
	# for idx, path in enumerate(all_paths[:-1]):
	#	 for path2 in all_paths_r[idx+1::]:
	#		 if path == path2:
	#			 all_paths[idx] = []
	#			 break

	# return list(filter(lambda a: a != [], all_paths))
	return all_walks
