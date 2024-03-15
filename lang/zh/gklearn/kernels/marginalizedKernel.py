"""
@author: linlin

@references:

	[1] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between 
	labeled graphs. In Proceedings of the 20th International Conference on 
	Machine Learning, Washington, DC, United States, 2003.

	[2] Pierre Mahé, Nobuhisa Ueda, Tatsuya Akutsu, Jean-Luc Perret, and 
	Jean-Philippe Vert. Extensions of marginalized graph kernels. In 
	Proceedings of the twenty-first international conference on Machine 
	learning, page 70. ACM, 2004.
"""

import sys
import time
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
tqdm.monitor_interval = 0
#import traceback

import networkx as nx
import numpy as np

from gklearn.utils.kernels import deltakernel
from gklearn.utils.utils import untotterTransformation
from gklearn.utils.graphdataset import get_dataset_attributes
from gklearn.utils.parallel import parallel_gm


def marginalizedkernel(*args,
					   node_label='atom',
					   edge_label='bond_type',
					   p_quit=0.5,
					   n_iteration=20,
					   remove_totters=False,
					   n_jobs=None,
					   chunksize=None,
					   verbose=True):
	"""Compute marginalized graph kernels between graphs.

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

	p_quit : integer
		The termination probability in the random walks generating step.

	n_iteration : integer
		Time of iterations to compute R_inf.

	remove_totters : boolean
		Whether to remove totterings by method introduced in [2]. The default 
		value is False.

	n_jobs : int
		Number of jobs for parallelization.   

	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the marginalized kernel between
		2 praphs.
	"""
	# pre-process
	n_iteration = int(n_iteration)
	Gn = args[0][:] if len(args) == 1 else [args[0].copy(), args[1].copy()]
	Gn = [g.copy() for g in Gn]
	
	ds_attrs = get_dataset_attributes(
		Gn,
		attr_names=['node_labeled', 'edge_labeled', 'is_directed'],
		node_label=node_label, edge_label=edge_label)
	if not ds_attrs['node_labeled'] or node_label is None:
		node_label = 'atom'
		for G in Gn:
			nx.set_node_attributes(G, '0', 'atom')
	if not ds_attrs['edge_labeled'] or edge_label is None:
		edge_label = 'bond_type'
		for G in Gn:
			nx.set_edge_attributes(G, '0', 'bond_type')

	start_time = time.time()
	
	if remove_totters:
		# ---- use pool.imap_unordered to parallel and track progress. ----
		pool = Pool(n_jobs)
		untotter_partial = partial(wrapper_untotter, Gn, node_label, edge_label)
		if chunksize is None:
			if len(Gn) < 100 * n_jobs:
				chunksize = int(len(Gn) / n_jobs) + 1
			else:
				chunksize = 100
		for i, g in tqdm(
				pool.imap_unordered(
					untotter_partial, range(0, len(Gn)), chunksize),
				desc='removing tottering',
				file=sys.stdout):
			Gn[i] = g
		pool.close()
		pool.join()

#		# ---- direct running, normally use single CPU core. ----
#		Gn = [
#			untotterTransformation(G, node_label, edge_label)
#			for G in tqdm(Gn, desc='removing tottering', file=sys.stdout)
#		]

	Kmatrix = np.zeros((len(Gn), len(Gn)))

	# ---- use pool.imap_unordered to parallel and track progress. ----
	def init_worker(gn_toshare):
				global G_gn
				G_gn = gn_toshare
	do_partial = partial(wrapper_marg_do, node_label, edge_label,
						 p_quit, n_iteration)   
	parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
				glbv=(Gn,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


#	# ---- direct running, normally use single CPU core. ----
##	pbar = tqdm(
##		total=(1 + len(Gn)) * len(Gn) / 2,
##		desc='Computing kernels',
##		file=sys.stdout)
#	for i in range(0, len(Gn)):
#		for j in range(i, len(Gn)):
##			print(i, j)
#			Kmatrix[i][j] = _marginalizedkernel_do(Gn[i], Gn[j], node_label,
#												   edge_label, p_quit, n_iteration)
#			Kmatrix[j][i] = Kmatrix[i][j]
##			pbar.update(1)

	run_time = time.time() - start_time
	if verbose:
		print("\n --- marginalized kernel matrix of size %d built in %s seconds ---"
			  % (len(Gn), run_time))

	return Kmatrix, run_time


def _marginalizedkernel_do(g1, g2, node_label, edge_label, p_quit, n_iteration):
	"""Compute marginalized graph kernel between 2 graphs.

	Parameters
	----------
	G1, G2 : NetworkX graphs
		2 graphs between which the kernel is computed.
	node_label : string
		node attribute used as label.
	edge_label : string
		edge attribute used as label.
	p_quit : integer
		the termination probability in the random walks generating step.
	n_iteration : integer
		time of iterations to compute R_inf.

	Return
	------
	kernel : float
		Marginalized Kernel between 2 graphs.
	"""
	# init parameters
	kernel = 0
	num_nodes_G1 = nx.number_of_nodes(g1)
	num_nodes_G2 = nx.number_of_nodes(g2)
	# the initial probability distribution in the random walks generating step
	# (uniform distribution over |G|)
	p_init_G1 = 1 / num_nodes_G1
	p_init_G2 = 1 / num_nodes_G2

	q = p_quit * p_quit
	r1 = q

#	# initial R_inf
#	# matrix to save all the R_inf for all pairs of nodes
#	R_inf = np.zeros([num_nodes_G1, num_nodes_G2])
#
#	# Compute R_inf with a simple interative method
#	for i in range(1, n_iteration):
#		R_inf_new = np.zeros([num_nodes_G1, num_nodes_G2])
#		R_inf_new.fill(r1)
#
#		# Compute R_inf for each pair of nodes
#		for node1 in g1.nodes(data=True):
#			neighbor_n1 = g1[node1[0]]
#			# the transition probability distribution in the random walks
#			# generating step (uniform distribution over the vertices adjacent
#			# to the current vertex)
#			if len(neighbor_n1) > 0:
#				p_trans_n1 = (1 - p_quit) / len(neighbor_n1)
#				for node2 in g2.nodes(data=True):
#					neighbor_n2 = g2[node2[0]]
#					if len(neighbor_n2) > 0:
#						p_trans_n2 = (1 - p_quit) / len(neighbor_n2)
#		
#						for neighbor1 in neighbor_n1:
#							for neighbor2 in neighbor_n2:
#								t = p_trans_n1 * p_trans_n2 * \
#									deltakernel(g1.node[neighbor1][node_label],
#												g2.node[neighbor2][node_label]) * \
#									deltakernel(
#										neighbor_n1[neighbor1][edge_label],
#										neighbor_n2[neighbor2][edge_label])
#		
#								R_inf_new[node1[0]][node2[0]] += t * R_inf[neighbor1][
#									neighbor2]  # ref [1] equation (8)
#		R_inf[:] = R_inf_new
#
#	# add elements of R_inf up and compute kernel.
#	for node1 in g1.nodes(data=True):
#		for node2 in g2.nodes(data=True):
#			s = p_init_G1 * p_init_G2 * deltakernel(
#				node1[1][node_label], node2[1][node_label])
#			kernel += s * R_inf[node1[0]][node2[0]]  # ref [1] equation (6)
	
	
	R_inf = {} # dict to save all the R_inf for all pairs of nodes
	# initial R_inf, the 1st iteration.
	for node1 in g1.nodes():
		for node2 in g2.nodes():
#			R_inf[(node1[0], node2[0])] = r1
			if len(g1[node1]) > 0:
				if len(g2[node2]) > 0:
					R_inf[(node1, node2)] = r1
				else:
					R_inf[(node1, node2)] = p_quit
			else:
				if len(g2[node2]) > 0:
					R_inf[(node1, node2)] = p_quit
				else:
					R_inf[(node1, node2)] = 1
			
	# compute all transition probability first.
	t_dict = {}
	if n_iteration > 1:
		for node1 in g1.nodes():
			neighbor_n1 = g1[node1]
			# the transition probability distribution in the random walks
			# generating step (uniform distribution over the vertices adjacent
			# to the current vertex)
			if len(neighbor_n1) > 0:
				p_trans_n1 = (1 - p_quit) / len(neighbor_n1)
				for node2 in g2.nodes():
					neighbor_n2 = g2[node2]
					if len(neighbor_n2) > 0:
						p_trans_n2 = (1 - p_quit) / len(neighbor_n2)
						for neighbor1 in neighbor_n1:
							for neighbor2 in neighbor_n2:
								t_dict[(node1, node2, neighbor1, neighbor2)] = \
									p_trans_n1 * p_trans_n2 * \
									deltakernel(g1.nodes[neighbor1][node_label],
												g2.nodes[neighbor2][node_label]) * \
									deltakernel(
										neighbor_n1[neighbor1][edge_label],
										neighbor_n2[neighbor2][edge_label])

	# Compute R_inf with a simple interative method
	for i in range(2, n_iteration + 1):
		R_inf_old = R_inf.copy()

		# Compute R_inf for each pair of nodes
		for node1 in g1.nodes():
			neighbor_n1 = g1[node1]
			# the transition probability distribution in the random walks
			# generating step (uniform distribution over the vertices adjacent
			# to the current vertex)
			if len(neighbor_n1) > 0:
				for node2 in g2.nodes():
					neighbor_n2 = g2[node2]
					if len(neighbor_n2) > 0:   
						R_inf[(node1, node2)] = r1
						for neighbor1 in neighbor_n1:
							for neighbor2 in neighbor_n2:
								R_inf[(node1, node2)] += \
									(t_dict[(node1, node2, neighbor1, neighbor2)] * \
									R_inf_old[(neighbor1, neighbor2)])  # ref [1] equation (8)

	# add elements of R_inf up and compute kernel.
	for (n1, n2), value in R_inf.items():
		s = p_init_G1 * p_init_G2 * deltakernel(
				g1.nodes[n1][node_label], g2.nodes[n2][node_label])
		kernel += s * value  # ref [1] equation (6)

	return kernel
		
		
def wrapper_marg_do(node_label, edge_label, p_quit, n_iteration, itr):
	i= itr[0]
	j = itr[1]
	return i, j, _marginalizedkernel_do(G_gn[i], G_gn[j], node_label, edge_label, p_quit, n_iteration)
	

def wrapper_untotter(Gn, node_label, edge_label, i):
	return i, untotterTransformation(Gn[i], node_label, edge_label)
