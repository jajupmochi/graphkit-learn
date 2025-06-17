#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:06:22 2020

@author: ljia
"""
import numpy as np
from itertools import combinations
import multiprocessing
from multiprocessing import Pool
from functools import partial
import sys
from tqdm import tqdm
import networkx as nx
from gklearn.ged.env import GEDEnv


def compute_ged(g1, g2, options):
	from gklearn.gedlib import librariesImport, gedlibpy

	ged_env = gedlibpy.GEDEnv()
	ged_env.set_edit_cost(options['edit_cost'], edit_cost_constant=options['edit_cost_constants'])
	ged_env.add_nx_graph(g1, '')
	ged_env.add_nx_graph(g2, '')
	listID = ged_env.get_all_graph_ids()	
	ged_env.init(init_type=options['init_option'])
	ged_env.set_method(options['method'], ged_options_to_string(options))
	ged_env.init_method()

	g = listID[0]
	h = listID[1]
	ged_env.run_method(g, h)
	pi_forward = ged_env.get_forward_map(g, h)
	pi_backward = ged_env.get_backward_map(g, h)
	upper = ged_env.get_upper_bound(g, h)	
	dis = upper
			
	# make the map label correct (label remove map as np.inf)
	nodes1 = [n for n in g1.nodes()]
	nodes2 = [n for n in g2.nodes()]
	nb1 = nx.number_of_nodes(g1)
	nb2 = nx.number_of_nodes(g2)
	pi_forward = [nodes2[pi] if pi < nb2 else np.inf for pi in pi_forward]
	pi_backward = [nodes1[pi] if pi < nb1 else np.inf for pi in pi_backward]
#		print(pi_forward)

	return dis, pi_forward, pi_backward


def compute_geds_cml(graphs, options={}, sort=True, parallel=False, verbose=True):

	# initialize ged env.
	ged_env = GEDEnv()
	ged_env.set_edit_cost(options['edit_cost'], edit_cost_constants=options['edit_cost_constants'])
	for g in graphs:
		ged_env.add_nx_graph(g, '')
	listID = ged_env.get_all_graph_ids()
	
	node_labels = ged_env.get_all_node_labels()
	edge_labels =  ged_env.get_all_edge_labels()
	node_label_costs = label_costs_to_matrix(options['node_label_costs'], len(node_labels)) if 'node_label_costs' in options else None
	edge_label_costs = label_costs_to_matrix(options['edge_label_costs'], len(edge_labels)) if 'edge_label_costs' in options else None
	ged_env.set_label_costs(node_label_costs, edge_label_costs)
	ged_env.init(init_type=options['init_option'])
	if parallel:
		options['threads'] = 1
	ged_env.set_method(options['method'], options)
	ged_env.init_method()

	# compute ged.
	# options used to compute numbers of edit operations.
	if node_label_costs is None and edge_label_costs is None:
		neo_options = {'edit_cost': options['edit_cost'],
				 'is_cml': False,
				 'node_labels': options['node_labels'], 'edge_labels': options['edge_labels'], 
				 'node_attrs': options['node_attrs'], 'edge_attrs': options['edge_attrs']}
	else:
		neo_options = {'edit_cost': options['edit_cost'],
				 'is_cml': True,
				 'node_labels': node_labels,
				 'edge_labels': edge_labels}
	ged_mat = np.zeros((len(graphs), len(graphs)))
	if parallel:
		len_itr = int(len(graphs) * (len(graphs) - 1) / 2)
		ged_vec = [0 for i in range(len_itr)]
		n_edit_operations = [0 for i in range(len_itr)]
		itr = combinations(range(0, len(graphs)), 2)
		n_jobs = multiprocessing.cpu_count()
		if len_itr < 100 * n_jobs:
			chunksize = int(len_itr / n_jobs) + 1
		else:
			chunksize = 100
		def init_worker(graphs_toshare, ged_env_toshare, listID_toshare):
			global G_graphs, G_ged_env, G_listID
			G_graphs = graphs_toshare
			G_ged_env = ged_env_toshare
			G_listID = listID_toshare
		do_partial = partial(_wrapper_compute_ged_parallel, neo_options, sort)
		pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(graphs, ged_env, listID))
		if verbose:
			iterator = tqdm(pool.imap_unordered(do_partial, itr, chunksize),
						desc='computing GEDs', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(do_partial, itr, chunksize)
#		iterator = pool.imap_unordered(do_partial, itr, chunksize)
		for i, j, dis, n_eo_tmp in iterator:
			idx_itr = int(len(graphs) * i + j - (i + 1) * (i + 2) / 2)
			ged_vec[idx_itr] = dis
			ged_mat[i][j] = dis
			ged_mat[j][i] = dis
			n_edit_operations[idx_itr] = n_eo_tmp
#			print('\n-------------------------------------------')
#			print(i, j, idx_itr, dis)
		pool.close()
		pool.join()
		
	else:
		ged_vec = []
		n_edit_operations = []
		if verbose:
			iterator = tqdm(range(len(graphs)), desc='computing GEDs', file=sys.stdout)
		else:
			iterator = range(len(graphs))
		for i in iterator:
#		for i in range(len(graphs)):
			for j in range(i + 1, len(graphs)):
				if nx.number_of_nodes(graphs[i]) <= nx.number_of_nodes(graphs[j]) or not sort:
					dis, pi_forward, pi_backward = _compute_ged(ged_env, listID[i], listID[j], graphs[i], graphs[j])
				else:
					dis, pi_backward, pi_forward = _compute_ged(ged_env, listID[j], listID[i], graphs[j], graphs[i])
				ged_vec.append(dis)
				ged_mat[i][j] = dis
				ged_mat[j][i] = dis
				n_eo_tmp = get_nb_edit_operations(graphs[i], graphs[j], pi_forward, pi_backward, 	**neo_options)
				n_edit_operations.append(n_eo_tmp)

	return ged_vec, ged_mat, n_edit_operations


def compute_geds(graphs, options={}, sort=True, parallel=False, verbose=True):
	from gklearn.gedlib import librariesImport, gedlibpy

	# initialize ged env.
	ged_env = gedlibpy.GEDEnv()
	ged_env.set_edit_cost(options['edit_cost'], edit_cost_constant=options['edit_cost_constants'])
	for g in graphs:
		ged_env.add_nx_graph(g, '')
	listID = ged_env.get_all_graph_ids()	
	ged_env.init()
	if parallel:
		options['threads'] = 1
	ged_env.set_method(options['method'], ged_options_to_string(options))
	ged_env.init_method()

	# compute ged.
	neo_options = {'edit_cost': options['edit_cost'],
				'node_labels': options['node_labels'], 'edge_labels': options['edge_labels'], 
				'node_attrs': options['node_attrs'], 'edge_attrs': options['edge_attrs']}
	ged_mat = np.zeros((len(graphs), len(graphs)))
	if parallel:
		len_itr = int(len(graphs) * (len(graphs) - 1) / 2)
		ged_vec = [0 for i in range(len_itr)]
		n_edit_operations = [0 for i in range(len_itr)]
		itr = combinations(range(0, len(graphs)), 2)
		n_jobs = multiprocessing.cpu_count()
		if len_itr < 100 * n_jobs:
			chunksize = int(len_itr / n_jobs) + 1
		else:
			chunksize = 100
		def init_worker(graphs_toshare, ged_env_toshare, listID_toshare):
			global G_graphs, G_ged_env, G_listID
			G_graphs = graphs_toshare
			G_ged_env = ged_env_toshare
			G_listID = listID_toshare
		do_partial = partial(_wrapper_compute_ged_parallel, neo_options, sort)
		pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(graphs, ged_env, listID))
		if verbose:
			iterator = tqdm(pool.imap_unordered(do_partial, itr, chunksize),
						desc='computing GEDs', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(do_partial, itr, chunksize)
#		iterator = pool.imap_unordered(do_partial, itr, chunksize)
		for i, j, dis, n_eo_tmp in iterator:
			idx_itr = int(len(graphs) * i + j - (i + 1) * (i + 2) / 2)
			ged_vec[idx_itr] = dis
			ged_mat[i][j] = dis
			ged_mat[j][i] = dis
			n_edit_operations[idx_itr] = n_eo_tmp
#			print('\n-------------------------------------------')
#			print(i, j, idx_itr, dis)
		pool.close()
		pool.join()
		
	else:
		ged_vec = []
		n_edit_operations = []
		if verbose:
			iterator = tqdm(range(len(graphs)), desc='computing GEDs', file=sys.stdout)
		else:
			iterator = range(len(graphs))
		for i in iterator:
#		for i in range(len(graphs)):
			for j in range(i + 1, len(graphs)):
				if nx.number_of_nodes(graphs[i]) <= nx.number_of_nodes(graphs[j]) or not sort:
					dis, pi_forward, pi_backward = _compute_ged(ged_env, listID[i], listID[j], graphs[i], graphs[j])
				else:
					dis, pi_backward, pi_forward = _compute_ged(ged_env, listID[j], listID[i], graphs[j], graphs[i])
				ged_vec.append(dis)
				ged_mat[i][j] = dis
				ged_mat[j][i] = dis
				n_eo_tmp = get_nb_edit_operations(graphs[i], graphs[j], pi_forward, pi_backward, 	**neo_options)
				n_edit_operations.append(n_eo_tmp)

	return ged_vec, ged_mat, n_edit_operations


def _wrapper_compute_ged_parallel(options, sort, itr):
	i = itr[0]
	j = itr[1]
	dis, n_eo_tmp = _compute_ged_parallel(G_ged_env, G_listID[i], G_listID[j], G_graphs[i], G_graphs[j], options, sort)
	return i, j, dis, n_eo_tmp


def _compute_ged_parallel(env, gid1, gid2, g1, g2, options, sort):
	if nx.number_of_nodes(g1) <= nx.number_of_nodes(g2) or not sort:
		dis, pi_forward, pi_backward = _compute_ged(env, gid1, gid2, g1, g2)
	else:
		dis, pi_backward, pi_forward = _compute_ged(env, gid2, gid1, g2, g1)
	n_eo_tmp = get_nb_edit_operations(g1, g2, pi_forward, pi_backward, **options) # [0,0,0,0,0,0]
	return dis, n_eo_tmp


def _compute_ged(env, gid1, gid2, g1, g2):
	env.run_method(gid1, gid2)
	pi_forward = env.get_forward_map(gid1, gid2)
	pi_backward = env.get_backward_map(gid1, gid2)
	upper = env.get_upper_bound(gid1, gid2)	
	dis = upper
			
	# make the map label correct (label remove map as np.inf)
	nodes1 = [n for n in g1.nodes()]
	nodes2 = [n for n in g2.nodes()]
	nb1 = nx.number_of_nodes(g1)
	nb2 = nx.number_of_nodes(g2)
	pi_forward = [nodes2[pi] if pi < nb2 else np.inf for pi in pi_forward]
	pi_backward = [nodes1[pi] if pi < nb1 else np.inf for pi in pi_backward]

	return dis, pi_forward, pi_backward


def label_costs_to_matrix(costs, nb_labels):
	"""Reform a label cost vector to a matrix.

	Parameters
	----------
	costs : numpy.array
		The vector containing costs between labels, in the order of node insertion costs, node deletion costs, node substitition costs, edge insertion costs, edge deletion costs, edge substitition costs.
	nb_labels : integer
		Number of labels.

	Returns
	-------
	cost_matrix : numpy.array. 
		The reformed label cost matrix of size (nb_labels, nb_labels). Each row/column of cost_matrix corresponds to a label, and the first label is the dummy label. This is the same setting as in GEDData.
	"""
	# Initialize label cost matrix.
	cost_matrix = np.zeros((nb_labels + 1, nb_labels + 1))
	i = 0
	# Costs of insertions.
	for col in range(1, nb_labels + 1):
		cost_matrix[0, col] = costs[i]
		i += 1
	# Costs of deletions.
	for row in range(1, nb_labels + 1):
		cost_matrix[row, 0] = costs[i]
		i += 1
	# Costs of substitutions.				
	for row in range(1, nb_labels + 1):
		for col in range(row + 1, nb_labels + 1):
			cost_matrix[row, col] = costs[i]
			cost_matrix[col, row] = costs[i]
			i += 1
			
	return cost_matrix


def get_nb_edit_operations(g1, g2, forward_map, backward_map, edit_cost=None, is_cml=False, **kwargs):
	if is_cml:
		if edit_cost == 'CONSTANT':
			node_labels = kwargs.get('node_labels', [])
			edge_labels = kwargs.get('edge_labels', [])
			return get_nb_edit_operations_symbolic_cml(g1, g2, forward_map, backward_map,
											  node_labels=node_labels, edge_labels=edge_labels)
		else: 
			raise Exception('Edit cost "', edit_cost, '" is not supported.')
	else:
		if edit_cost == 'LETTER' or edit_cost == 'LETTER2':
			return get_nb_edit_operations_letter(g1, g2, forward_map, backward_map)
		elif edit_cost == 'NON_SYMBOLIC':
			node_attrs = kwargs.get('node_attrs', [])
			edge_attrs = kwargs.get('edge_attrs', [])
			return get_nb_edit_operations_nonsymbolic(g1, g2, forward_map, backward_map, 
												node_attrs=node_attrs, edge_attrs=edge_attrs)
		elif edit_cost == 'CONSTANT':
			node_labels = kwargs.get('node_labels', [])
			edge_labels = kwargs.get('edge_labels', [])
			return get_nb_edit_operations_symbolic(g1, g2, forward_map, backward_map, 
											 node_labels=node_labels, edge_labels=edge_labels)
		else: 
			return get_nb_edit_operations_symbolic(g1, g2, forward_map, backward_map)	
		
		
def get_nb_edit_operations_symbolic_cml(g1, g2, forward_map, backward_map, 
										node_labels=[], edge_labels=[]):
	"""Compute times that edit operations are used in an edit path for symbolic-labeled graphs, where the costs are different for each pair of nodes.
	
	Returns
	-------
	list
		A vector of numbers of times that costs bewteen labels are used in an edit path, formed in the order of node insertion costs, node deletion costs, node substitition costs, edge insertion costs, edge deletion costs, edge substitition costs. The dummy label is the first label, and the self label costs are not included.
	"""
	# Initialize.
	nb_ops_node = np.zeros((1 + len(node_labels), 1 + len(node_labels)))
	nb_ops_edge = np.zeros((1 + len(edge_labels), 1 + len(edge_labels)))
	
	# For nodes.
	nodes1 = [n for n in g1.nodes()]
	for i, map_i in enumerate(forward_map):
		label1 = tuple(g1.nodes[nodes1[i]].items()) # @todo: order and faster
		idx_label1 = node_labels.index(label1) # @todo: faster
		if map_i == np.inf: # deletions.
			nb_ops_node[idx_label1 + 1, 0] += 1
		else: # substitutions.
			label2 = tuple(g2.nodes[map_i].items())
			if label1 != label2:
				idx_label2 = node_labels.index(label2) # @todo: faster
				nb_ops_node[idx_label1 + 1, idx_label2 + 1] += 1
	# insertions.
	nodes2 = [n for n in g2.nodes()]
	for i, map_i in enumerate(backward_map):
		if map_i == np.inf:
			label = tuple(g2.nodes[nodes2[i]].items())
			idx_label = node_labels.index(label) # @todo: faster
			nb_ops_node[0, idx_label + 1] += 1
			
	# For edges.
	edges1 = [e for e in g1.edges()]
	edges2_marked = []
	for nf1, nt1 in edges1:
		label1 = tuple(g1.edges[(nf1, nt1)].items())
		idx_label1 = edge_labels.index(label1) # @todo: faster
		idxf1 = nodes1.index(nf1) # @todo: faster
		idxt1 = nodes1.index(nt1) # @todo: faster
		# At least one of the nodes is removed, thus the edge is removed.
		if forward_map[idxf1] == np.inf or forward_map[idxt1] == np.inf:
			nb_ops_edge[idx_label1 + 1, 0] += 1
		# corresponding edge is in g2.
		else:
			nf2, nt2 = forward_map[idxf1], forward_map[idxt1]
			if (nf2, nt2) in g2.edges():
				edges2_marked.append((nf2, nt2))
				# If edge labels are different.
				label2 = tuple(g2.edges[(nf2, nt2)].items())
				if label1 != label2:
					idx_label2 = edge_labels.index(label2) # @todo: faster
					nb_ops_edge[idx_label1 + 1, idx_label2 + 1] += 1		
			# Switch nf2 and nt2, for directed graphs.
			elif (nt2, nf2) in g2.edges():
				edges2_marked.append((nt2, nf2))
				# If edge labels are different.
				label2 = tuple(g2.edges[(nt2, nf2)].items())
				if label1 != label2:
					idx_label2 = edge_labels.index(label2) # @todo: faster
					nb_ops_edge[idx_label1 + 1, idx_label2 + 1] += 1
			# Corresponding nodes are in g2, however the edge is removed.
			else:
				nb_ops_edge[idx_label1 + 1, 0] += 1
	# insertions.
	for nt, nf in g2.edges():
		if (nt, nf) not in edges2_marked and (nf, nt) not in edges2_marked: # @todo: for directed.
			label = tuple(g2.edges[(nt, nf)].items())
			idx_label = edge_labels.index(label) # @todo: faster
			nb_ops_edge[0, idx_label + 1] += 1
			
	# Reform the numbers of edit oeprations into a vector.
	nb_eo_vector = []
	# node insertion.
	for i in range(1, len(nb_ops_node)):
		nb_eo_vector.append(nb_ops_node[0, i])
	# node deletion.
	for i in range(1, len(nb_ops_node)):
		nb_eo_vector.append(nb_ops_node[i, 0])
	# node substitution.
	for i in range(1, len(nb_ops_node)):
		for j in range(i + 1, len(nb_ops_node)):
			nb_eo_vector.append(nb_ops_node[i, j])
	# edge insertion.
	for i in range(1, len(nb_ops_edge)):
		nb_eo_vector.append(nb_ops_edge[0, i])
	# edge deletion.
	for i in range(1, len(nb_ops_edge)):
		nb_eo_vector.append(nb_ops_edge[i, 0])
	# edge substitution.
	for i in range(1, len(nb_ops_edge)):
		for j in range(i + 1, len(nb_ops_edge)):
			nb_eo_vector.append(nb_ops_edge[i, j])
	
	return nb_eo_vector
	

def get_nb_edit_operations_symbolic(g1, g2, forward_map, backward_map,
									node_labels=[], edge_labels=[]):
	"""Compute the number of each edit operations for symbolic-labeled graphs.
	"""
	n_vi = 0
	n_vr = 0
	n_vs = 0
	n_ei = 0
	n_er = 0
	n_es = 0
	
	nodes1 = [n for n in g1.nodes()]
	for i, map_i in enumerate(forward_map):
		if map_i == np.inf:
			n_vr += 1
		else:
			for nl in node_labels:
				label1 = g1.nodes[nodes1[i]][nl]
				label2 = g2.nodes[map_i][nl]
				if label1 != label2:
					n_vs += 1
					break
	for map_i in backward_map:
		if map_i == np.inf:
			n_vi += 1
	
#	idx_nodes1 = range(0, len(node1))
	
	edges1 = [e for e in g1.edges()]
	nb_edges2_cnted = 0
	for n1, n2 in edges1:
		idx1 = nodes1.index(n1)
		idx2 = nodes1.index(n2)
		# one of the nodes is removed, thus the edge is removed.
		if forward_map[idx1] == np.inf or forward_map[idx2] == np.inf:
			n_er += 1
		# corresponding edge is in g2.
		elif (forward_map[idx1], forward_map[idx2]) in g2.edges():
			nb_edges2_cnted += 1
			# edge labels are different.
			for el in edge_labels:
				label1 = g2.edges[((forward_map[idx1], forward_map[idx2]))][el]
				label2 = g1.edges[(n1, n2)][el]
				if label1 != label2:
					n_es += 1
					break
		elif (forward_map[idx2], forward_map[idx1]) in g2.edges():
			nb_edges2_cnted += 1
			# edge labels are different.
			for el in edge_labels:
				label1 = g2.edges[((forward_map[idx2], forward_map[idx1]))][el]
				label2 = g1.edges[(n1, n2)][el]
				if label1 != label2:
					n_es += 1
					break
		# corresponding nodes are in g2, however the edge is removed.
		else:
			n_er += 1
	n_ei = nx.number_of_edges(g2) - nb_edges2_cnted
	
	return n_vi, n_vr, n_vs, n_ei, n_er, n_es


def get_nb_edit_operations_letter(g1, g2, forward_map, backward_map):
	"""Compute the number of each edit operations.
	"""
	n_vi = 0
	n_vr = 0
	n_vs = 0
	sod_vs = 0
	n_ei = 0
	n_er = 0
	
	nodes1 = [n for n in g1.nodes()]
	for i, map_i in enumerate(forward_map):
		if map_i == np.inf:
			n_vr += 1
		else:
			n_vs += 1
			diff_x = float(g1.nodes[nodes1[i]]['x']) - float(g2.nodes[map_i]['x'])
			diff_y = float(g1.nodes[nodes1[i]]['y']) - float(g2.nodes[map_i]['y'])
			sod_vs += np.sqrt(np.square(diff_x) + np.square(diff_y))
	for map_i in backward_map:
		if map_i == np.inf:
			n_vi += 1
	
#	idx_nodes1 = range(0, len(node1))
	
	edges1 = [e for e in g1.edges()]
	nb_edges2_cnted = 0
	for n1, n2 in edges1:
		idx1 = nodes1.index(n1)
		idx2 = nodes1.index(n2)
		# one of the nodes is removed, thus the edge is removed.
		if forward_map[idx1] == np.inf or forward_map[idx2] == np.inf:
			n_er += 1
		# corresponding edge is in g2. Edge label is not considered.
		elif (forward_map[idx1], forward_map[idx2]) in g2.edges() or \
			(forward_map[idx2], forward_map[idx1]) in g2.edges():
				nb_edges2_cnted += 1
		# corresponding nodes are in g2, however the edge is removed.
		else:
			n_er += 1
	n_ei = nx.number_of_edges(g2) - nb_edges2_cnted
	
	return n_vi, n_vr, n_vs, sod_vs, n_ei, n_er


def get_nb_edit_operations_nonsymbolic(g1, g2, forward_map, backward_map,
									   node_attrs=[], edge_attrs=[]):
	"""Compute the number of each edit operations.
	"""
	n_vi = 0
	n_vr = 0
	n_vs = 0
	sod_vs = 0
	n_ei = 0
	n_er = 0
	n_es = 0
	sod_es = 0
	
	nodes1 = [n for n in g1.nodes()]
	for i, map_i in enumerate(forward_map):
		if map_i == np.inf:
			n_vr += 1
		else:
			n_vs += 1
			sum_squares = 0
			for a_name in node_attrs:
				diff = float(g1.nodes[nodes1[i]][a_name]) - float(g2.nodes[map_i][a_name])
				sum_squares += np.square(diff)
			sod_vs += np.sqrt(sum_squares)
	for map_i in backward_map:
		if map_i == np.inf:
			n_vi += 1
	
#	idx_nodes1 = range(0, len(node1))
	
	edges1 = [e for e in g1.edges()]
	for n1, n2 in edges1:
		idx1 = nodes1.index(n1)
		idx2 = nodes1.index(n2)
		n1_g2 = forward_map[idx1]
		n2_g2 = forward_map[idx2]
		# one of the nodes is removed, thus the edge is removed.
		if n1_g2 == np.inf or n2_g2 == np.inf:
			n_er += 1
		# corresponding edge is in g2.
		elif (n1_g2, n2_g2) in g2.edges():
			n_es += 1
			sum_squares = 0
			for a_name in edge_attrs:
				diff = float(g1.edges[n1, n2][a_name]) - float(g2.edges[n1_g2, n2_g2][a_name])
				sum_squares += np.square(diff)
			sod_es += np.sqrt(sum_squares)
		elif (n2_g2, n1_g2) in g2.edges():
			n_es += 1
			sum_squares = 0
			for a_name in edge_attrs:
				diff = float(g1.edges[n2, n1][a_name]) - float(g2.edges[n2_g2, n1_g2][a_name])
				sum_squares += np.square(diff)
			sod_es += np.sqrt(sum_squares)
		# corresponding nodes are in g2, however the edge is removed.
		else:
			n_er += 1
	n_ei = nx.number_of_edges(g2) - n_es
		
	return n_vi, n_vr, sod_vs, n_ei, n_er, sod_es


def ged_options_to_string(options):
	opt_str = ' '
	for key, val in options.items():
		if key == 'initialization_method':
			opt_str += '--initialization-method ' + str(val) + ' '
		elif key == 'initialization_options':
			opt_str += '--initialization-options ' + str(val) + ' '
		elif key == 'lower_bound_method':
			opt_str += '--lower-bound-method ' + str(val) + ' '
		elif key == 'random_substitution_ratio':
			opt_str += '--random-substitution-ratio ' + str(val) + ' '
		elif key == 'initial_solutions':
 			opt_str += '--initial-solutions ' + str(val) + ' '
		elif key == 'ratio_runs_from_initial_solutions':
 			opt_str += '--ratio-runs-from-initial-solutions ' + str(val) + ' '
		elif key == 'threads':
			opt_str += '--threads ' + str(val) + ' '
		elif key == 'num_randpost_loops':
			opt_str += '--num-randpost-loops ' + str(val) + ' '
		elif key == 'max_randpost_retrials':
			opt_str += '--maxrandpost-retrials ' + str(val) + ' '
		elif key == 'randpost_penalty':
			opt_str += '--randpost-penalty ' + str(val) + ' '
		elif key == 'randpost_decay':
			opt_str += '--randpost-decay ' + str(val) + ' '
		elif key == 'log':
			opt_str += '--log ' + str(val) + ' '
		elif key == 'randomness':
			opt_str += '--randomness ' + str(val) + ' '
			
# 		if not isinstance(val, list):
# 			opt_str += '--' + key.replace('_', '-') + ' '
# 			if val == False:
# 	 			val_str = 'FALSE'
# 			else:
# 	 			val_str = str(val)
# 			opt_str += val_str + ' '

	return opt_str