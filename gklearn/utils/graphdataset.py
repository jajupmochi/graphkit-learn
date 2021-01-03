""" Obtain all kinds of attributes of a graph dataset.

This file is for old version of graphkit-learn.
"""


def get_dataset_attributes(Gn,
						   target=None,
						   attr_names=[],
						   node_label=None,
						   edge_label=None):
	"""Returns the structure and property information of the graph dataset Gn.

	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs whose information will be returned.

	target : list
		The list of classification targets corresponding to Gn. Only works for
		classification problems.

	attr_names : list
		List of strings which indicate which informations will be returned. The
		possible choices includes:

		'substructures': sub-structures Gn contains, including 'linear', 'non
	linear' and 'cyclic'.

		'node_labeled': whether vertices have symbolic labels.

		'edge_labeled': whether egdes have symbolic labels.

		'is_directed': whether graphs in Gn are directed.

		'dataset_size': number of graphs in Gn.

		'ave_node_num': average number of vertices of graphs in Gn.

		'min_node_num': minimum number of vertices of graphs in Gn.

		'max_node_num': maximum number of vertices of graphs in Gn.

		'ave_edge_num': average number of edges of graphs in Gn.

		'min_edge_num': minimum number of edges of graphs in Gn.

		'max_edge_num': maximum number of edges of graphs in Gn.

		'ave_node_degree': average vertex degree of graphs in Gn.

		'min_node_degree': minimum vertex degree of graphs in Gn.

		'max_node_degree': maximum vertex degree of graphs in Gn.

		'ave_fill_factor': average fill factor (number_of_edges /
	(number_of_nodes ** 2)) of graphs in Gn.

		'min_fill_factor': minimum fill factor of graphs in Gn.

		'max_fill_factor': maximum fill factor of graphs in Gn.

		'node_label_num': number of symbolic vertex labels.

		'edge_label_num': number of symbolic edge labels.

		'node_attr_dim': number of dimensions of non-symbolic vertex labels.
	Extracted from the 'attributes' attribute of graph nodes.

		'edge_attr_dim': number of dimensions of non-symbolic edge labels.
	Extracted from the 'attributes' attribute of graph edges.

		'class_number': number of classes. Only available for classification problems.

	node_label : string
		Node attribute used as label. The default node label is atom. Mandatory
		when 'node_labeled' or 'node_label_num' is required.

	edge_label : string
		Edge attribute used as label. The default edge label is bond_type.
		Mandatory when 'edge_labeled' or 'edge_label_num' is required.

	Return
	------
	attrs : dict
		Value for each property.
	"""
	import networkx as nx
	import numpy as np

	attrs = {}

	def get_dataset_size(Gn):
		return len(Gn)

	def get_all_node_num(Gn):
		return [nx.number_of_nodes(G) for G in Gn]

	def get_ave_node_num(all_node_num):
		return np.mean(all_node_num)

	def get_min_node_num(all_node_num):
		return np.amin(all_node_num)

	def get_max_node_num(all_node_num):
		return np.amax(all_node_num)

	def get_all_edge_num(Gn):
		return [nx.number_of_edges(G) for G in Gn]

	def get_ave_edge_num(all_edge_num):
		return np.mean(all_edge_num)

	def get_min_edge_num(all_edge_num):
		return np.amin(all_edge_num)

	def get_max_edge_num(all_edge_num):
		return np.amax(all_edge_num)

	def is_node_labeled(Gn):
		return False if node_label is None else True

	def get_node_label_num(Gn):
		nl = set()
		for G in Gn:
			nl = nl | set(nx.get_node_attributes(G, node_label).values())
		return len(nl)

	def is_edge_labeled(Gn):
		return False if edge_label is None else True

	def get_edge_label_num(Gn):
		el = set()
		for G in Gn:
			el = el | set(nx.get_edge_attributes(G, edge_label).values())
		return len(el)

	def is_directed(Gn):
		return nx.is_directed(Gn[0])

	def get_ave_node_degree(Gn):
		return np.mean([np.mean(list(dict(G.degree()).values())) for G in Gn])

	def get_max_node_degree(Gn):
		return np.amax([np.mean(list(dict(G.degree()).values())) for G in Gn])

	def get_min_node_degree(Gn):
		return np.amin([np.mean(list(dict(G.degree()).values())) for G in Gn])

	# get fill factor, the number of non-zero entries in the adjacency matrix.
	def get_ave_fill_factor(Gn):
		return np.mean([nx.number_of_edges(G) / (nx.number_of_nodes(G)
										   * nx.number_of_nodes(G)) for G in Gn])

	def get_max_fill_factor(Gn):
		return np.amax([nx.number_of_edges(G) / (nx.number_of_nodes(G)
										   * nx.number_of_nodes(G)) for G in Gn])

	def get_min_fill_factor(Gn):
		return np.amin([nx.number_of_edges(G) / (nx.number_of_nodes(G)
										   * nx.number_of_nodes(G)) for G in Gn])

	def get_substructures(Gn):
		subs = set()
		for G in Gn:
			degrees = list(dict(G.degree()).values())
			if any(i == 2 for i in degrees):
				subs.add('linear')
			if np.amax(degrees) >= 3:
				subs.add('non linear')
			if 'linear' in subs and 'non linear' in subs:
				break

		if is_directed(Gn):
			for G in Gn:
				if len(list(nx.find_cycle(G))) > 0:
					subs.add('cyclic')
					break
# 		else:
# 			# @todo: this method does not work for big graph with large amount of edges like D&D, try a better way.
# 			upper = np.amin([nx.number_of_edges(G) for G in Gn]) * 2 + 10
# 			for G in Gn:
# 				if (nx.number_of_edges(G) < upper):
# 					cyc = list(nx.simple_cycles(G.to_directed()))
# 					if any(len(i) > 2 for i in cyc):
# 						subs.add('cyclic')
# 						break
# 			if 'cyclic' not in subs:
# 				for G in Gn:
# 					cyc = list(nx.simple_cycles(G.to_directed()))
# 					if any(len(i) > 2 for i in cyc):
# 						subs.add('cyclic')
# 						break

		return subs

	def get_class_num(target):
		return len(set(target))

	def get_node_attr_dim(Gn):
		for G in Gn:
			for n in G.nodes(data=True):
				if 'attributes' in n[1]:
					return len(n[1]['attributes'])
		return 0

	def get_edge_attr_dim(Gn):
		for G in Gn:
			if nx.number_of_edges(G) > 0:
				for e in G.edges(data=True):
					if 'attributes' in e[2]:
						return len(e[2]['attributes'])
		return 0

	if attr_names == []:
		attr_names = [
			'substructures',
			'node_labeled',
			'edge_labeled',
			'is_directed',
			'dataset_size',
			'ave_node_num',
			'min_node_num',
			'max_node_num',
			'ave_edge_num',
			'min_edge_num',
			'max_edge_num',
			'ave_node_degree',
			'min_node_degree',
			'max_node_degree',
			'ave_fill_factor',
			'min_fill_factor',
			'max_fill_factor',
			'node_label_num',
			'edge_label_num',
			'node_attr_dim',
			'edge_attr_dim',
			'class_number',
		]

	# dataset size
	if 'dataset_size' in attr_names:

		attrs.update({'dataset_size': get_dataset_size(Gn)})

	# graph node number
	if any(i in attr_names
		   for i in ['ave_node_num', 'min_node_num', 'max_node_num']):

		all_node_num = get_all_node_num(Gn)

	if 'ave_node_num' in attr_names:

		attrs.update({'ave_node_num': get_ave_node_num(all_node_num)})

	if 'min_node_num' in attr_names:

		attrs.update({'min_node_num': get_min_node_num(all_node_num)})

	if 'max_node_num' in attr_names:

		attrs.update({'max_node_num': get_max_node_num(all_node_num)})

	# graph edge number
	if any(i in attr_names for i in
		   ['ave_edge_num', 'min_edge_num', 'max_edge_num']):

		all_edge_num = get_all_edge_num(Gn)

	if 'ave_edge_num' in attr_names:

		attrs.update({'ave_edge_num': get_ave_edge_num(all_edge_num)})

	if 'max_edge_num' in attr_names:

		attrs.update({'max_edge_num': get_max_edge_num(all_edge_num)})

	if 'min_edge_num' in attr_names:

		attrs.update({'min_edge_num': get_min_edge_num(all_edge_num)})

	# label number
	if any(i in attr_names for i in ['node_labeled', 'node_label_num']):
		is_nl = is_node_labeled(Gn)
		node_label_num = get_node_label_num(Gn)

	if 'node_labeled' in attr_names:
		# graphs are considered node unlabeled if all nodes have the same label.
		attrs.update({'node_labeled': is_nl if node_label_num > 1 else False})

	if 'node_label_num' in attr_names:
		attrs.update({'node_label_num': node_label_num})

	if any(i in attr_names for i in ['edge_labeled', 'edge_label_num']):
		is_el = is_edge_labeled(Gn)
		edge_label_num = get_edge_label_num(Gn)

	if 'edge_labeled' in attr_names:
		# graphs are considered edge unlabeled if all edges have the same label.
		attrs.update({'edge_labeled': is_el if edge_label_num > 1 else False})

	if 'edge_label_num' in attr_names:
		attrs.update({'edge_label_num': edge_label_num})

	if 'is_directed' in attr_names:
		attrs.update({'is_directed': is_directed(Gn)})

	if 'ave_node_degree' in attr_names:
		attrs.update({'ave_node_degree': get_ave_node_degree(Gn)})

	if 'max_node_degree' in attr_names:
		attrs.update({'max_node_degree': get_max_node_degree(Gn)})

	if 'min_node_degree' in attr_names:
		attrs.update({'min_node_degree': get_min_node_degree(Gn)})

	if 'ave_fill_factor' in attr_names:
		attrs.update({'ave_fill_factor': get_ave_fill_factor(Gn)})

	if 'max_fill_factor' in attr_names:
		attrs.update({'max_fill_factor': get_max_fill_factor(Gn)})

	if 'min_fill_factor' in attr_names:
		attrs.update({'min_fill_factor': get_min_fill_factor(Gn)})

	if 'substructures' in attr_names:
		attrs.update({'substructures': get_substructures(Gn)})

	if 'class_number' in attr_names:
		attrs.update({'class_number': get_class_num(target)})

	if 'node_attr_dim' in attr_names:
		attrs['node_attr_dim'] = get_node_attr_dim(Gn)

	if 'edge_attr_dim' in attr_names:
		attrs['edge_attr_dim'] = get_edge_attr_dim(Gn)

	from collections import OrderedDict
	return OrderedDict(
		sorted(attrs.items(), key=lambda i: attr_names.index(i[0])))


def load_predefined_dataset(ds_name):
	import os
	from gklearn.utils.graphfiles import loadDataset

	current_path = os.path.dirname(os.path.realpath(__file__)) + '/'
	if ds_name == 'Acyclic':
		ds_file = current_path + '../../datasets/Acyclic/dataset_bps.ds'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'AIDS':
		ds_file = current_path + '../../datasets/AIDS/AIDS_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Alkane':
		ds_file = current_path + '../../datasets/Alkane/dataset.ds'
		fn_targets = current_path + '../../datasets/Alkane/dataset_boiling_point_names.txt'
		graphs, targets = loadDataset(ds_file, filename_y=fn_targets)
	elif ds_name == 'COIL-DEL':
		ds_file = current_path + '../../datasets/COIL-DEL/COIL-DEL_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'COIL-RAG':
		ds_file = current_path + '../../datasets/COIL-RAG/COIL-RAG_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'COLORS-3':
		ds_file = current_path + '../../datasets/COLORS-3/COLORS-3_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Cuneiform':
		ds_file = current_path + '../../datasets/Cuneiform/Cuneiform_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'DD':
		ds_file = current_path + '../../datasets/DD/DD_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'ENZYMES':
		ds_file = current_path + '../../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Fingerprint':
		ds_file = current_path + '../../datasets/Fingerprint/Fingerprint_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'FRANKENSTEIN':
		ds_file = current_path + '../../datasets/FRANKENSTEIN/FRANKENSTEIN_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Letter-high': # node non-symb
		ds_file = current_path + '../../datasets/Letter-high/Letter-high_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Letter-low': # node non-symb
		ds_file = current_path + '../../datasets/Letter-low/Letter-low_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Letter-med': # node non-symb
		ds_file = current_path + '../../datasets/Letter-med/Letter-med_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'MAO':
		ds_file = current_path + '../../datasets/MAO/dataset.ds'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Monoterpenoides':
		ds_file = current_path + '../../datasets/Monoterpenoides/dataset_10+.ds'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'MUTAG':
		ds_file = current_path + '../../datasets/MUTAG/MUTAG_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'NCI1':
		ds_file = current_path + '../../datasets/NCI1/NCI1_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'NCI109':
		ds_file = current_path + '../../datasets/NCI109/NCI109_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'PAH':
		ds_file = current_path + '../../datasets/PAH/dataset.ds'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'SYNTHETIC':
		pass
	elif ds_name == 'SYNTHETICnew':
		ds_file = current_path + '../../datasets/SYNTHETICnew/SYNTHETICnew_A.txt'
		graphs, targets = loadDataset(ds_file)
	elif ds_name == 'Synthie':
		pass
	else:
		raise Exception('The dataset name "', ds_name, '" is not pre-defined.')

	return graphs, targets