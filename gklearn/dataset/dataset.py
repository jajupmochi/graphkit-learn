#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:48:27 2020

@author: ljia
"""
import numpy as np
import networkx as nx
import os
from gklearn.dataset import DATASET_META, DataFetcher, DataLoader


class Dataset(object):


	def __init__(self, inputs=None, root='datasets', filename_targets=None, targets=None, mode='networkx', remove_null_graphs=True, clean_labels=True, reload=False, verbose=False, **kwargs):
		self._substructures = None
		self._node_label_dim = None
		self._edge_label_dim = None
		self._directed = None
		self._dataset_size = None
		self._total_node_num = None
		self._ave_node_num = None
		self._min_node_num = None
		self._max_node_num = None
		self._total_edge_num = None
		self._ave_edge_num = None
		self._min_edge_num = None
		self._max_edge_num = None
		self._ave_node_degree = None
		self._min_node_degree = None
		self._max_node_degree = None
		self._ave_fill_factor = None
		self._min_fill_factor = None
		self._max_fill_factor = None
		self._node_label_nums = None
		self._edge_label_nums = None
		self._node_attr_dim = None
		self._edge_attr_dim = None
		self._class_number = None
		self._ds_name = None
		self._task_type = None

		if inputs is None:
			self._graphs = None
			self._targets = None
			self._node_labels = None
			self._edge_labels = None
			self._node_attrs = None
			self._edge_attrs = None

		# If inputs is a list of graphs.
		elif isinstance(inputs, list):
			node_labels = kwargs.get('node_labels', None)
			node_attrs = kwargs.get('node_attrs', None)
			edge_labels = kwargs.get('edge_labels', None)
			edge_attrs = kwargs.get('edge_attrs', None)
			self.load_graphs(inputs, targets=targets)
			self.set_labels(node_labels=node_labels, node_attrs=node_attrs, edge_labels=edge_labels, edge_attrs=edge_attrs)
			if clean_labels:
				self.clean_labels()

		elif isinstance(inputs, str):
			# If inputs is predefined dataset name.
			if inputs in DATASET_META:
				self.load_predefined_dataset(inputs, root=root, clean_labels=clean_labels, reload=reload, verbose=verbose)
				self._ds_name = inputs

			# If the dataset is specially defined, i.g., Alkane_unlabeled, MAO_lite.
			elif self.is_special_dataset(inputs):
				self.load_special_dataset(inputs, root, clean_labels, reload, verbose)
				self._ds_name = inputs

			# If inputs is a file name.
			elif os.path.isfile(inputs):
				self.load_dataset(inputs, filename_targets=filename_targets, clean_labels=clean_labels, **kwargs)

			# If inputs is a file name.
			else:
				raise ValueError('The "inputs" argument "' + inputs + '" is not a valid dataset name or file name.')

		else:
			raise TypeError('The "inputs" argument cannot be recognized. "Inputs" can be a list of graphs, a predefined dataset name, or a file name of a dataset.')

		if remove_null_graphs:
			self.trim_dataset(edge_required=False)


	def load_dataset(self, filename, filename_targets=None, clean_labels=True, **kwargs):
		self._graphs, self._targets, label_names = DataLoader(filename, filename_targets=filename_targets, **kwargs).data
		self._node_labels = label_names['node_labels']
		self._node_attrs = label_names['node_attrs']
		self._edge_labels = label_names['edge_labels']
		self._edge_attrs = label_names['edge_attrs']
		if clean_labels:
			self.clean_labels()


	def load_graphs(self, graphs, targets=None):
		# this has to be followed by set_labels().
		self._graphs = graphs
		self._targets = targets
#		self.set_labels_attrs() # @todo


	def load_predefined_dataset(self, ds_name, root='datasets', clean_labels=True, reload=False, verbose=False):
		path = DataFetcher(name=ds_name, root=root, reload=reload, verbose=verbose).path

		if DATASET_META[ds_name]['database'] == 'tudataset':
			ds_file = os.path.join(path, ds_name + '_A.txt')
			fn_targets = None
		else:
			load_files = DATASET_META[ds_name]['load_files']
			if isinstance(load_files[0], str):
				ds_file = os.path.join(path, load_files[0])
			else: # load_files[0] is a list of files.
				ds_file = [os.path.join(path, fn) for fn in load_files[0]]
			fn_targets = os.path.join(path, load_files[1]) if len(load_files) == 2 else None

		# Get extra_params.
		if 'extra_params' in DATASET_META[ds_name]:
			kwargs = DATASET_META[ds_name]['extra_params']
		else:
			kwargs = {}

		# Get the task type that is associated with the dataset. If it is classification, get the number of classes.
		self._get_task_type(ds_name)


		self._graphs, self._targets, label_names = DataLoader(ds_file, filename_targets=fn_targets, **kwargs).data

		self._node_labels = label_names['node_labels']
		self._node_attrs = label_names['node_attrs']
		self._edge_labels = label_names['edge_labels']
		self._edge_attrs = label_names['edge_attrs']
		if clean_labels:
			self.clean_labels()

		# Deal with specific datasets.
		if ds_name == 'Alkane':
			self.trim_dataset(edge_required=True)
			self.remove_labels(node_labels=['atom_symbol'])


	def set_labels(self, node_labels=[], node_attrs=[], edge_labels=[], edge_attrs=[]):
		self._node_labels = node_labels
		self._node_attrs = node_attrs
		self._edge_labels = edge_labels
		self._edge_attrs = edge_attrs


	def set_labels_attrs(self, node_labels=None, node_attrs=None, edge_labels=None, edge_attrs=None):
		# @todo: remove labels which have only one possible values.
		if node_labels is None:
			self._node_labels = self._graphs[0].graph['node_labels']
#			# graphs are considered node unlabeled if all nodes have the same label.
#			infos.update({'node_labeled': is_nl if node_label_num > 1 else False})
		if node_attrs is None:
			self._node_attrs = self._graphs[0].graph['node_attrs']
#		for G in Gn:
#			for n in G.nodes(data=True):
#				if 'attributes' in n[1]:
#					return len(n[1]['attributes'])
#		return 0
		if edge_labels is None:
			self._edge_labels = self._graphs[0].graph['edge_labels']
#			# graphs are considered edge unlabeled if all edges have the same label.
#			infos.update({'edge_labeled': is_el if edge_label_num > 1 else False})
		if edge_attrs is None:
			self._edge_attrs = self._graphs[0].graph['edge_attrs']
#		for G in Gn:
#			if nx.number_of_edges(G) > 0:
#				for e in G.edges(data=True):
#					if 'attributes' in e[2]:
#						return len(e[2]['attributes'])
#		return 0


	def get_dataset_infos(self, keys=None, params=None):
		"""Computes and returns the structure and property information of the graph dataset.

		Parameters
		----------
		keys : list, optional
			A list of strings which indicate which informations will be returned. The
			possible choices includes:

			'substructures': sub-structures graphs contains, including 'linear', 'non
		linear' and 'cyclic'.

			'node_label_dim': whether vertices have symbolic labels.

			'edge_label_dim': whether egdes have symbolic labels.

			'directed': whether graphs in dataset are directed.

			'dataset_size': number of graphs in dataset.

			'total_node_num': total number of vertices of all graphs in dataset.

			'ave_node_num': average number of vertices of graphs in dataset.

			'min_node_num': minimum number of vertices of graphs in dataset.

			'max_node_num': maximum number of vertices of graphs in dataset.

			'total_edge_num': total number of edges of all graphs in dataset.

			'ave_edge_num': average number of edges of graphs in dataset.

			'min_edge_num': minimum number of edges of graphs in dataset.

			'max_edge_num': maximum number of edges of graphs in dataset.

			'ave_node_degree': average vertex degree of graphs in dataset.

			'min_node_degree': minimum vertex degree of graphs in dataset.

			'max_node_degree': maximum vertex degree of graphs in dataset.

			'ave_fill_factor': average fill factor (number_of_edges /
		(number_of_nodes ** 2)) of graphs in dataset.

			'min_fill_factor': minimum fill factor of graphs in dataset.

			'max_fill_factor': maximum fill factor of graphs in dataset.

			'node_label_nums': list of numbers of symbolic vertex labels of graphs in dataset.

			'edge_label_nums': list number of symbolic edge labels of graphs in dataset.

			'node_attr_dim': number of dimensions of non-symbolic vertex labels.
		Extracted from the 'attributes' attribute of graph nodes.

			'edge_attr_dim': number of dimensions of non-symbolic edge labels.
		Extracted from the 'attributes' attribute of graph edges.

			'class_number': number of classes. Only available for classification problems.

			'all_degree_entropy': the entropy of degree distribution of each graph.

			'ave_degree_entropy': the average entropy of degree distribution of all graphs.

			All informations above will be returned if `keys` is not given.

		params: dict of dict, optional
			A dictinary which contains extra parameters for each possible
			element in ``keys``.

		Return
		------
		dict
			Information of the graph dataset keyed by `keys`.
		"""
		infos = {}

		if keys == None:
			keys = [
				'substructures',
				'node_label_dim',
				'edge_label_dim',
				'directed',
				'dataset_size',
				'total_node_num',
				'ave_node_num',
				'min_node_num',
				'max_node_num',
				'total_edge_num',
				'ave_edge_num',
				'min_edge_num',
				'max_edge_num',
				'ave_node_degree',
				'min_node_degree',
				'max_node_degree',
				'ave_fill_factor',
				'min_fill_factor',
				'max_fill_factor',
				'node_label_nums',
				'edge_label_nums',
				'node_attr_dim',
				'edge_attr_dim',
				'class_number',
				'all_degree_entropy',
				'ave_degree_entropy',
				'class_type'
			]

		# dataset size
		if 'dataset_size' in keys:
			if self._dataset_size is None:
				self._dataset_size = self._get_dataset_size()
			infos['dataset_size'] = self._dataset_size

		# graph node number
		if any(i in keys for i in ['total_node_num', 'ave_node_num', 'min_node_num', 'max_node_num']):
			all_node_nums = self._get_all_node_nums()

		if 'total_node_num' in keys:
			if self._total_node_num is None:
				self._total_node_num = self._get_total_node_num(all_node_nums)
			infos['total_node_num'] = self._total_node_num

		if 'ave_node_num' in keys:
			if self._ave_node_num is None:
				self._ave_node_num = self._get_ave_node_num(all_node_nums)
			infos['ave_node_num'] = self._ave_node_num

		if 'min_node_num' in keys:
			if self._min_node_num is None:
				self._min_node_num = self._get_min_node_num(all_node_nums)
			infos['min_node_num'] = self._min_node_num

		if 'max_node_num' in keys:
			if self._max_node_num is None:
				self._max_node_num = self._get_max_node_num(all_node_nums)
			infos['max_node_num'] = self._max_node_num

		# graph edge number
		if any(i in keys for i in ['total_edge_num', 'ave_edge_num', 'min_edge_num', 'max_edge_num']):
			all_edge_nums = self._get_all_edge_nums()

		if 'total_edge_num' in keys:
			if self._total_edge_num is None:
				self._total_edge_num = self._get_total_edge_num(all_edge_nums)
			infos['total_edge_num'] = self._total_edge_num

		if 'ave_edge_num' in keys:
			if self._ave_edge_num is None:
				self._ave_edge_num = self._get_ave_edge_num(all_edge_nums)
			infos['ave_edge_num'] = self._ave_edge_num

		if 'max_edge_num' in keys:
			if self._max_edge_num is None:
				self._max_edge_num = self._get_max_edge_num(all_edge_nums)
			infos['max_edge_num'] = self._max_edge_num

		if 'min_edge_num' in keys:
			if self._min_edge_num is None:
				self._min_edge_num = self._get_min_edge_num(all_edge_nums)
			infos['min_edge_num'] = self._min_edge_num

		# label number
		if 'node_label_dim' in keys:
			if self._node_label_dim is None:
				self._node_label_dim = self._get_node_label_dim()
			infos['node_label_dim'] = self._node_label_dim

		if 'node_label_nums' in keys:
			if self._node_label_nums is None:
				self._node_label_nums = {}
				for node_label in self._node_labels:
					self._node_label_nums[node_label] = self._get_node_label_num(node_label)
			infos['node_label_nums'] = self._node_label_nums

		if 'edge_label_dim' in keys:
			if self._edge_label_dim is None:
				self._edge_label_dim = self._get_edge_label_dim()
			infos['edge_label_dim'] = self._edge_label_dim

		if 'edge_label_nums' in keys:
			if self._edge_label_nums is None:
				self._edge_label_nums = {}
				for edge_label in self._edge_labels:
					self._edge_label_nums[edge_label] = self._get_edge_label_num(edge_label)
			infos['edge_label_nums'] = self._edge_label_nums

		if 'directed' in keys or 'substructures' in keys:
			if self._directed is None:
				self._directed = self._is_directed()
			infos['directed'] = self._directed

		# node degree
		if any(i in keys for i in ['ave_node_degree', 'max_node_degree', 'min_node_degree']):
			all_node_degrees = self._get_all_node_degrees()

		if 'ave_node_degree' in keys:
			if self._ave_node_degree is None:
				self._ave_node_degree = self._get_ave_node_degree(all_node_degrees)
			infos['ave_node_degree'] = self._ave_node_degree

		if 'max_node_degree' in keys:
			if self._max_node_degree is None:
				self._max_node_degree = self._get_max_node_degree(all_node_degrees)
			infos['max_node_degree'] = self._max_node_degree

		if 'min_node_degree' in keys:
			if self._min_node_degree is None:
				self._min_node_degree = self._get_min_node_degree(all_node_degrees)
			infos['min_node_degree'] = self._min_node_degree

		# fill factor
		if any(i in keys for i in ['ave_fill_factor', 'max_fill_factor', 'min_fill_factor']):
			all_fill_factors = self._get_all_fill_factors()

		if 'ave_fill_factor' in keys:
			if self._ave_fill_factor is None:
				self._ave_fill_factor = self._get_ave_fill_factor(all_fill_factors)
			infos['ave_fill_factor'] = self._ave_fill_factor

		if 'max_fill_factor' in keys:
			if self._max_fill_factor is None:
				self._max_fill_factor = self._get_max_fill_factor(all_fill_factors)
			infos['max_fill_factor'] = self._max_fill_factor

		if 'min_fill_factor' in keys:
			if self._min_fill_factor is None:
				self._min_fill_factor = self._get_min_fill_factor(all_fill_factors)
			infos['min_fill_factor'] = self._min_fill_factor

		if 'substructures' in keys:
			if self._substructures is None:
				self._substructures = self._get_substructures()
			infos['substructures'] = self._substructures

		if 'class_number' in keys:
			if self._class_number is None:
				self._class_number = self._get_class_num()
			infos['class_number'] = self._class_number

		if 'node_attr_dim' in keys:
			if self._node_attr_dim is None:
				self._node_attr_dim = self._get_node_attr_dim()
			infos['node_attr_dim'] = self._node_attr_dim

		if 'edge_attr_dim' in keys:
			if self._edge_attr_dim is None:
				self._edge_attr_dim = self._get_edge_attr_dim()
			infos['edge_attr_dim'] = self._edge_attr_dim

		# entropy of degree distribution.

		if 'all_degree_entropy' in keys:
			if params is not None and ('all_degree_entropy' in params) and ('base' in params['all_degree_entropy']):
				base = params['all_degree_entropy']['base']
			else:
				base = None
			infos['all_degree_entropy'] = self._compute_all_degree_entropy(base=base)

		if 'ave_degree_entropy' in keys:
			if params is not None and ('ave_degree_entropy' in params) and ('base' in params['ave_degree_entropy']):
				base = params['ave_degree_entropy']['base']
			else:
				base = None
			infos['ave_degree_entropy'] = np.mean(self._compute_all_degree_entropy(base=base))

		if 'task_type' in keys:
			if self._task_type is None:
				self._task_type = self._get_task_type()
			infos['task_type'] = self._task_type

		return infos


	def print_graph_infos(self, infos):
		from collections import OrderedDict
		keys = list(infos.keys())
		print(OrderedDict(sorted(infos.items(), key=lambda i: keys.index(i[0]))))


	def remove_labels(self, node_labels=[], edge_labels=[], node_attrs=[], edge_attrs=[]):
		node_labels = [item for item in node_labels if item in self._node_labels]
		edge_labels = [item for item in edge_labels if item in self._edge_labels]
		node_attrs = [item for item in node_attrs if item in self._node_attrs]
		edge_attrs = [item for item in edge_attrs if item in self._edge_attrs]

		for g in self._graphs:
			for nd in g.nodes():
				for nl in node_labels:
					del g.nodes[nd][nl]
				for na in node_attrs:
					del g.nodes[nd][na]
			for ed in g.edges():
				for el in edge_labels:
					del g.edges[ed][el]
				for ea in edge_attrs:
					del g.edges[ed][ea]
		if len(node_labels) > 0:
			self._node_labels = [nl for nl in self._node_labels if nl not in node_labels]
		if len(edge_labels) > 0:
			self._edge_labels = [el for el in self._edge_labels if el not in edge_labels]
		if len(node_attrs) > 0:
			self._node_attrs = [na for na in self._node_attrs if na not in node_attrs]
		if len(edge_attrs) > 0:
			self._edge_attrs = [ea for ea in self._edge_attrs if ea not in edge_attrs]


	def clean_labels(self):
		labels = []
		for name in self._node_labels:
			label = set()
			for G in self._graphs:
				label = label | set(nx.get_node_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self._graphs:
					for nd in G.nodes():
						del G.nodes[nd][name]
		self._node_labels = labels

		labels = []
		for name in self._edge_labels:
			label = set()
			for G in self._graphs:
				label = label | set(nx.get_edge_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self._graphs:
					for ed in G.edges():
						del G.edges[ed][name]
		self._edge_labels = labels

		labels = []
		for name in self._node_attrs:
			label = set()
			for G in self._graphs:
				label = label | set(nx.get_node_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self._graphs:
					for nd in G.nodes():
						del G.nodes[nd][name]
		self._node_attrs = labels

		labels = []
		for name in self._edge_attrs:
			label = set()
			for G in self._graphs:
				label = label | set(nx.get_edge_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self._graphs:
					for ed in G.edges():
						del G.edges[ed][name]
		self._edge_attrs = labels


	def cut_graphs(self, range_):
		self._graphs = [self._graphs[i] for i in range_]
		if self._targets is not None:
			self._targets = [self._targets[i] for i in range_]
		self.clean_labels()


	def trim_dataset(self, edge_required=False):
		if edge_required: # @todo: there is a possibility that some node labels will be removed.
			trimed_pairs = [(idx, g) for idx, g in enumerate(self._graphs) if (nx.number_of_nodes(g) != 0 and nx.number_of_edges(g) != 0)]
		else:
			trimed_pairs = [(idx, g) for idx, g in enumerate(self._graphs) if nx.number_of_nodes(g) != 0]
		idx = [p[0] for p in trimed_pairs]
		self._graphs = [p[1] for p in trimed_pairs]
		self._targets = [self._targets[i] for i in idx]
		self.clean_labels()


	def copy(self):
		dataset = Dataset()
		graphs = [g.copy() for g in self._graphs] if self._graphs is not None else None
		target = self._targets.copy() if self._targets is not None else None
		node_labels = self._node_labels.copy() if self._node_labels is not None else None
		node_attrs = self._node_attrs.copy() if self._node_attrs is not None else None
		edge_labels = self._edge_labels.copy() if self._edge_labels is not None else None
		edge_attrs = self._edge_attrs.copy() if self._edge_attrs is not None else None
		dataset.load_graphs(graphs, target)
		dataset.set_labels(node_labels=node_labels, node_attrs=node_attrs, edge_labels=edge_labels, edge_attrs=edge_attrs)
		# @todo: clean_labels and add other class members?
		return dataset


	def is_special_dataset(self, inputs):
		if inputs.endswith('_unlabeled'):
			return True
		if inputs == 'MAO_lite':
			return True
		if inputs == 'Monoterpens':
			return True
		return False


	def load_special_dataset(self, inputs, root, clean_labels, reload, verbose):
		if inputs.endswith('_unlabeled'):
			self.load_predefined_dataset(inputs[:len(inputs) - 10], root=root, clean_labels=clean_labels, reload=reload, verbose=verbose)

			self.remove_labels(node_labels=self._node_labels,
					  edge_labels=self._edge_labels,
					  node_attrs=self._node_attrs,
					  edge_attrs=self._edge_attrs)

		elif inputs == 'MAO_lite':
			self.load_predefined_dataset(inputs[:len(inputs) - 5], root=root, clean_labels=clean_labels, reload=reload, verbose=verbose)

			self.remove_labels(edge_labels=['bond_stereo'], node_attrs=['x', 'y'])

		elif inputs == 'Monoterpens':
			self.load_predefined_dataset('Monoterpenoides', root=root, clean_labels=clean_labels, reload=reload, verbose=verbose)


	def get_all_node_labels(self):
		node_labels = []
		for g in self._graphs:
			for n in g.nodes():
				nl = tuple(g.nodes[n].items())
				if nl not in node_labels:
					node_labels.append(nl)
		return node_labels


	def get_all_edge_labels(self):
		edge_labels = []
		for g in self._graphs:
			for e in g.edges():
				el = tuple(g.edges[e].items())
				if el not in edge_labels:
					edge_labels.append(el)
		return edge_labels


	def _get_dataset_size(self):
		return len(self._graphs)


	def _get_all_node_nums(self):
		return [nx.number_of_nodes(G) for G in self._graphs]


	def _get_total_node_nums(self, all_node_nums):
		return np.sum(all_node_nums)


	def _get_ave_node_num(self, all_node_nums):
		return np.mean(all_node_nums)


	def _get_min_node_num(self, all_node_nums):
		return np.amin(all_node_nums)


	def _get_max_node_num(self, all_node_nums):
		return np.amax(all_node_nums)


	def _get_all_edge_nums(self):
		return [nx.number_of_edges(G) for G in self._graphs]


	def _get_total_edge_nums(self, all_edge_nums):
		return np.sum(all_edge_nums)


	def _get_ave_edge_num(self, all_edge_nums):
		return np.mean(all_edge_nums)


	def _get_min_edge_num(self, all_edge_nums):
		return np.amin(all_edge_nums)


	def _get_max_edge_num(self, all_edge_nums):
		return np.amax(all_edge_nums)


	def _get_node_label_dim(self):
		return len(self._node_labels)


	def _get_node_label_num(self, node_label):
		nl = set()
		for G in self._graphs:
			nl = nl | set(nx.get_node_attributes(G, node_label).values())
		return len(nl)


	def _get_edge_label_dim(self):
		return len(self._edge_labels)


	def _get_edge_label_num(self, edge_label):
		el = set()
		for G in self._graphs:
			el = el | set(nx.get_edge_attributes(G, edge_label).values())
		return len(el)


	def _is_directed(self):
		return nx.is_directed(self._graphs[0])


	def _get_all_node_degrees(self):
		return [np.mean(list(dict(G.degree()).values())) for G in self._graphs]


	def _get_ave_node_degree(self, all_node_degrees):
		return np.mean(all_node_degrees)


	def _get_max_node_degree(self, all_node_degrees):
		return np.amax(all_node_degrees)


	def _get_min_node_degree(self, all_node_degrees):
		return np.amin(all_node_degrees)


	def _get_all_fill_factors(self):
		"""Get fill factor, the number of non-zero entries in the adjacency matrix.

		Returns
		-------
		list[float]
			List of fill factors for all graphs.
		"""
		return [nx.number_of_edges(G) / (nx.number_of_nodes(G) ** 2) for G in self._graphs]


	def _get_ave_fill_factor(self, all_fill_factors):
		return np.mean(all_fill_factors)


	def _get_max_fill_factor(self, all_fill_factors):
		return np.amax(all_fill_factors)


	def _get_min_fill_factor(self, all_fill_factors):
		return np.amin(all_fill_factors)


	def _get_substructures(self):
		subs = set()
		for G in self._graphs:
			degrees = list(dict(G.degree()).values())
			if any(i == 2 for i in degrees):
				subs.add('linear')
			if np.amax(degrees) >= 3:
				subs.add('non linear')
			if 'linear' in subs and 'non linear' in subs:
				break

		if self._directed:
			for G in self._graphs:
				if len(list(nx.find_cycle(G))) > 0:
					subs.add('cyclic')
					break
			# else:
			#	 # @todo: this method does not work for big graph with large amount of edges like D&D, try a better way.
			#	 upper = np.amin([nx.number_of_edges(G) for G in Gn]) * 2 + 10
			#	 for G in Gn:
			#		 if (nx.number_of_edges(G) < upper):
			#			 cyc = list(nx.simple_cycles(G.to_directed()))
			#			 if any(len(i) > 2 for i in cyc):
			#				 subs.add('cyclic')
			#				 break
			#	 if 'cyclic' not in subs:
			#		 for G in Gn:
			#			 cyc = list(nx.simple_cycles(G.to_directed()))
			#			 if any(len(i) > 2 for i in cyc):
			#				 subs.add('cyclic')
			#				 break

			return subs


	def _get_class_num(self):
		return len(set(self._targets))


	def _get_node_attr_dim(self):
		return len(self._node_attrs)


	def _get_edge_attr_dim(self):
		return len(self._edge_attrs)


	def _compute_all_degree_entropy(self, base=None):
		"""Compute the entropy of degree distribution of each graph.

		Parameters
		----------
		base : float, optional
			The logarithmic base to use. The default is ``e`` (natural logarithm).

		Returns
		-------
		degree_entropy : float
			The calculated entropy.
		"""
		from gklearn.utils.stats import entropy

		degree_entropy = []
		for g in self._graphs:
			degrees = list(dict(g.degree()).values())
			en = entropy(degrees, base=base)
			degree_entropy.append(en)
		return degree_entropy


	def _get_task_type(self, ds_name):
		if 'task_type' in DATASET_META[ds_name]:
			self._task_type = DATASET_META[ds_name]['task_type']
		if self._task_type == 'classification' and self._class_number is None and 'class_number' in DATASET_META[ds_name]:
				self._class_number = DATASET_META[ds_name]['class_number']


	@property
	def graphs(self):
		return self._graphs


	@property
	def targets(self):
		return self._targets


	@property
	def node_labels(self):
		return self._node_labels


	@property
	def edge_labels(self):
		return self._edge_labels


	@property
	def node_attrs(self):
		return self._node_attrs


	@property
	def edge_attrs(self):
		return self._edge_attrs


def split_dataset_by_target(dataset):
	from gklearn.preimage.utils import get_same_item_indices

	graphs = dataset.graphs
	targets = dataset.targets
	datasets = []
	idx_targets = get_same_item_indices(targets)
	for key, val in idx_targets.items():
		sub_graphs = [graphs[i] for i in val]
		sub_dataset = Dataset()
		sub_dataset.load_graphs(sub_graphs, [key] * len(val))
		node_labels = dataset.node_labels.copy() if dataset.node_labels is not None else None
		node_attrs = dataset.node_attrs.copy() if dataset.node_attrs is not None else None
		edge_labels = dataset.edge_labels.copy() if dataset.edge_labels is not None else None
		edge_attrs = dataset.edge_attrs.copy() if dataset.edge_attrs is not None else None
		sub_dataset.set_labels(node_labels=node_labels, node_attrs=node_attrs, edge_labels=edge_labels, edge_attrs=edge_attrs)
		datasets.append(sub_dataset)
		# @todo: clean_labels?
	return datasets