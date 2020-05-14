#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:48:27 2020

@author: ljia
"""
import numpy as np
import networkx as nx
from gklearn.utils.graph_files import load_dataset
import os


class Dataset(object):
	
	def __init__(self, filename=None, filename_targets=None, **kwargs):
		if filename is None:
			self.__graphs = None
			self.__targets = None
			self.__node_labels = None
			self.__edge_labels = None
			self.__node_attrs = None
			self.__edge_attrs = None
		else:
			self.load_dataset(filename, filename_targets=filename_targets, **kwargs)
		
		self.__substructures = None
		self.__node_label_dim = None
		self.__edge_label_dim = None
		self.__directed = None
		self.__dataset_size = None
		self.__total_node_num = None
		self.__ave_node_num = None
		self.__min_node_num = None
		self.__max_node_num = None
		self.__total_edge_num = None
		self.__ave_edge_num = None
		self.__min_edge_num = None
		self.__max_edge_num = None
		self.__ave_node_degree = None
		self.__min_node_degree = None
		self.__max_node_degree = None
		self.__ave_fill_factor = None
		self.__min_fill_factor = None
		self.__max_fill_factor = None
		self.__node_label_nums = None
		self.__edge_label_nums = None
		self.__node_attr_dim = None
		self.__edge_attr_dim = None
		self.__class_number = None
	
	
	def load_dataset(self, filename, filename_targets=None, **kwargs):
		self.__graphs, self.__targets, label_names = load_dataset(filename, filename_targets=filename_targets, **kwargs)
		self.__node_labels = label_names['node_labels']
		self.__node_attrs = label_names['node_attrs']
		self.__edge_labels = label_names['edge_labels']
		self.__edge_attrs = label_names['edge_attrs']
		self.clean_labels()
		
		
	def load_graphs(self, graphs, targets=None):
		# this has to be followed by set_labels().
		self.__graphs = graphs
		self.__targets = targets
#		self.set_labels_attrs() # @todo
		
		
	def load_predefined_dataset(self, ds_name):
		current_path = os.path.dirname(os.path.realpath(__file__)) + '/'
		if ds_name == 'Acyclic':
			ds_file = current_path + '../../datasets/Acyclic/dataset_bps.ds'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'AIDS':
			ds_file = current_path + '../../datasets/AIDS/AIDS_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Alkane':
			ds_file = current_path + '../../datasets/Alkane/dataset.ds'
			fn_targets = current_path + '../../datasets/Alkane/dataset_boiling_point_names.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file, filename_targets=fn_targets)
		elif ds_name == 'COIL-DEL':
			ds_file = current_path + '../../datasets/COIL-DEL/COIL-DEL_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'COIL-RAG':
			ds_file = current_path + '../../datasets/COIL-RAG/COIL-RAG_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'COLORS-3':
			ds_file = current_path + '../../datasets/COLORS-3/COLORS-3_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Cuneiform':
			ds_file = current_path + '../../datasets/Cuneiform/Cuneiform_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'DD':
			ds_file = current_path + '../../datasets/DD/DD_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Fingerprint':
			ds_file = current_path + '../../datasets/Fingerprint/Fingerprint_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'FRANKENSTEIN':
			ds_file = current_path + '../../datasets/FRANKENSTEIN/FRANKENSTEIN_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Letter-high': # node non-symb
			ds_file = current_path + '../../datasets/Letter-high/Letter-high_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Letter-low': # node non-symb
			ds_file = current_path + '../../datasets/Letter-low/Letter-low_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Letter-med': # node non-symb
			ds_file = current_path + '../../datasets/Letter-med/Letter-med_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'MAO':
			ds_file = current_path + '../../datasets/MAO/dataset.ds'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Monoterpenoides':
			ds_file = current_path + '../../datasets/Monoterpenoides/dataset_10+.ds'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'MUTAG':
			ds_file = current_path + '../../datasets/MUTAG/MUTAG_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'PAH':
			ds_file = current_path + '../../datasets/PAH/dataset.ds'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'SYNTHETIC':
			pass
		elif ds_name == 'SYNTHETICnew':
			ds_file = current_path + '../../datasets/SYNTHETICnew/SYNTHETICnew_A.txt'
			self.__graphs, self.__targets, label_names = load_dataset(ds_file)
		elif ds_name == 'Synthie':
			pass
		else:
			raise Exception('The dataset name "', ds_name, '" is not pre-defined.')
	
		self.__node_labels = label_names['node_labels']
		self.__node_attrs = label_names['node_attrs']
		self.__edge_labels = label_names['edge_labels']
		self.__edge_attrs = label_names['edge_attrs']
		self.clean_labels()
	

	def set_labels(self, node_labels=[], node_attrs=[], edge_labels=[], edge_attrs=[]):
		self.__node_labels = node_labels
		self.__node_attrs = node_attrs
		self.__edge_labels = edge_labels
		self.__edge_attrs = edge_attrs

		
	def set_labels_attrs(self, node_labels=None, node_attrs=None, edge_labels=None, edge_attrs=None):
		# @todo: remove labels which have only one possible values.
		if node_labels is None:
			self.__node_labels = self.__graphs[0].graph['node_labels']
#			# graphs are considered node unlabeled if all nodes have the same label.
#			infos.update({'node_labeled': is_nl if node_label_num > 1 else False})
		if node_attrs is None:
			self.__node_attrs = self.__graphs[0].graph['node_attrs']
#		for G in Gn:
#			for n in G.nodes(data=True):
#				if 'attributes' in n[1]:
#					return len(n[1]['attributes'])
#		return 0
		if edge_labels is None:
			self.__edge_labels = self.__graphs[0].graph['edge_labels']
#			# graphs are considered edge unlabeled if all edges have the same label.
#			infos.update({'edge_labeled': is_el if edge_label_num > 1 else False})
		if edge_attrs is None:
			self.__edge_attrs = self.__graphs[0].graph['edge_attrs']
#		for G in Gn:
#			if nx.number_of_edges(G) > 0:
#				for e in G.edges(data=True):
#					if 'attributes' in e[2]:
#						return len(e[2]['attributes'])
#		return 0
			
			
	def get_dataset_infos(self, keys=None):
		"""Computes and returns the structure and property information of the graph dataset.
	
		Parameters
		----------
		keys : list
			List of strings which indicate which informations will be returned. The
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
			
			All informations above will be returned if `keys` is not given.
	
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
			]
	
		# dataset size
		if 'dataset_size' in keys:
			if self.__dataset_size is None:
				self.__dataset_size = self.__get_dataset_size()
			infos['dataset_size'] = self.__dataset_size
	
		# graph node number
		if any(i in keys for i in ['total_node_num', 'ave_node_num', 'min_node_num', 'max_node_num']):
			all_node_nums = self.__get_all_node_nums()

		if 'total_node_num' in keys:
			if self.__total_node_num is None:
				self.__total_node_num = self.__get_total_node_num(all_node_nums)
			infos['total_node_num'] = self.__total_node_num
	
		if 'ave_node_num' in keys:
			if self.__ave_node_num is None:
				self.__ave_node_num = self.__get_ave_node_num(all_node_nums)
			infos['ave_node_num'] = self.__ave_node_num
	
		if 'min_node_num' in keys:
			if self.__min_node_num is None:
				self.__min_node_num = self.__get_min_node_num(all_node_nums)
			infos['min_node_num'] = self.__min_node_num
	
		if 'max_node_num' in keys:
			if self.__max_node_num is None:
				self.__max_node_num = self.__get_max_node_num(all_node_nums)
			infos['max_node_num'] = self.__max_node_num
	
		# graph edge number
		if any(i in keys for i in ['total_edge_num', 'ave_edge_num', 'min_edge_num', 'max_edge_num']):
			all_edge_nums = self.__get_all_edge_nums()

		if 'total_edge_num' in keys:
			if self.__total_edge_num is None:
				self.__total_edge_num = self.__get_total_edge_num(all_edge_nums)
			infos['total_edge_num'] = self.__total_edge_num
			
		if 'ave_edge_num' in keys:
			if self.__ave_edge_num is None:
				self.__ave_edge_num = self.__get_ave_edge_num(all_edge_nums)
			infos['ave_edge_num'] = self.__ave_edge_num
	
		if 'max_edge_num' in keys:
			if self.__max_edge_num is None:
				self.__max_edge_num = self.__get_max_edge_num(all_edge_nums)
			infos['max_edge_num'] = self.__max_edge_num

		if 'min_edge_num' in keys:
			if self.__min_edge_num is None:
				self.__min_edge_num = self.__get_min_edge_num(all_edge_nums)
			infos['min_edge_num'] = self.__min_edge_num
	
		# label number
		if 'node_label_dim' in keys:
			if self.__node_label_dim is None:
				self.__node_label_dim = self.__get_node_label_dim()
			infos['node_label_dim'] = self.__node_label_dim	
	
		if 'node_label_nums' in keys:
			if self.__node_label_nums is None:
				self.__node_label_nums = {}
				for node_label in self.__node_labels:
					self.__node_label_nums[node_label] = self.__get_node_label_num(node_label)
			infos['node_label_nums'] = self.__node_label_nums
	
		if 'edge_label_dim' in keys:
			if self.__edge_label_dim is None:
				self.__edge_label_dim = self.__get_edge_label_dim()
			infos['edge_label_dim'] = self.__edge_label_dim	
	
		if 'edge_label_nums' in keys:
			if self.__edge_label_nums is None:
				self.__edge_label_nums = {}
				for edge_label in self.__edge_labels:
					self.__edge_label_nums[edge_label] = self.__get_edge_label_num(edge_label)
			infos['edge_label_nums'] = self.__edge_label_nums
	
		if 'directed' in keys or 'substructures' in keys:
			if self.__directed is None:
				self.__directed = self.__is_directed()
			infos['directed'] = self.__directed
	
		# node degree
		if any(i in keys for i in ['ave_node_degree', 'max_node_degree', 'min_node_degree']):
			all_node_degrees = self.__get_all_node_degrees()
			
		if 'ave_node_degree' in keys:
			if self.__ave_node_degree is None:
				self.__ave_node_degree = self.__get_ave_node_degree(all_node_degrees)
			infos['ave_node_degree'] = self.__ave_node_degree
	
		if 'max_node_degree' in keys:
			if self.__max_node_degree is None:
				self.__max_node_degree = self.__get_max_node_degree(all_node_degrees)
			infos['max_node_degree'] = self.__max_node_degree
	
		if 'min_node_degree' in keys:
			if self.__min_node_degree is None:
				self.__min_node_degree = self.__get_min_node_degree(all_node_degrees)
			infos['min_node_degree'] = self.__min_node_degree
			
		# fill factor
		if any(i in keys for i in ['ave_fill_factor', 'max_fill_factor', 'min_fill_factor']):
			all_fill_factors = self.__get_all_fill_factors()
			
		if 'ave_fill_factor' in keys:
			if self.__ave_fill_factor is None:
				self.__ave_fill_factor = self.__get_ave_fill_factor(all_fill_factors)
			infos['ave_fill_factor'] = self.__ave_fill_factor
	
		if 'max_fill_factor' in keys:
			if self.__max_fill_factor is None:
				self.__max_fill_factor = self.__get_max_fill_factor(all_fill_factors)
			infos['max_fill_factor'] = self.__max_fill_factor
	
		if 'min_fill_factor' in keys:
			if self.__min_fill_factor is None:
				self.__min_fill_factor = self.__get_min_fill_factor(all_fill_factors)
			infos['min_fill_factor'] = self.__min_fill_factor
	
		if 'substructures' in keys:
			if self.__substructures is None:
				self.__substructures = self.__get_substructures()
			infos['substructures'] = self.__substructures
	
		if 'class_number' in keys:
			if self.__class_number is None:
				self.__class_number = self.__get_class_number()
			infos['class_number'] = self.__class_number
	
		if 'node_attr_dim' in keys:
			if self.__node_attr_dim is None:
				self.__node_attr_dim = self.__get_node_attr_dim()
			infos['node_attr_dim'] = self.__node_attr_dim
	
		if 'edge_attr_dim' in keys:
			if self.__edge_attr_dim is None:
				self.__edge_attr_dim = self.__get_edge_attr_dim()
			infos['edge_attr_dim'] = self.__edge_attr_dim
			
		return infos
			
			
	def print_graph_infos(self, infos):
		from collections import OrderedDict
		keys = list(infos.keys())
		print(OrderedDict(sorted(infos.items(), key=lambda i: keys.index(i[0]))))
		
		
	def remove_labels(self, node_labels=[], edge_labels=[], node_attrs=[], edge_attrs=[]):
		node_labels = [item for item in node_labels if item in self.__node_labels]
		edge_labels = [item for item in edge_labels if item in self.__edge_labels]
		node_attrs = [item for item in node_attrs if item in self.__node_attrs]
		edge_attrs = [item for item in edge_attrs if item in self.__edge_attrs]

		for g in self.__graphs:
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
			self.__node_labels = [nl for nl in self.__node_labels if nl not in node_labels]
		if len(edge_labels) > 0:
			self.__edge_labels = [el for el in self.__edge_labels if el not in edge_labels]
		if len(node_attrs) > 0:
			self.__node_attrs = [na for na in self.__node_attrs if na not in node_attrs]
		if len(edge_attrs) > 0:
			self.__edge_attrs = [ea for ea in self.__edge_attrs if ea not in edge_attrs]
	
			
	def clean_labels(self):
		labels = []
		for name in self.__node_labels:
			label = set()
			for G in self.__graphs:
				label = label | set(nx.get_node_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self.__graphs:
					for nd in G.nodes():
						del G.nodes[nd][name]
		self.__node_labels = labels

		labels = []
		for name in self.__edge_labels:
			label = set()
			for G in self.__graphs:
				label = label | set(nx.get_edge_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self.__graphs:
					for ed in G.edges():
						del G.edges[ed][name]
		self.__edge_labels = labels

		labels = []
		for name in self.__node_attrs:
			label = set()
			for G in self.__graphs:
				label = label | set(nx.get_node_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self.__graphs:
					for nd in G.nodes():
						del G.nodes[nd][name]
		self.__node_attrs = labels

		labels = []
		for name in self.__edge_attrs:
			label = set()
			for G in self.__graphs:
				label = label | set(nx.get_edge_attributes(G, name).values())
				if len(label) > 1:
					labels.append(name)
					break
			if len(label) < 2:
				for G in self.__graphs:
					for ed in G.edges():
						del G.edges[ed][name]
		self.__edge_attrs = labels
				
		
	def cut_graphs(self, range_):
		self.__graphs = [self.__graphs[i] for i in range_]
		if self.__targets is not None:
			self.__targets = [self.__targets[i] for i in range_]
		self.clean_labels()


	def trim_dataset(self, edge_required=False):
		if edge_required:
			trimed_pairs = [(idx, g) for idx, g in enumerate(self.__graphs) if (nx.number_of_nodes(g) != 0 and nx.number_of_edges(g) != 0)]
		else:
			trimed_pairs = [(idx, g) for idx, g in enumerate(self.__graphs) if nx.number_of_nodes(g) != 0]
		idx = [p[0] for p in trimed_pairs]
		self.__graphs = [p[1] for p in trimed_pairs]
		self.__targets = [self.__targets[i] for i in idx]
		self.clean_labels()
		
		
	def copy(self):
		dataset = Dataset()
		graphs = [g.copy() for g in self.__graphs] if self.__graphs is not None else None
		target = self.__targets.copy() if self.__targets is not None else None
		node_labels = self.__node_labels.copy() if self.__node_labels is not None else None
		node_attrs = self.__node_attrs.copy() if self.__node_attrs is not None else None
		edge_labels = self.__edge_labels.copy() if self.__edge_labels is not None else None
		edge_attrs = self.__edge_attrs.copy() if self.__edge_attrs is not None else None
		dataset.load_graphs(graphs, target)
		dataset.set_labels(node_labels=node_labels, node_attrs=node_attrs, edge_labels=edge_labels, edge_attrs=edge_attrs)
		# @todo: clean_labels and add other class members?
		return dataset
		
	
	def __get_dataset_size(self):
		return len(self.__graphs)
	
	
	def __get_all_node_nums(self):
		return [nx.number_of_nodes(G) for G in self.__graphs]
	
	
	def __get_total_node_nums(self, all_node_nums):
		return np.sum(all_node_nums)
	
	
	def __get_ave_node_num(self, all_node_nums):
		return np.mean(all_node_nums)
	
	
	def __get_min_node_num(self, all_node_nums):
		return np.amin(all_node_nums)
	
	
	def __get_max_node_num(self, all_node_nums):
		return np.amax(all_node_nums)
	
	
	def __get_all_edge_nums(self):
		return [nx.number_of_edges(G) for G in self.__graphs]
	
	
	def __get_total_edge_nums(self, all_edge_nums):
		return np.sum(all_edge_nums)
	
	
	def __get_ave_edge_num(self, all_edge_nums):
		return np.mean(all_edge_nums)
	
		
	def __get_min_edge_num(self, all_edge_nums):
		return np.amin(all_edge_nums)
	
		
	def __get_max_edge_num(self, all_edge_nums):
		return np.amax(all_edge_nums)
	
	
	def __get_node_label_dim(self):
		return len(self.__node_labels)
	
		
	def __get_node_label_num(self, node_label):
		nl = set()
		for G in self.__graphs:
			nl = nl | set(nx.get_node_attributes(G, node_label).values())
		return len(nl)
	
	
	def __get_edge_label_dim(self):
		return len(self.__edge_labels)
	
		
	def __get_edge_label_num(self, edge_label):
		el = set()
		for G in self.__graphs:
			el = el | set(nx.get_edge_attributes(G, edge_label).values())
		return len(el)
	
		
	def __is_directed(self):
		return nx.is_directed(self.__graphs[0])
	
		
	def __get_all_node_degrees(self):
		return [np.mean(list(dict(G.degree()).values())) for G in self.__graphs]
	
	
	def __get_ave_node_degree(self, all_node_degrees):
		return np.mean(all_node_degrees)
	
	
	def __get_max_node_degree(self, all_node_degrees):
		return np.amax(all_node_degrees)
	
		
	def __get_min_node_degree(self, all_node_degrees):
		return np.amin(all_node_degrees)
		
	
	def __get_all_fill_factors(self):
		"""
		Get fill factor, the number of non-zero entries in the adjacency matrix.

		Returns
		-------
		list[float]
			List of fill factors for all graphs.
		"""
		return [nx.number_of_edges(G) / (nx.number_of_nodes(G) ** 2) for G in self.__graphs]
	   

	def __get_ave_fill_factor(self, all_fill_factors):
		return np.mean(all_fill_factors)
	
		
	def __get_max_fill_factor(self, all_fill_factors):
		return np.amax(all_fill_factors)
	
		
	def __get_min_fill_factor(self, all_fill_factors):
		return np.amin(all_fill_factors)
	
		
	def __get_substructures(self):
		subs = set()
		for G in self.__graphs:
			degrees = list(dict(G.degree()).values())
			if any(i == 2 for i in degrees):
				subs.add('linear')
			if np.amax(degrees) >= 3:
				subs.add('non linear')
			if 'linear' in subs and 'non linear' in subs:
				break

		if self.__directed:
			for G in self.__graphs:
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
	
		
	def __get_class_num(self):
		return len(set(self.__targets))
	
		
	def __get_node_attr_dim(self):
		return len(self.__node_attrs)
	
		
	def __get_edge_attr_dim(self):
		return len(self.__edge_attrs)
	
	
	@property
	def graphs(self):
		return self.__graphs


	@property
	def targets(self):
		return self.__targets
	
		
	@property
	def node_labels(self):
		return self.__node_labels


	@property
	def edge_labels(self):
		return self.__edge_labels
	
	
	@property
	def node_attrs(self):
		return self.__node_attrs
	
	
	@property
	def edge_attrs(self):
		return self.__edge_attrs
	
	
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