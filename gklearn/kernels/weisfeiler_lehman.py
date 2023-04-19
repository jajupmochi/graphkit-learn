#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:16:34 2020

@author: ljia

@references:

	[1] Shervashidze N, Schweitzer P, Leeuwen EJ, Mehlhorn K, Borgwardt KM.
	Weisfeiler-lehman graph kernels. Journal of Machine Learning Research.
	2011;12(Sep):2539-61.
"""

import sys
from collections import Counter
# from functools import partial
from itertools import combinations_with_replacement

import networkx as nx
import numpy as np

from gklearn.kernels import GraphKernel
from gklearn.utils import SpecialLabel
from gklearn.utils.iters import get_iters
from gklearn.utils.parallel import parallel_gm, parallel_me


class WeisfeilerLehman(GraphKernel):  # @todo: sp, edge user kernel.

	def __init__(self, **kwargs):
		GraphKernel.__init__(
			self, **{
				k: kwargs.get(k) for k in
				['parallel', 'n_jobs', 'chunksize', 'normalize',
				 'copy_graphs', 'verbose'] if k in kwargs
			}
		)
		self.node_labels = kwargs.get('node_labels', [])
		self.edge_labels = kwargs.get('edge_labels', [])
		self.height = int(kwargs.get('height', 0))
		self._base_kernel = kwargs.get('base_kernel', 'subtree')
		self._ds_infos = kwargs.get('ds_infos', {})


	##########################################################################
	# The following is the 1st paradigm to compute kernel matrix, which is
	# compatible with `scikit-learn`.
	# -------------------------------------------------------------------
	# Special thanks to the "GraKeL" library for providing an excellent template!
	##########################################################################

	##########################################################################
	# The following is the 2nd paradigm to compute kernel matrix. It is
	# simplified and not compatible with `scikit-learn`.
	##########################################################################

	def _compute_gm_series(self, graphs):
		#		if self.verbose >= 2:
		#			import warnings
		#			warnings.warn('A part of the computation is parallelized.')

		# 		self._add_dummy_node_labels(self._graphs)

		# for WL subtree kernel
		if self._base_kernel == 'subtree':
			gram_matrix = self._subtree_kernel_do(graphs)

		# for WL shortest path kernel
		elif self._base_kernel == 'sp':
			gram_matrix = self._sp_kernel_do(graphs)

		# for WL edge kernel
		elif self._base_kernel == 'edge':
			gram_matrix = self._edge_kernel_do(graphs)

		# for user defined base kernel
		else:
			gram_matrix = self._user_kernel_do(graphs)

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		# 		self._add_dummy_node_labels(self._graphs)

		if self._base_kernel == 'subtree':
			gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))


			#			for i in range(len(self._graphs)):
			#				for j in range(i, len(self._graphs)):
			#					gram_matrix[i][j] = self.pairwise_kernel(self._graphs[i], self._graphs[j])
			#					gram_matrix[j][i] = gram_matrix[i][j]

			def init_worker(gn_toshare):
				global G_gn
				G_gn = gn_toshare


			do_fun = self._wrapper_pairwise
			parallel_gm(
				do_fun, gram_matrix, self._graphs, init_worker=init_worker,
				glbv=(self._graphs,), n_jobs=self.n_jobs, verbose=self.verbose
			)
			return gram_matrix
		else:
			if self.verbose >= 2:
				import warnings
				warnings.warn(
					'This base kernel is not parallelized. The serial computation '
					'is used instead.'
				)
			return self._compute_gm_series(self._graphs)


	def _compute_kernel_list_series(
			self, g1, g_list
	):  # @todo: this should be better.
		#		if self.verbose >= 2:
		#			import warnings
		#			warnings.warn('A part of the computation is parallelized.')

		self._add_dummy_node_labels(g_list + [g1])

		# for WL subtree kernel
		if self._base_kernel == 'subtree':
			gram_matrix = self._subtree_kernel_do(g_list + [g1])

		# for WL shortest path kernel
		elif self._base_kernel == 'sp':
			gram_matrix = self._sp_kernel_do(g_list + [g1])

		# for WL edge kernel
		elif self._base_kernel == 'edge':
			gram_matrix = self._edge_kernel_do(g_list + [g1])

		# for user defined base kernel
		else:
			gram_matrix = self._user_kernel_do(g_list + [g1])

		return list(gram_matrix[-1][0:-1])


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._add_dummy_node_labels(g_list + [g1])

		if self._base_kernel == 'subtree':
			kernel_list = [None] * len(g_list)


			def init_worker(g1_toshare, g_list_toshare):
				global G_g1, G_g_list
				G_g1 = g1_toshare
				G_g_list = g_list_toshare


			do_fun = self._wrapper_kernel_list_do


			def func_assign(result, var_to_assign):
				var_to_assign[result[0]] = result[1]


			itr = range(len(g_list))
			len_itr = len(g_list)
			parallel_me(
				do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
				init_worker=init_worker, glbv=(g1, g_list),
				method='imap_unordered',
				n_jobs=self.n_jobs, itr_desc='Computing kernels',
				verbose=self.verbose
			)
			return kernel_list
		else:
			if self.verbose >= 2:
				import warnings
				warnings.warn(
					'This base kernel is not parallelized. The serial computation '
					'is used instead.'
				)
			return self._compute_kernel_list_series(g1, g_list)


	def _wrapper_kernel_list_do(self, itr):
		return itr, self.pairwise_kernel(G_g1, G_g_list[itr])


	def _compute_single_kernel_series(
			self, g1, g2
	):  # @todo: this should be better.
		self._add_dummy_node_labels([g1] + [g2])

		# for WL subtree kernel
		if self._base_kernel == 'subtree':
			gram_matrix = self._subtree_kernel_do([g1] + [g2])

		# for WL shortest path kernel
		elif self._base_kernel == 'sp':
			gram_matrix = self._sp_kernel_do([g1] + [g2])

		# for WL edge kernel
		elif self._base_kernel == 'edge':
			gram_matrix = self._edge_kernel_do([g1] + [g2])

		# for user defined base kernel
		else:
			gram_matrix = self._user_kernel_do([g1] + [g2])

		return gram_matrix[0][1]


	##########################################################################
	# The following are the methods used by both diagrams.
	##########################################################################

	def validate_parameters(self):
		"""Validate all parameters for the transformer.

		Returns
		-------
		None.

		"""
		super().validate_parameters()
		if len(self.node_labels) == 0:
			if len(self.edge_labels) == 0:
				self._subtree_kernel_do = self._subtree_kernel_do_unlabeled
			else:
				self._subtree_kernel_do = self._subtree_kernel_do_el
		else:
			if len(self.edge_labels) == 0:
				self._subtree_kernel_do = self._subtree_kernel_do_nl
			else:
				self._subtree_kernel_do = self._subtree_kernel_do_labeled


	def pairwise_kernel(self, g1, g2):
		# 		Gn = [g1.copy(), g2.copy()] # @todo: make sure it is a full deep copy. and faster!
		Gn = [g1, g2]
		# for WL subtree kernel
		if self._base_kernel == 'subtree':
			kernel = self._subtree_kernel_do(Gn, return_mat=False)

		# @todo: other subkernels.

		return kernel


	def _wrapper_pairwise(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self.pairwise_kernel(G_gn[i], G_gn[j])


	def _compute_kernel_itr(self, kernel, all_num_of_each_label):
		labels = set(
			list(all_num_of_each_label[0].keys()) +
			list(all_num_of_each_label[1].keys())
		)
		vector1 = np.array(
			[(all_num_of_each_label[0][label]
			  if (label in all_num_of_each_label[0].keys()) else 0)
			 for label in labels]
		)
		vector2 = np.array(
			[(all_num_of_each_label[1][label]
			  if (label in all_num_of_each_label[1].keys()) else 0)
			 for label in labels]
		)
		kernel += np.dot(vector1, vector2)
		return kernel


	def _subtree_kernel_do_nl(self, Gn, return_mat=True):
		"""Compute Weisfeiler-Lehman kernels between graphs with node labels.

		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are computed.

		Return
		------
		kernel_matrix : Numpy matrix / float
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		kernel_matrix = (np.zeros((len(Gn), len(Gn))) if return_mat else 0)
		gram_itr_fun = (
			self._compute_gram_itr if return_mat else self._compute_kernel_itr)

		# initial for height = 0
		all_num_of_each_label = []  # number of occurence of each label in each graph in this iteration

		# for each graph
		if self.verbose >= 2:
			iterator = get_iters(Gn, desc='Setting all labels into a tuple')
		else:
			iterator = Gn
		for G in iterator:
			# set all labels into a tuple. # @todo: remove this original labels or not?
			for nd, attrs in G.nodes(
					data=True
			):  # @todo: there may be a better way.
				G.nodes[nd]['lt'] = tuple(
					attrs[name] for name in self.node_labels
				)
			# get the set of original labels
			labels_ori = list(nx.get_node_attributes(G, 'lt').values())
			# number of occurence of each label in G
			all_num_of_each_label.append(dict(Counter(labels_ori)))

		# Compute subtree kernel with the 0th iteration and add it to the final kernel.
		kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		# iterate each height
		for h in range(1, self.height + 1):
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			#		all_labels_ori = set() # all unique orignal labels in all graphs in this iteration
			all_num_of_each_label = []  # number of occurence of each label in G

			# @todo: parallel this part.
			# 			if self.verbose >= 2:
			# 				iterator = get_iters(enumerate(Gn), desc='Going through iteration ' + str(h), length=len(Gn))
			# 			else:
			# 				iterator = enumerate(Gn)
			for G in Gn:
				num_of_labels_occured = self._subtree_1graph_nl(
					G, all_set_compressed, all_num_of_each_label,
					num_of_labels_occured
				)

			# Compute subtree kernel with h iterations and add it to the final kernel
			kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		return kernel_matrix


	def _subtree_kernel_do_el(self, Gn, return_mat=True):
		"""Compute Weisfeiler-Lehman kernels between graphs with edge labels.

		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are computed.

		Return
		------
		kernel_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		kernel_matrix = (np.zeros((len(Gn), len(Gn))) if return_mat else 0)
		gram_itr_fun = (
			self._compute_gram_itr if return_mat else self._compute_kernel_itr)

		# initial for height = 0
		all_num_of_each_label = []  # number of occurence of each label in each graph in this iteration

		# Compute subtree kernel with the 0th iteration and add it to the final kernel.
		# if return a kernel matrix:
		if return_mat:
			iterator = combinations_with_replacement(
				range(0, len(kernel_matrix)), 2
			)
			for i, j in iterator:
				kernel_matrix[i][j] += nx.number_of_nodes(
					Gn[i]
				) * nx.number_of_nodes(Gn[j])
				kernel_matrix[j][i] = kernel_matrix[i][j]
		# if return a single kernel between two graphs:
		else:
			kernel_matrix += nx.number_of_nodes(Gn[0]) * nx.number_of_nodes(
				Gn[1]
			)

		# if h >= 1.
		if self.height > 0:
			# Set all edge labels into a tuple. # @todo: remove this original labels or not?
			if self.verbose >= 2:
				iterator = get_iters(Gn, desc='Setting all labels into a tuple')
			else:
				iterator = Gn
			for G in iterator:
				for n1, n2, attrs in G.edges(
						data=True
				):  # @todo: there may be a better way.
					G.edges[(n1, n2)]['lt'] = tuple(
						attrs[name] for name in self.edge_labels
					)

			# When h == 1, compute the kernel.
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			all_num_of_each_label = []  # number of occurence of each label in G

			# @todo: parallel this part.
			for G in Gn:
				num_of_labels_occured = self._subtree_1graph_el(
					G, all_set_compressed, all_num_of_each_label,
					num_of_labels_occured
				)

			# Compute subtree kernel with h iterations and add it to the final kernel.
			kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		# Iterate along heights (>= 2).
		for h in range(2, self.height + 1):
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			all_num_of_each_label = []  # number of occurence of each label in G

			# @todo: parallel this part.
			for G in Gn:
				num_of_labels_occured = self._subtree_1graph_nl(
					G, all_set_compressed, all_num_of_each_label,
					num_of_labels_occured
				)

			# Compute subtree kernel with h iterations and add it to the final kernel.
			kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		return kernel_matrix


	def _subtree_kernel_do_labeled(self, Gn, return_mat=True):
		"""Compute Weisfeiler-Lehman kernels between graphs with both node and
		edge labels.

		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are computed.

		Return
		------
		kernel_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		kernel_matrix = (np.zeros((len(Gn), len(Gn))) if return_mat else 0)
		gram_itr_fun = (
			self._compute_gram_itr if return_mat else self._compute_kernel_itr)

		# initial for height = 0
		all_num_of_each_label = []  # number of occurence of each label in each graph in this iteration

		# Set all node labels into a tuple and get # of occurence of each label.
		if self.verbose >= 2:
			iterator = get_iters(
				Gn, desc='Setting all node labels into a tuple'
			)
		else:
			iterator = Gn
		for G in iterator:
			# Set all node labels into a tuple. # @todo: remove this original labels or not?
			for nd, attrs in G.nodes(
					data=True
			):  # @todo: there may be a better way.
				G.nodes[nd]['lt'] = tuple(
					attrs[name] for name in self.node_labels
				)
			# Get the set of original labels.
			labels_ori = list(nx.get_node_attributes(G, 'lt').values())
			# number of occurence of each label in G
			all_num_of_each_label.append(dict(Counter(labels_ori)))

		# Compute subtree kernel with the 0th iteration and add it to the final kernel.
		kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		# if h >= 1:
		if self.height > 0:
			# Set all edge labels into a tuple. # @todo: remove this original labels or not?
			if self.verbose >= 2:
				iterator = get_iters(
					Gn, desc='Setting all edge labels into a tuple'
				)
			else:
				iterator = Gn
			for G in iterator:
				for n1, n2, attrs in G.edges(
						data=True
				):  # @todo: there may be a better way.
					G.edges[(n1, n2)]['lt'] = tuple(
						attrs[name] for name in self.edge_labels
					)

			# When h == 1, compute the kernel.
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			all_num_of_each_label = []  # number of occurence of each label in G

			# @todo: parallel this part.
			for G in Gn:
				num_of_labels_occured = self._subtree_1graph_labeled(
					G, all_set_compressed, all_num_of_each_label,
					num_of_labels_occured
				)

			# Compute subtree kernel with h iterations and add it to the final kernel.
			kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		# Iterate along heights.
		for h in range(2, self.height + 1):
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			all_num_of_each_label = []  # number of occurence of each label in G

			# @todo: parallel this part.
			for G in Gn:
				num_of_labels_occured = self._subtree_1graph_nl(
					G, all_set_compressed, all_num_of_each_label,
					num_of_labels_occured
				)

			# Compute subtree kernel with h iterations and add it to the final kernel.
			kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		return kernel_matrix


	def _subtree_kernel_do_unlabeled(self, Gn, return_mat=True):
		"""Compute Weisfeiler-Lehman kernels between graphs without labels.

		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are computed.

		Return
		------
		kernel_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		kernel_matrix = (np.zeros((len(Gn), len(Gn))) if return_mat else 0)
		gram_itr_fun = (
			self._compute_gram_itr if return_mat else self._compute_kernel_itr)

		# initial for height = 0
		all_num_of_each_label = []  # number of occurence of each label in each graph in this iteration

		# Compute subtree kernel with the 0th iteration and add it to the final kernel.
		# if return a kernel matrix:
		if return_mat:
			iterator = combinations_with_replacement(
				range(0, len(kernel_matrix)), 2
			)
			for i, j in iterator:
				kernel_matrix[i][j] += nx.number_of_nodes(
					Gn[i]
				) * nx.number_of_nodes(Gn[j])
				kernel_matrix[j][i] = kernel_matrix[i][j]
		# if return a single kernel between two graphs:
		else:
			kernel_matrix += nx.number_of_nodes(Gn[0]) * nx.number_of_nodes(
				Gn[1]
			)

		# if h >= 1.
		if self.height > 0:
			# When h == 1, compute the kernel.
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			all_num_of_each_label = []  # number of occurence of each label in G

			# @todo: parallel this part.
			for G in Gn:
				num_of_labels_occured = self._subtree_1graph_unlabeled(
					G, all_set_compressed, all_num_of_each_label,
					num_of_labels_occured
				)

			# Compute subtree kernel with h iterations and add it to the final kernel.
			kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		# Iterate along heights (>= 2).
		for h in range(2, self.height + 1):
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			all_num_of_each_label = []  # number of occurence of each label in G

			# @todo: parallel this part.
			for G in Gn:
				num_of_labels_occured = self._subtree_1graph_nl(
					G, all_set_compressed, all_num_of_each_label,
					num_of_labels_occured
				)

			# Compute subtree kernel with h iterations and add it to the final kernel.
			kernel_matrix = gram_itr_fun(kernel_matrix, all_num_of_each_label)

		return kernel_matrix


	def _subtree_1graph_nl(
			self, G, all_set_compressed, all_num_of_each_label,
			num_of_labels_occured
	):
		all_multisets = []
		for node, attrs in G.nodes(data=True):
			# Multiset-label determination.
			multiset = [G.nodes[neighbors]['lt'] for neighbors in G[node]]
			# sorting each multiset
			multiset.sort()
			multiset = [attrs['lt']] + multiset  # add the prefix
			all_multisets.append(tuple(multiset))

		# label compression
		set_unique = list(set(all_multisets))  # set of unique multiset labels
		# a dictionary mapping original labels to new ones.
		set_compressed = {}
		# If a label occured before, assign its former compressed label;
		# otherwise assign the number of labels occured + 1 as the
		# compressed label.
		for value in set_unique:
			if value in all_set_compressed.keys():  # @todo: put keys() function out of for loop?
				set_compressed[value] = all_set_compressed[value]
			else:
				set_compressed[value] = str(
					num_of_labels_occured + 1
				)  # @todo: remove str? and what if num_of_labels_occured is extremely big.
				num_of_labels_occured += 1

		all_set_compressed.update(set_compressed)

		# Relabel nodes.
		for idx, node in enumerate(G.nodes()):
			G.nodes[node]['lt'] = set_compressed[all_multisets[idx]]

		# Get the set of compressed labels.
		labels_comp = list(nx.get_node_attributes(G, 'lt').values())
		all_num_of_each_label.append(dict(Counter(labels_comp)))

		return num_of_labels_occured


	def _subtree_1graph_el(
			self, G, all_set_compressed, all_num_of_each_label,
			num_of_labels_occured
	):
		all_multisets = []
		# 		for node, attrs in G.nodes(data=True):
		for node in G.nodes():
			# Multiset-label determination.
			multiset = [G.edges[(node, neighbors)]['lt'] for neighbors in
			            G[node]]  # @todo: check reference for this.
			# sorting each multiset
			multiset.sort()
			# 			multiset = [attrs['lt']] + multiset # add the prefix
			all_multisets.append(tuple(multiset))

		# label compression
		set_unique = list(set(all_multisets))  # set of unique multiset labels
		# a dictionary mapping original labels to new ones.
		set_compressed = {}
		# If a label occured before, assign its former compressed label;
		# otherwise assign the number of labels occured + 1 as the
		# compressed label.
		for value in set_unique:
			if value in all_set_compressed.keys():  # @todo: put keys() function out of for loop?
				set_compressed[value] = all_set_compressed[value]
			else:
				set_compressed[value] = str(
					num_of_labels_occured + 1
				)  # @todo: remove str?
				num_of_labels_occured += 1

		all_set_compressed.update(set_compressed)

		# Relabel nodes.
		for idx, node in enumerate(G.nodes()):
			G.nodes[node]['lt'] = set_compressed[all_multisets[idx]]

		# Get the set of compressed labels.
		labels_comp = list(
			nx.get_node_attributes(G, 'lt').values()
		)  # @todo: maybe can be faster.
		all_num_of_each_label.append(dict(Counter(labels_comp)))

		return num_of_labels_occured


	def _subtree_1graph_labeled(
			self, G, all_set_compressed, all_num_of_each_label,
			num_of_labels_occured
	):
		all_multisets = []
		for node, attrs in G.nodes(data=True):
			# Multiset-label determination.
			multiset = [tuple(
				(G.edges[(node, neighbors)]['lt'], G.nodes[neighbors]['lt'])
			) for neighbors in G[node]]  # @todo: check reference for this.
			# sorting each multiset
			multiset.sort()
			multiset = [attrs['lt']] + multiset  # add the prefix
			all_multisets.append(tuple(multiset))

		# label compression
		set_unique = list(set(all_multisets))  # set of unique multiset labels
		# a dictionary mapping original labels to new ones.
		set_compressed = {}
		# If a label occured before, assign its former compressed label;
		# otherwise assign the number of labels occured + 1 as the
		# compressed label.
		for value in set_unique:
			if value in all_set_compressed.keys():  # @todo: put keys() function out of for loop?
				set_compressed[value] = all_set_compressed[value]
			else:
				set_compressed[value] = str(
					num_of_labels_occured + 1
				)  # @todo: remove str?
				num_of_labels_occured += 1

		all_set_compressed.update(set_compressed)

		# Relabel nodes.
		for idx, node in enumerate(G.nodes()):
			G.nodes[node]['lt'] = set_compressed[all_multisets[idx]]

		# Get the set of compressed labels.
		labels_comp = list(nx.get_node_attributes(G, 'lt').values())
		all_num_of_each_label.append(dict(Counter(labels_comp)))

		return num_of_labels_occured


	def _subtree_1graph_unlabeled(
			self, G, all_set_compressed, all_num_of_each_label,
			num_of_labels_occured
	):
		# 		all_multisets = []
		# 		for node, attrs in G.nodes(data=True): # @todo: it can be better.
		# 			# Multiset-label determination.
		# 			multiset = [0 for neighbors in G[node]]
		# 			# sorting each multiset
		# 			multiset.sort()
		# 			multiset = [0] + multiset # add the prefix
		# 			all_multisets.append(tuple(multiset))
		all_multisets = [len(G[node]) for node in G.nodes()]

		# label compression
		set_unique = list(set(all_multisets))  # set of unique multiset labels
		# a dictionary mapping original labels to new ones.
		set_compressed = {}
		# If a label occurred before, assign its former compressed label;
		# otherwise assign the number of labels occurred + 1 as the
		# compressed label.
		for value in set_unique:
			if value in all_set_compressed.keys():  # @todo: put keys() function out of for loop?
				set_compressed[value] = all_set_compressed[value]
			else:
				set_compressed[value] = str(
					num_of_labels_occured + 1
				)  # @todo: remove str?
				num_of_labels_occured += 1

		all_set_compressed.update(set_compressed)

		# Relabel nodes.
		for idx, node in enumerate(G.nodes()):
			G.nodes[node]['lt'] = set_compressed[all_multisets[idx]]

		# Get the set of compressed labels.
		labels_comp = list(nx.get_node_attributes(G, 'lt').values())
		all_num_of_each_label.append(dict(Counter(labels_comp)))

		return num_of_labels_occured


	def _compute_gram_itr(self, gram_matrix, all_num_of_each_label):
		"""Compute Gram matrix using the base kernel.
		"""
		#		if self.parallel == 'imap_unordered':
		#			# compute kernels.
		#			def init_worker(alllabels_toshare):
		#				global G_alllabels
		#				G_alllabels = alllabels_toshare
		#			do_partial = partial(self._wrapper_compute_subtree_kernel, gram_matrix)
		#			parallel_gm(do_partial, gram_matrix, Gn, init_worker=init_worker,
		#						glbv=(all_num_of_each_label,), n_jobs=self.n_jobs, verbose=self.verbose)
		#		elif self.parallel is None:
		itr = combinations_with_replacement(range(0, len(gram_matrix)), 2)
		len_itr = int(len(gram_matrix) * (len(gram_matrix) + 1) / 2)
		iterator = get_iters(
			itr, desc='Computing Gram matrix for this iteration',
			file=sys.stdout, length=len_itr, verbose=(self.verbose >= 2)
		)
		for i, j in iterator:
			# 		for i in iterator:
			# 			for j in range(i, len(gram_matrix)):
			gram_matrix[i][j] += self._compute_subtree_kernel(
				all_num_of_each_label[i],
				all_num_of_each_label[j]
			)
			gram_matrix[j][i] = gram_matrix[i][j]

		return gram_matrix


	def _compute_subtree_kernel(self, num_of_each_label1, num_of_each_label2):
		"""Compute the subtree kernel.
		"""
		labels = set(
			list(num_of_each_label1.keys()) + list(num_of_each_label2.keys())
		)
		vector1 = np.array(
			[(num_of_each_label1[label]
			  if (label in num_of_each_label1.keys()) else 0)
			 for label in labels]
		)
		vector2 = np.array(
			[(num_of_each_label2[label]
			  if (label in num_of_each_label2.keys()) else 0)
			 for label in labels]
		)
		kernel = np.dot(vector1, vector2)
		return kernel


	#	def _wrapper_compute_subtree_kernel(self, gram_matrix, itr):
	#		i = itr[0]
	#		j = itr[1]
	#		return i, j, self._compute_subtree_kernel(G_alllabels[i], G_alllabels[j], gram_matrix[i][j])

	def _wl_spkernel_do(Gn, node_label, edge_label, height):
		"""Compute Weisfeiler-Lehman shortest path kernels between graphs.

		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are computed.
		node_label : string
			node attribute used as label.
		edge_label : string
			edge attribute used as label.
		height : int
			subtree height.

		Return
		------
		gram_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		pass
		from gklearn.utils.utils import getSPGraph

		# init.
		height = int(height)
		gram_matrix = np.zeros((len(Gn), len(Gn)))  # init kernel

		Gn = [getSPGraph(G, edge_weight=edge_label) for G in
		      Gn]  # get shortest path graphs of Gn

		# initial for height = 0
		for i in range(0, len(Gn)):
			for j in range(i, len(Gn)):
				for e1 in Gn[i].edges(data=True):
					for e2 in Gn[j].edges(data=True):
						if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2][
							'cost'] and (
								(e1[0] == e2[0] and e1[1] == e2[1]) or (
								e1[0] == e2[1] and e1[1] == e2[0])):
							gram_matrix[i][j] += 1
				gram_matrix[j][i] = gram_matrix[i][j]

		# iterate each height
		for h in range(1, height + 1):
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			for G in Gn:  # for each graph
				set_multisets = []
				for node in G.nodes(data=True):
					# Multiset-label determination.
					multiset = [G.node[neighbors][node_label] for neighbors in
					            G[node[0]]]
					# sorting each multiset
					multiset.sort()
					multiset = node[1][node_label] + ''.join(
						multiset
					)  # concatenate to a string and add the prefix
					set_multisets.append(multiset)

				# label compression
				set_unique = list(
					set(set_multisets)
				)  # set of unique multiset labels
				# a dictionary mapping original labels to new ones.
				set_compressed = {}
				# if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label
				for value in set_unique:
					if value in all_set_compressed.keys():
						set_compressed[value] = all_set_compressed[value]
					else:
						set_compressed[value] = str(num_of_labels_occured + 1)
						num_of_labels_occured += 1

				all_set_compressed.update(set_compressed)

				# relabel nodes
				for node in G.nodes(data=True):
					node[1][node_label] = set_compressed[set_multisets[node[0]]]

			# Compute subtree kernel with h iterations and add it to the final kernel
			for i in range(0, len(Gn)):
				for j in range(i, len(Gn)):
					for e1 in Gn[i].edges(data=True):
						for e2 in Gn[j].edges(data=True):
							if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2][
								'cost'] and (
									(e1[0] == e2[0] and e1[1] == e2[1]) or (
									e1[0] == e2[1] and e1[1] == e2[0])):
								gram_matrix[i][j] += 1
					gram_matrix[j][i] = gram_matrix[i][j]

		return gram_matrix


	def _wl_edgekernel_do(Gn, node_label, edge_label, height):
		"""Compute Weisfeiler-Lehman edge kernels between graphs.

		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are computed.
		node_label : string
			node attribute used as label.
		edge_label : string
			edge attribute used as label.
		height : int
			subtree height.

		Return
		------
		gram_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		pass
		# init.
		height = int(height)
		gram_matrix = np.zeros((len(Gn), len(Gn)))  # init kernel

		# initial for height = 0
		for i in range(0, len(Gn)):
			for j in range(i, len(Gn)):
				for e1 in Gn[i].edges(data=True):
					for e2 in Gn[j].edges(data=True):
						if e1[2][edge_label] == e2[2][edge_label] and (
								(e1[0] == e2[0] and e1[1] == e2[1]) or (
								e1[0] == e2[1] and e1[1] == e2[0])):
							gram_matrix[i][j] += 1
				gram_matrix[j][i] = gram_matrix[i][j]

		# iterate each height
		for h in range(1, height + 1):
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			for G in Gn:  # for each graph
				set_multisets = []
				for node in G.nodes(data=True):
					# Multiset-label determination.
					multiset = [G.node[neighbors][node_label] for neighbors in
					            G[node[0]]]
					# sorting each multiset
					multiset.sort()
					multiset = node[1][node_label] + ''.join(
						multiset
					)  # concatenate to a string and add the prefix
					set_multisets.append(multiset)

				# label compression
				set_unique = list(
					set(set_multisets)
				)  # set of unique multiset labels
				# a dictionary mapping original labels to new ones.
				set_compressed = {}
				# if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label
				for value in set_unique:
					if value in all_set_compressed.keys():
						set_compressed[value] = all_set_compressed[value]
					else:
						set_compressed[value] = str(num_of_labels_occured + 1)
						num_of_labels_occured += 1

				all_set_compressed.update(set_compressed)

				# relabel nodes
				for node in G.nodes(data=True):
					node[1][node_label] = set_compressed[set_multisets[node[0]]]

			# Compute subtree kernel with h iterations and add it to the final kernel
			for i in range(0, len(Gn)):
				for j in range(i, len(Gn)):
					for e1 in Gn[i].edges(data=True):
						for e2 in Gn[j].edges(data=True):
							if e1[2][edge_label] == e2[2][edge_label] and (
									(e1[0] == e2[0] and e1[1] == e2[1]) or (
									e1[0] == e2[1] and e1[1] == e2[0])):
								gram_matrix[i][j] += 1
					gram_matrix[j][i] = gram_matrix[i][j]

		return gram_matrix


	def _wl_userkernel_do(Gn, node_label, edge_label, height, base_kernel):
		"""Compute Weisfeiler-Lehman kernels based on user-defined kernel between graphs.

		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are computed.
		node_label : string
			node attribute used as label.
		edge_label : string
			edge attribute used as label.
		height : int
			subtree height.
		base_kernel : string
			Name of the base kernel function used in each iteration of WL kernel. This function returns a Numpy matrix, each element of which is the user-defined Weisfeiler-Lehman kernel between 2 praphs.

		Return
		------
		gram_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		pass
		# init.
		height = int(height)
		gram_matrix = np.zeros((len(Gn), len(Gn)))  # init kernel

		# initial for height = 0
		gram_matrix = base_kernel(Gn, node_label, edge_label)

		# iterate each height
		for h in range(1, height + 1):
			all_set_compressed = {}  # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0  # number of the set of letters that occur before as node labels at least once in all graphs
			for G in Gn:  # for each graph
				set_multisets = []
				for node in G.nodes(data=True):
					# Multiset-label determination.
					multiset = [G.node[neighbors][node_label] for neighbors in
					            G[node[0]]]
					# sorting each multiset
					multiset.sort()
					multiset = node[1][node_label] + ''.join(
						multiset
					)  # concatenate to a string and add the prefix
					set_multisets.append(multiset)

				# label compression
				set_unique = list(
					set(set_multisets)
				)  # set of unique multiset labels
				# a dictionary mapping original labels to new ones.
				set_compressed = {}
				# if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label
				for value in set_unique:
					if value in all_set_compressed.keys():
						set_compressed[value] = all_set_compressed[value]
					else:
						set_compressed[value] = str(num_of_labels_occured + 1)
						num_of_labels_occured += 1

				all_set_compressed.update(set_compressed)

				# relabel nodes
				for node in G.nodes(data=True):
					node[1][node_label] = set_compressed[set_multisets[node[0]]]

			# Compute kernel with h iterations and add it to the final kernel
			gram_matrix += base_kernel(Gn, node_label, edge_label)

		return gram_matrix


	def _add_dummy_node_labels(self, Gn):
		if len(self.node_labels) == 0 or (
				len(self.node_labels) == 1 and self.node_labels[
			0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self.node_labels = [SpecialLabel.DUMMY]


class WLSubtree(WeisfeilerLehman):

	def __init__(self, **kwargs):
		kwargs['base_kernel'] = 'subtree'
		super().__init__(**kwargs)
