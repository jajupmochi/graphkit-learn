#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:09:51 2020

@author: ljia

@references:

	[1] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11(Apr):1201–1242, 2010.
"""

import sys
from gklearn.utils import get_iters
import numpy as np
import networkx as nx
from scipy import optimize
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.kernels import RandomWalkMeta
from gklearn.utils.utils import compute_vertex_kernels



class FixedPoint(RandomWalkMeta):


	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._node_kernels = kwargs.get('node_kernels', None)
		self._edge_kernels = kwargs.get('edge_kernels', None)
		self._node_labels = kwargs.get('node_labels', [])
		self._edge_labels = kwargs.get('edge_labels', [])
		self._node_attrs = kwargs.get('node_attrs', [])
		self._edge_attrs = kwargs.get('edge_attrs', [])


	def _compute_gm_series(self, graphs):
		self._check_edge_weight(graphs, self.verbose)
		self._check_graphs(graphs)

		lmda = self._weight

		# Compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		# Reindex nodes using consecutive integers for the convenience of kernel computation.
		iterator = get_iters(graphs, desc='Reindex vertices', file=sys.stdout,verbose=(self.verbose >= 2))
		graphs = [nx.convert_node_labels_to_integers(g, first_label=0, label_attribute='label_orignal') for g in iterator]

		if self._p is None and self._q is None: # p and q are uniform distributions as default.

			from itertools import combinations_with_replacement
			itr = combinations_with_replacement(range(0, len(graphs)), 2)
			len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
			iterator = get_iters(itr, desc='Computing kernels', file=sys.stdout, length=len_itr, verbose=(self.verbose >= 2))

			for i, j in iterator:
				kernel = self._kernel_do(graphs[i], graphs[j], lmda)
				gram_matrix[i][j] = kernel
				gram_matrix[j][i] = kernel

		else: # @todo
			pass

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		self._check_edge_weight(self._graphs, self.verbose)
		self._check_graphs(self._graphs)

		# Compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		# @todo: parallel this.
		# Reindex nodes using consecutive integers for the convenience of kernel computation.
		iterator = get_iters(self._graphs, desc='Reindex vertices', file=sys.stdout, verbose=(self.verbose >= 2))
		self._graphs = [nx.convert_node_labels_to_integers(g, first_label=0, label_attribute='label_orignal') for g in iterator]

		if self._p is None and self._q is None: # p and q are uniform distributions as default.

			def init_worker(gn_toshare):
				global G_gn
				G_gn = gn_toshare

			do_fun = self._wrapper_kernel_do

			parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=init_worker,
						glbv=(self._graphs,), n_jobs=self.n_jobs, verbose=self.verbose)

		else: # @todo
			pass

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		self._check_edge_weight(g_list + [g1], self.verbose)
		self._check_graphs(g_list + [g1])

		lmda = self._weight

		# compute kernel list.
		kernel_list = [None] * len(g_list)

		# Reindex nodes using consecutive integers for the convenience of kernel computation.
		g1 = nx.convert_node_labels_to_integers(g1, first_label=0, label_attribute='label_orignal')
		iterator = get_iters(g_list, desc='Reindex vertices', file=sys.stdout, verbose=(self.verbose >= 2))
		g_list = [nx.convert_node_labels_to_integers(g, first_label=0, label_attribute='label_orignal') for g in iterator]

		if self._p is None and self._q is None: # p and q are uniform distributions as default.

			iterator = get_iters(range(len(g_list)), desc='Computing kernels', file=sys.stdout, length=len(g_list), verbose=(self.verbose >= 2))

			for i in iterator:
				kernel = self._kernel_do(g1, g_list[i], lmda)
				kernel_list[i] = kernel

		else: # @todo
			pass

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._check_edge_weight(g_list + [g1], self.verbose)
		self._check_graphs(g_list + [g1])

		# compute kernel list.
		kernel_list = [None] * len(g_list)

		# Reindex nodes using consecutive integers for the convenience of kernel computation.
		g1 = nx.convert_node_labels_to_integers(g1, first_label=0, label_attribute='label_orignal')
		# @todo: parallel this.
		iterator = get_iters(g_list, desc='Reindex vertices', file=sys.stdout, verbose=(self.verbose >= 2))
		g_list = [nx.convert_node_labels_to_integers(g, first_label=0, label_attribute='label_orignal') for g in iterator]

		if self._p is None and self._q is None: # p and q are uniform distributions as default.

			def init_worker(g1_toshare, g_list_toshare):
				global G_g1, G_g_list
				G_g1 = g1_toshare
				G_g_list = g_list_toshare

			do_fun = self._wrapper_kernel_list_do

			def func_assign(result, var_to_assign):
				var_to_assign[result[0]] = result[1]
			itr = range(len(g_list))
			len_itr = len(g_list)
			parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
				init_worker=init_worker, glbv=(g1, g_list), method='imap_unordered',
				n_jobs=self.n_jobs, itr_desc='Computing kernels', verbose=self.verbose)

		else: # @todo
			pass

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		return itr, self._kernel_do(G_g1, G_g_list[itr], self._weight)


	def _compute_single_kernel_series(self, g1, g2):
		self._check_edge_weight([g1] + [g2], self.verbose)
		self._check_graphs([g1] + [g2])

		lmda = self._weight

		# Reindex nodes using consecutive integers for the convenience of kernel computation.
		g1 = nx.convert_node_labels_to_integers(g1, first_label=0, label_attribute='label_orignal')
		g2 = nx.convert_node_labels_to_integers(g2, first_label=0, label_attribute='label_orignal')

		if self._p is None and self._q is None: # p and q are uniform distributions as default.
			kernel = self._kernel_do(g1, g2, lmda)

		else: # @todo
			pass

		return kernel


	def _kernel_do(self, g1, g2, lmda):

		# Frist, compute kernels between all pairs of nodes using the method borrowed
		# from FCSP. It is faster than directly computing all edge kernels
		# when $d_1d_2>2$, where $d_1$ and $d_2$ are vertex degrees of the
		# graphs compared, which is the most case we went though. For very
		# sparse graphs, this would be slow.
		vk_dict = self._compute_vertex_kernels(g1, g2)

		# Compute the weight matrix of the direct product graph.
		w_times, w_dim = self._compute_weight_matrix(g1, g2, vk_dict)
		# use uniform distribution if there is no prior knowledge.
		p_times_uni = 1 / w_dim
		p_times = np.full((w_dim, 1), p_times_uni)
		x = optimize.fixed_point(self._func_fp, p_times, args=(p_times, lmda, w_times), xtol=1e-06, maxiter=1000)
		# use uniform distribution if there is no prior knowledge.
		q_times = np.full((1, w_dim), p_times_uni)
		return np.dot(q_times, x)


	def _wrapper_kernel_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do(G_gn[i], G_gn[j], self._weight)


	def _func_fp(self, x, p_times, lmda, w_times):
		haha = w_times * x
		haha = lmda * haha
		haha = p_times + haha
		return p_times + lmda * np.dot(w_times, x)


	def _compute_vertex_kernels(self, g1, g2):
		"""Compute vertex kernels between vertices of two graphs.
		"""
		return compute_vertex_kernels(g1, g2, self._node_kernels, node_labels=self._node_labels, node_attrs=self._node_attrs)


	# @todo: move if out to make it faster.
	# @todo: node/edge kernels use direct function rather than dicts.
	def _compute_weight_matrix(self, g1, g2, vk_dict):
		"""Compute the weight matrix of the direct product graph.
		"""
		# Define edge kernels.
		def compute_ek_11(e1, e2, ke):
			e1_labels = [e1[2][el] for el in self._edge_labels]
			e2_labels = [e2[2][el] for el in self._edge_labels]
			e1_attrs = [e1[2][ea] for ea in self._edge_attrs]
			e2_attrs = [e2[2][ea] for ea in self._edge_attrs]
			return ke(e1_labels, e2_labels, e1_attrs, e2_attrs)

		def compute_ek_10(e1, e2, ke):
			e1_labels = [e1[2][el] for el in self._edge_labels]
			e2_labels = [e2[2][el] for el in self._edge_labels]
			return ke(e1_labels, e2_labels)

		def compute_ek_01(e1, e2, ke):
			e1_attrs = [e1[2][ea] for ea in self._edge_attrs]
			e2_attrs = [e2[2][ea] for ea in self._edge_attrs]
			return ke(e1_attrs, e2_attrs)

		def compute_ek_00(e1, e2, ke):
			return 1

		# Select the proper edge kernel.
		if len(self._edge_labels) > 0:
			# edge symb and non-synb labeled
			if len(self._edge_attrs) > 0:
				ke = self._edge_kernels['mix']
				ek_temp = compute_ek_11
			# edge symb labeled
			else:
				ke = self._edge_kernels['symb']
				ek_temp = compute_ek_10
		else:
			# edge non-synb labeled
			if len(self._edge_attrs) > 0:
				ke = self._edge_kernels['nsymb']
				ek_temp = compute_ek_01
			# edge unlabeled
			else:
				ke = None
				ek_temp = compute_ek_00 # @todo: check how much slower is this.

		# Compute the weight matrix.
		w_dim = nx.number_of_nodes(g1) * nx.number_of_nodes(g2)
		w_times = np.zeros((w_dim, w_dim))

		if vk_dict: # node labeled
			if self._ds_infos['directed']:
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						w_idx = (e1[0] * nx.number_of_nodes(g2) + e2[0], e1[1] * nx.number_of_nodes(g2) + e2[1])
						w_times[w_idx] = vk_dict[(e1[0], e2[0])] * ek_temp(e1, e2, ke) * vk_dict[(e1[1], e2[1])]
			else: # undirected
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						w_idx = (e1[0] * nx.number_of_nodes(g2) + e2[0], e1[1] * nx.number_of_nodes(g2) + e2[1])
						w_times[w_idx] = vk_dict[(e1[0], e2[0])] * ek_temp(e1, e2, ke) * vk_dict[(e1[1], e2[1])] + vk_dict[(e1[0], e2[1])] * ek_temp(e1, e2, ke) * vk_dict[(e1[1], e2[0])]
						w_times[w_idx[1], w_idx[0]] = w_times[w_idx[0], w_idx[1]]
						w_idx2 = (e1[0] * nx.number_of_nodes(g2) + e2[1], e1[1] * nx.number_of_nodes(g2) + e2[0])
						w_times[w_idx2[0], w_idx2[1]] = w_times[w_idx[0], w_idx[1]]
						w_times[w_idx2[1], w_idx2[0]] = w_times[w_idx[0], w_idx[1]]
		else: # node unlabeled
			if self._ds_infos['directed']:
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						w_idx = (e1[0] * nx.number_of_nodes(g2) + e2[0], e1[1] * nx.number_of_nodes(g2) + e2[1])
						w_times[w_idx] = ek_temp(e1, e2, ke)
			else: # undirected
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						w_idx = (e1[0] * nx.number_of_nodes(g2) + e2[0], e1[1] * nx.number_of_nodes(g2) + e2[1])
						w_times[w_idx] = ek_temp(e1, e2, ke)
						w_times[w_idx[1], w_idx[0]] = w_times[w_idx[0], w_idx[1]]
						w_idx2 = (e1[0] * nx.number_of_nodes(g2) + e2[1], e1[1] * nx.number_of_nodes(g2) + e2[0])
						w_times[w_idx2[0], w_idx2[1]] = w_times[w_idx[0], w_idx[1]]
						w_times[w_idx2[1], w_idx2[0]] = w_times[w_idx[0], w_idx[1]]

		return w_times, w_dim
