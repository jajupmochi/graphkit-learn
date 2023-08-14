#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:24:58 2020

@author: ljia

@references:
	[1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData
	Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
"""

import sys
from itertools import product
from multiprocessing import Pool
from gklearn.utils import get_iters
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np
import networkx as nx
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.utils.utils import get_sp_graph
from gklearn.kernels import GraphKernel


class ShortestPath(GraphKernel):

	def __init__(self, **kwargs):
		GraphKernel.__init__(
			self, **{
				k: kwargs.get(k) for k in
				['parallel', 'n_jobs', 'chunksize', 'normalize', 'copy_graphs',
				 'verbose'] if k in kwargs}
		)
		self._node_labels = kwargs.get('node_labels', [])
		self._node_attrs = kwargs.get('node_attrs', [])
		self._edge_weight = kwargs.get('edge_weight', None)
		self._node_kernels = kwargs.get('node_kernels', None)
		self._fcsp = kwargs.get('fcsp', True)
		self._ds_infos = kwargs.get('ds_infos', {})
		self._save_sp_graphs = kwargs.get('save_sp_graphs', True)
		self._sp_func = get_sp_graph
		self._kernel_do_func = self._sp_do


	##########################################################################
	# The following is the 1st paradigm to compute kernel matrix, which is
	# compatible with `scikit-learn`.
	# -------------------------------------------------------------------
	# Special thanks to the "GraKeL" library for providing an excellent template!
	##########################################################################


	def clear_attributes(self):
		super().clear_attributes()
		if hasattr(self, '_sp_graphs'):
			delattr(self, '_sp_graphs')
		if hasattr(self, '_Y_sp_graphs'):
			delattr(self, '_Y_sp_graphs')


	def validate_parameters(self):
		super().validate_parameters()


	# if self._depth < 1:
	# 	raise ValueError('`depth` must be greater than 0.')
	# if self._k_func not in ['MinMax', 'tanimoto']:
	# 	raise ValueError('`k_func` must be either `MinMax` or `tanimoto`.')
	# if self._compute_method not in ['trie']:
	# 	raise ValueError('`compute_method` must be `trie`.')


	def _compute_kernel_matrix_series(self, Y, X=None, load_sp_graphs=True):
		"""Compute the kernel matrix between a given target graphs (Y) and
		the fitted graphs (X / self._graphs) without parallelization.

		Parameters
		----------
		Y : list of graphs, optional
			The target graphs.

		Returns
		-------
		kernel_matrix : numpy array, shape = [n_targets, n_inputs]
			The computed kernel matrix.

		"""
		if_comp_X_sp_graphs = True

		# if load saved sp_graphs of X from the instance:
		if load_sp_graphs:
			# sp_graphs for self._graphs.
			try:
				check_is_fitted(self, ['_sp_graphs'])
				sp_graphs_list1 = self._sp_graphs
				if_comp_X_sp_graphs = False
			except NotFittedError:
				import warnings
				warnings.warn(
					'The sp_graphs of self._graphs are not computed/saved. '
					'The sp_graphs of `X` is computed instead.'
				)
				if_comp_X_sp_graphs = True

		# Get all sp_graphs of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.

		# Compute the sp_graphs of X.
		if if_comp_X_sp_graphs:
			if X is None:
				raise ('X can not be None.')
			# self._add_dummy_labels will modify the input in place.
			self._all_graphs_have_edges(X)  # for X
			sp_graphs_list1 = []
			iterator = get_iters(
				self._graphs, desc='Getting sp_graphs for X',
				file=sys.stdout, verbose=(self.verbose >= 2)
			)
			for g in iterator:
				sp_graphs_list1.append(self._sp_func(g))

		# sp_graphs for Y.
		# 		Y = [g.copy() for g in Y] # @todo: ?
		self._all_graphs_have_edges(Y)
		sp_graphs_list2 = []
		iterator = get_iters(
			Y, desc='Getting sp_graphs for Y', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		for g in iterator:
			sp_graphs_list2.append(self._sp_func(g))

		# 		if self.save_sp_graphs:
		# 			self._Y_sp_graphs = sp_graphs_list2

		# compute kernel matrix.
		kernel_matrix = np.zeros((len(Y), len(sp_graphs_list1)))

		from itertools import product
		itr = product(range(len(Y)), range(len(sp_graphs_list1)))
		len_itr = int(len(Y) * len(sp_graphs_list1))
		iterator = get_iters(
			itr, desc='Computing kernels', file=sys.stdout,
			length=len_itr, verbose=(self.verbose >= 2)
		)
		for i_y, i_x in iterator:
			kernel = self._kernel_do_func(
				sp_graphs_list2[i_y], sp_graphs_list1[i_x]
			)
			kernel_matrix[i_y][i_x] = kernel

		return kernel_matrix


	def _compute_kernel_matrix_imap_unordered(self, Y):
		"""Compute the kernel matrix between a given target graphs (Y) and
		the fitted graphs (X / self._graphs) using imap unordered parallelization.

		Parameters
		----------
		Y : list of graphs, optional
			The target graphs.

		Returns
		-------
		kernel_matrix : numpy array, shape = [n_targets, n_inputs]
			The computed kernel matrix.

		"""
		raise NotImplementedError(
			'Parallelization for kernel matrix is not implemented.'
		)


	def pairwise_kernel(self, x, y, are_sp_graphs=False):
		"""Compute pairwise kernel between two graphs.

		Parameters
		----------
		x, y : NetworkX Graph.
			Graphs bewteen which the kernel is computed.

		are_sp_graphs : boolean, optional
			If `True`, `x` and `y` are sp graphs, otherwise are graphs.
			The default is False.

		Returns
		-------
		kernel: float
			The computed kernel.

		"""
		if are_sp_graphs:
			# x, y are canonical keys.
			kernel = self._kernel_do_func(x, y)

		else:
			# x, y are graphs.
			kernel = self._compute_single_kernel_series(x, y)

		return kernel


	def diagonals(self):
		"""Compute the kernel matrix diagonals of the fit/transformed data.

		Returns
		-------
		X_diag : numpy array
			The diagonal of the kernel matrix between the fitted data.
			This consists of each element calculated with itself.

		Y_diag : numpy array
			The diagonal of the kernel matrix, of the transform.
			This consists of each element calculated with itself.
		"""
		# Check if method "fit" had been called.
		check_is_fitted(self, ['_graphs'])

		# Check if the diagonals of X exist.
		try:
			check_is_fitted(self, ['_X_diag'])
		except NotFittedError:
			# Compute diagonals of X.
			self._X_diag = np.empty(shape=(len(self._graphs),))
			try:
				check_is_fitted(self, ['_all_sp_graphs'])
				for i, x in enumerate(self._all_graphs):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_sp_graphs=True
					)  # @todo: parallel?
			except NotFittedError:
				for i, x in enumerate(self._graphs):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_sp_graphs=False
					)  # @todo: parallel?

		try:
			# If transform has happened, return both diagonals.
			check_is_fitted(self, ['_Y'])
			self._Y_diag = np.empty(shape=(len(self._Y),))
			try:
				check_is_fitted(self, ['_Y_all_sp_graphs'])
				for (i, y) in enumerate(self._Y_all_sp_graphs):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_sp_graphs=True
					)  # @todo: parallel?
			except NotFittedError:
				for (i, y) in enumerate(self._Y):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_sp_graphs=False
					)  # @todo: parallel?

			return self._X_diag, self._Y_diag

		except NotFittedError:
			# Else just return both X_diag
			return self._X_diag


	##########################################################################
	# The following is the 2nd paradigm to compute kernel matrix. It is
	# simplified and not compatible with `scikit-learn`.
	##########################################################################


	def _compute_gm_series(self, graphs):
		self._all_graphs_have_edges(graphs)
		# get shortest path graph of each graph.
		iterator = get_iters(
			graphs, desc='getting sp graphs', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		graphs = [
			get_sp_graph(g, edge_weight=self._edge_weight) for g in
			iterator
		]
		if self._save_sp_graphs:
			self._sp_graphs = graphs

		# compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(graphs)), 2)
		len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
		iterator = get_iters(
			itr, desc='Computing kernels',
			length=len_itr, file=sys.stdout, verbose=(self.verbose >= 2)
		)
		for i, j in iterator:
			kernel = self._sp_do(graphs[i], graphs[j])
			gram_matrix[i][j] = kernel
			gram_matrix[j][i] = kernel

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		self._all_graphs_have_edges(self._graphs)
		# get shortest path graph of each graph.
		pool = Pool(self.n_jobs)
		get_sp_graphs_fun = self._wrapper_get_sp_graphs
		itr = zip(self._graphs, range(0, len(self._graphs)))
		if len(self._graphs) < 100 * self.n_jobs:
			chunksize = int(len(self._graphs) / self.n_jobs) + 1
		else:
			chunksize = 100
		iterator = get_iters(
			pool.imap_unordered(get_sp_graphs_fun, itr, chunksize),
			desc='getting sp graphs', file=sys.stdout,
			length=len(self._graphs), verbose=(self.verbose >= 2)
		)
		for i, g in iterator:
			self._graphs[i] = g
		pool.close()
		pool.join()

		if self._save_sp_graphs:
			self._sp_graphs = self._graphs  # @TODO: deepcopy? save original graphs?

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))


		def init_worker(gs_toshare):
			global G_gs
			G_gs = gs_toshare


		do_fun = self._wrapper_sp_do
		parallel_gm(
			do_fun, gram_matrix, self._graphs, init_worker=init_worker,
			glbv=(self._graphs,), n_jobs=self.n_jobs, verbose=self.verbose
		)

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		self._all_graphs_have_edges([g1] + g_list)
		# get shortest path graphs of g1 and each graph in g_list.
		g1 = get_sp_graph(g1, edge_weight=self._edge_weight)
		iterator = get_iters(
			g_list, desc='getting sp graphs', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		g_list = [
			get_sp_graph(g, edge_weight=self._edge_weight) for g in
			iterator
		]

		# compute kernel list.
		kernel_list = [None] * len(g_list)
		iterator = get_iters(
			range(len(g_list)), desc='Computing kernels', file=sys.stdout,
			length=len(g_list), verbose=(self.verbose >= 2)
		)
		for i in iterator:
			kernel = self._sp_do(g1, g_list[i])
			kernel_list[i] = kernel

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._all_graphs_have_edges([g1] + g_list)
		# get shortest path graphs of g1 and each graph in g_list.
		g1 = get_sp_graph(g1, edge_weight=self._edge_weight)
		pool = Pool(self.n_jobs)
		get_sp_graphs_fun = self._wrapper_get_sp_graphs
		itr = zip(g_list, range(0, len(g_list)))
		if len(g_list) < 100 * self.n_jobs:
			chunksize = int(len(g_list) / self.n_jobs) + 1
		else:
			chunksize = 100
		iterator = get_iters(
			pool.imap_unordered(get_sp_graphs_fun, itr, chunksize),
			desc='getting sp graphs', file=sys.stdout,
			length=len(g_list), verbose=(self.verbose >= 2)
		)
		for i, g in iterator:
			g_list[i] = g
		pool.close()
		pool.join()

		# compute Gram matrix.
		kernel_list = [None] * len(g_list)


		def init_worker(g1_toshare, gl_toshare):
			global G_g1, G_gl
			G_g1 = g1_toshare
			G_gl = gl_toshare


		do_fun = self._wrapper_kernel_list_do


		def func_assign(result, var_to_assign):
			var_to_assign[result[0]] = result[1]


		itr = range(len(g_list))
		len_itr = len(g_list)
		parallel_me(
			do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(g1, g_list), method='imap_unordered',
			n_jobs=self.n_jobs, itr_desc='Computing kernels',
			verbose=self.verbose
		)

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		return itr, self._sp_do(G_g1, G_gl[itr])


	def _compute_single_kernel_series(self, g1, g2):
		self._all_graphs_have_edges([g1] + [g2])
		g1 = get_sp_graph(g1, edge_weight=self._edge_weight)
		g2 = get_sp_graph(g2, edge_weight=self._edge_weight)
		kernel = self._sp_do(g1, g2)
		return kernel


	def _wrapper_get_sp_graphs(self, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, get_sp_graph(g, edge_weight=self._edge_weight)


	def _sp_do(self, g1, g2):

		if self._fcsp:  # @todo: it may be put outside the _sp_do().
			return self._sp_do_fcsp(g1, g2)
		else:
			return self._sp_do_naive(g1, g2)


	def _sp_do_fcsp(self, g1, g2):

		kernel = 0

		# compute shortest path matrices first, method borrowed from FCSP.
		vk_dict = {}  # shortest path matrices dict
		if len(
				self._node_labels
		) > 0:  # @todo: it may be put outside the _sp_do().
			# node symb and non-synb labeled
			if len(self._node_attrs) > 0:
				kn = self._node_kernels['mix']
				for n1, n2 in product(
						g1.nodes(data=True), g2.nodes(data=True)
				):
					n1_labels = [n1[1][nl] for nl in self._node_labels]
					n2_labels = [n2[1][nl] for nl in self._node_labels]
					# @TODO: reformat attrs during data processing a priori to save time.
					n1_attrs = np.array(
						[n1[1][na] for na in self._node_attrs]
					).astype(float)
					n2_attrs = np.array(
						[n2[1][na] for na in self._node_attrs]
					).astype(float)
					vk_dict[(n1[0], n2[0])] = kn(
						n1_labels, n2_labels, n1_attrs, n2_attrs
					)
			# node symb labeled
			else:
				kn = self._node_kernels['symb']
				for n1 in g1.nodes(data=True):
					for n2 in g2.nodes(data=True):
						n1_labels = [n1[1][nl] for nl in self._node_labels]
						n2_labels = [n2[1][nl] for nl in self._node_labels]
						vk_dict[(n1[0], n2[0])] = kn(n1_labels, n2_labels)
		else:
			# node non-synb labeled
			if len(self._node_attrs) > 0:
				kn = self._node_kernels['nsymb']
				for n1 in g1.nodes(data=True):
					for n2 in g2.nodes(data=True):
						n1_attrs = np.array(
							[n1[1][na] for na in self._node_attrs]
						).astype(float)
						n2_attrs = np.array(
							[n2[1][na] for na in self._node_attrs]
						).astype(float)
						vk_dict[(n1[0], n2[0])] = kn(n1_attrs, n2_attrs)
			# node unlabeled
			else:
				for e1, e2 in product(
						g1.edges(data=True), g2.edges(data=True)
				):
					if e1[2]['cost'] == e2[2]['cost']:
						kernel += 1
				return kernel

		# compute graph kernels
		if self._ds_infos['directed']:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					nk11, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[
						(e1[1], e2[1])]
					kn1 = nk11 * nk22
					kernel += kn1
		else:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					# each edge walk is counted twice, starting from both its extreme nodes.
					nk11, nk12, nk21, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(
						e1[0], e2[1])], vk_dict[(e1[1], e2[0])], vk_dict[
						(e1[1], e2[1])]
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


	def _sp_do_naive(self, g1, g2):

		kernel = 0

		# Define the function to compute kernels between vertices in each condition.
		if len(self._node_labels) > 0:
			# node symb and non-synb labeled
			if len(self._node_attrs) > 0:
				def compute_vk(n1, n2):
					kn = self._node_kernels['mix']
					n1_labels = [g1.nodes[n1][nl] for nl in self._node_labels]
					n2_labels = [g2.nodes[n2][nl] for nl in self._node_labels]
					# @TODO: reformat attrs during data processing a priori to save time.
					n1_attrs = np.array(
						[g1.nodes[n1][na] for na in self._node_attrs]
					).astype(float)
					n2_attrs = np.array(
						[g2.nodes[n2][na] for na in self._node_attrs]
					).astype(float)
					return kn(n1_labels, n2_labels, n1_attrs, n2_attrs)
			# node symb labeled
			else:
				def compute_vk(n1, n2):
					kn = self._node_kernels['symb']
					n1_labels = [g1.nodes[n1][nl] for nl in self._node_labels]
					n2_labels = [g2.nodes[n2][nl] for nl in self._node_labels]
					return kn(n1_labels, n2_labels)
		else:
			# node non-synb labeled
			if len(self._node_attrs) > 0:
				def compute_vk(n1, n2):
					kn = self._node_kernels['nsymb']
					n1_attrs = np.array(
						[g1.nodes[n1][na] for na in self._node_attrs]
					).astype(float)
					n2_attrs = np.array(
						[g2.nodes[n2][na] for na in self._node_attrs]
					).astype(float)
					return kn(n1_attrs, n2_attrs)
			# node unlabeled
			else:
				for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
					if e1[2]['cost'] == e2[2]['cost']:
						kernel += 1
				return kernel

		# compute graph kernels
		if self._ds_infos['directed']:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					nk11, nk22 = compute_vk(e1[0], e2[0]), compute_vk(
						e1[1], e2[1]
					)
					kn1 = nk11 * nk22
					kernel += kn1
		else:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					# each edge walk is counted twice, starting from both its extreme nodes.
					nk11, nk12, nk21, nk22 = compute_vk(
						e1[0], e2[0]
					), compute_vk(
						e1[0], e2[1]
					), compute_vk(e1[1], e2[0]), compute_vk(e1[1], e2[1])
					kn1 = nk11 * nk22
					kn2 = nk12 * nk21
					kernel += kn1 + kn2

		return kernel


	def _wrapper_sp_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._sp_do(G_gs[i], G_gs[j])


	@staticmethod
	def _all_graphs_have_edges(graphs):
		for G in graphs:
			if nx.number_of_edges(G) == 0:
				raise ValueError('Not all graphs have edges!!!')
