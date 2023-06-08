#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:59:57 2020

@author: ljia

@references:
	[1] Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For
	Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).
"""
import sys
from itertools import product
from multiprocessing import Pool
from gklearn.utils import get_iters
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.utils.utils import get_shortest_paths, compute_vertex_kernels
from gklearn.kernels import GraphKernel


class StructuralSP(GraphKernel):

	def __init__(self, **kwargs):
		GraphKernel.__init__(
			self, **{
				k: kwargs.get(k) for k in
				['parallel', 'n_jobs', 'chunksize', 'normalize', 'copy_graphs',
				 'verbose'] if k in kwargs}
		)
		self._node_labels = kwargs.get('node_labels', [])
		self._edge_labels = kwargs.get('edge_labels', [])
		self._node_attrs = kwargs.get('node_attrs', [])
		self._edge_attrs = kwargs.get('edge_attrs', [])
		self._edge_weight = kwargs.get('edge_weight', None)
		self._node_kernels = kwargs.get('node_kernels', None)
		self._edge_kernels = kwargs.get('edge_kernels', None)
		self._compute_method = kwargs.get('compute_method', 'naive')
		self._fcsp = kwargs.get('fcsp', True)
		self._ds_infos = kwargs.get('ds_infos', {})
		self._save_sp_lists = kwargs.get('save_sp_lists', True)
		if self._compute_method == 'trie':
			self._sp_func = self._get_sps_as_trie
			self._kernel_do_func = self._ssp_do_trie
		else:
			self._sp_func = get_shortest_paths
			self._kernel_do_func = self._ssp_do_naive


	##########################################################################
	# The following is the 1st paradigm to compute kernel matrix, which is
	# compatible with `scikit-learn`.
	# -------------------------------------------------------------------
	# Special thanks to the "GraKeL" library for providing an excellent template!
	##########################################################################


	def clear_attributes(self):
		super().clear_attributes()
		if hasattr(self, '_sp_lists'):
			delattr(self, '_sp_lists')
		if hasattr(self, '_Y_sp_lists'):
			delattr(self, '_Y_sp_lists')


	def validate_parameters(self):
		super().validate_parameters()


	def _compute_kernel_matrix_series(self, Y, X=None, load_sp_lists=True):
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
		if_comp_X_sp_lists = True

		# if load saved sp_lists of X from the instance:
		if load_sp_lists:
			# sp_lists for self._graphs.
			try:
				check_is_fitted(self, ['_sp_lists'])
				sp_lists_list1 = self._sp_lists
				if_comp_X_sp_lists = False
			except NotFittedError:
				import warnings
				warnings.warn(
					'The sp_lists of self._graphs are not computed/saved. '
					'The sp_lists of `X` is computed instead.'
				)
				if_comp_X_sp_lists = True

		# Get all sp_lists of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.

		# Compute the sp_lists of X.
		if if_comp_X_sp_lists:
			if X is None:
				raise ('X can not be None.')
			# self._add_dummy_labels will modify the input in place.
			sp_lists_list1 = []
			iterator = get_iters(
				self._graphs, desc='Getting sp_lists for X',
				file=sys.stdout, verbose=(self.verbose >= 2)
			)
			for g in iterator:
				sp_lists_list1.append(self._sp_func(g))

		# sp_lists for Y.
		# 		Y = [g.copy() for g in Y] # @todo: ?
		sp_lists_list2 = []
		iterator = get_iters(
			Y, desc='Getting sp_lists for Y', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		for g in iterator:
			sp_lists_list2.append(self._sp_func(g))

		# 		if self.save_sp_lists:
		# 			self._Y_sp_lists = sp_lists_list2

		# compute kernel matrix.
		kernel_matrix = np.zeros((len(Y), len(sp_lists_list1)))

		from itertools import product
		itr = product(range(len(Y)), range(len(sp_lists_list1)))
		len_itr = int(len(Y) * len(sp_lists_list1))
		iterator = get_iters(
			itr, desc='Computing kernels', file=sys.stdout,
			length=len_itr, verbose=(self.verbose >= 2)
		)
		for i_y, i_x in iterator:
			kernel = self._kernel_do_func(
				sp_lists_list2[i_y], sp_lists_list1[i_x]
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


	def pairwise_kernel(self, x, y, are_sp_lists=False):
		"""Compute pairwise kernel between two graphs.

		Parameters
		----------
		x, y : NetworkX Graph.
			Graphs bewteen which the kernel is computed.

		are_sp_lists : boolean, optional
			If `True`, `x` and `y` are sp lists, otherwise are graphs.
			The default is False.

		Returns
		-------
		kernel: float
			The computed kernel.

		"""
		if are_sp_lists:
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
				check_is_fitted(self, ['_all_sp_lists'])
				for i, x in enumerate(self._all_graphs):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_sp_lists=True
					)  # @todo: parallel?
			except NotFittedError:
				for i, x in enumerate(self._graphs):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_sp_lists=False
					)  # @todo: parallel?

		try:
			# If transform has happened, return both diagonals.
			check_is_fitted(self, ['_Y'])
			self._Y_diag = np.empty(shape=(len(self._Y),))
			try:
				check_is_fitted(self, ['_Y_all_sp_lists'])
				for (i, y) in enumerate(self._Y_all_sp_lists):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_sp_lists=True
					)  # @todo: parallel?
			except NotFittedError:
				for (i, y) in enumerate(self._Y):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_sp_lists=False
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
		# get shortest paths of each graph in the graphs.
		splist = []
		iterator = get_iters(
			graphs, desc='getting sp graphs', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		if self._compute_method == 'trie':
			for g in iterator:
				splist.append(self._get_sps_as_trie(g))
		else:
			for g in iterator:
				splist.append(
					get_shortest_paths(
						g, self._edge_weight, self._ds_infos['directed']
					)
				)

		if self._save_sp_lists:
			self._sp_lists = splist

		# compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(graphs)), 2)
		len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
		iterator = get_iters(
			itr, desc='Computing kernels', file=sys.stdout,
			length=len_itr, verbose=(self.verbose >= 2)
		)
		for i, j in iterator:
			kernel = self._kernel_do_func(
				graphs[i], graphs[j], splist[i], splist[j]
			)
			#		if(kernel > 1):
			#			print("error here ")
			gram_matrix[i][j] = kernel
			gram_matrix[j][i] = kernel

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		# get shortest paths of each graph in the graphs.
		splist = [None] * len(self._graphs)
		pool = Pool(self.n_jobs)
		itr = zip(self._graphs, range(0, len(self._graphs)))
		if len(self._graphs) < 100 * self.n_jobs:
			chunksize = int(len(self._graphs) / self.n_jobs) + 1
		else:
			chunksize = 100
		# get shortest path graphs of self._graphs
		if self._compute_method == 'trie':
			get_sps_fun = self._wrapper_get_sps_trie
		else:
			get_sps_fun = self._wrapper_get_sps_naive
		iterator = get_iters(
			pool.imap_unordered(get_sps_fun, itr, chunksize),
			desc='getting shortest paths', file=sys.stdout,
			length=len(self._graphs), verbose=(self.verbose >= 2)
		)
		for i, sp in iterator:
			splist[i] = sp
		pool.close()
		pool.join()

		if self._save_sp_lists:
			self._sp_lists = splist

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))


		def init_worker(spl_toshare, gs_toshare):
			global G_spl, G_gs
			G_spl = spl_toshare
			G_gs = gs_toshare


		if self._compute_method == 'trie':
			do_fun = self._wrapper_ssp_do_trie
		else:
			do_fun = self._wrapper_ssp_do_naive
		parallel_gm(
			do_fun, gram_matrix, self._graphs, init_worker=init_worker,
			glbv=(splist, self._graphs), n_jobs=self.n_jobs,
			verbose=self.verbose
		)

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		# get shortest paths of g1 and each graph in g_list.
		sp1 = get_shortest_paths(
			g1, self._edge_weight, self._ds_infos['directed']
		)
		splist = []
		iterator = get_iters(
			g_list, desc='getting sp graphs', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		if self._compute_method == 'trie':
			for g in iterator:
				splist.append(self._get_sps_as_trie(g))
		else:
			for g in iterator:
				splist.append(
					get_shortest_paths(
						g, self._edge_weight, self._ds_infos['directed']
					)
				)

		# compute kernel list.
		kernel_list = [None] * len(g_list)
		iterator = get_iters(
			range(len(g_list)), desc='Computing kernels',
			file=sys.stdout, length=len(g_list), verbose=(self.verbose >= 2)
		)
		for i in iterator:
			kernel = self._kernel_do_func(g1, g_list[i], sp1, splist[i])
			kernel_list[i] = kernel

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		# get shortest paths of g1 and each graph in g_list.
		sp1 = get_shortest_paths(
			g1, self._edge_weight, self._ds_infos['directed']
		)
		splist = [None] * len(g_list)
		pool = Pool(self.n_jobs)
		itr = zip(g_list, range(0, len(g_list)))
		if len(g_list) < 100 * self.n_jobs:
			chunksize = int(len(g_list) / self.n_jobs) + 1
		else:
			chunksize = 100
		# get shortest path graphs of g_list
		if self._compute_method == 'trie':
			get_sps_fun = self._wrapper_get_sps_trie
		else:
			get_sps_fun = self._wrapper_get_sps_naive
		iterator = get_iters(
			pool.imap_unordered(get_sps_fun, itr, chunksize),
			desc='getting shortest paths', file=sys.stdout,
			length=len(g_list), verbose=(self.verbose >= 2)
		)
		for i, sp in iterator:
			splist[i] = sp
		pool.close()
		pool.join()

		# compute Gram matrix.
		kernel_list = [None] * len(g_list)


		def init_worker(sp1_toshare, spl_toshare, g1_toshare, gl_toshare):
			global G_sp1, G_spl, G_g1, G_gl
			G_sp1 = sp1_toshare
			G_spl = spl_toshare
			G_g1 = g1_toshare
			G_gl = gl_toshare


		if self._compute_method == 'trie':
			do_fun = self._wrapper_ssp_do_trie
		else:
			do_fun = self._wrapper_kernel_list_do


		def func_assign(result, var_to_assign):
			var_to_assign[result[0]] = result[1]


		itr = range(len(g_list))
		len_itr = len(g_list)
		parallel_me(
			do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(sp1, splist, g1, g_list),
			method='imap_unordered', n_jobs=self.n_jobs,
			itr_desc='Computing kernels', verbose=self.verbose
		)

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		return itr, self._ssp_do_naive(G_g1, G_gl[itr], G_sp1, G_spl[itr])


	def _compute_single_kernel_series(self, g1, g2):
		sp1 = get_shortest_paths(
			g1, self._edge_weight, self._ds_infos['directed']
		)
		sp2 = get_shortest_paths(
			g2, self._edge_weight, self._ds_infos['directed']
		)
		kernel = self._kernel_do_func(g1, g2, sp1, sp2)
		return kernel


	def _wrapper_get_sps_naive(self, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, get_shortest_paths(
			g, self._edge_weight, self._ds_infos['directed']
		)


	def _ssp_do_naive(self, g1, g2, spl1, spl2):
		if self._fcsp:  # @todo: it may be put outside the _sp_do().
			return self._sp_do_naive_fcsp(g1, g2, spl1, spl2)
		else:
			return self._sp_do_naive_naive(g1, g2, spl1, spl2)


	def _sp_do_naive_fcsp(self, g1, g2, spl1, spl2):

		kernel = 0

		# First, compute shortest path matrices, method borrowed from FCSP.
		vk_dict = self._get_all_node_kernels(g1, g2)
		# Then, compute kernels between all pairs of edges, which is an idea of
		# extension of FCSP. It suits sparse graphs, which is the most case we
		# went though. For dense graphs, this would be slow.
		ek_dict = self._get_all_edge_kernels(g1, g2)

		# compute graph kernels
		if vk_dict:
			if ek_dict:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						kpath = vk_dict[(p1[0], p2[0])]
						if kpath:
							for idx in range(1, len(p1)):
								kpath *= vk_dict[(p1[idx], p2[idx])] * \
								         ek_dict[((p1[idx - 1], p1[idx]),
								                  (p2[idx - 1], p2[idx]))]
								if not kpath:
									break
							kernel += kpath  # add up kernels of all paths
			else:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						kpath = vk_dict[(p1[0], p2[0])]
						if kpath:
							for idx in range(1, len(p1)):
								kpath *= vk_dict[(p1[idx], p2[idx])]
								if not kpath:
									break
							kernel += kpath  # add up kernels of all paths
		else:
			if ek_dict:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						if len(p1) == 0:
							kernel += 1
						else:
							kpath = 1
							for idx in range(0, len(p1) - 1):
								kpath *= ek_dict[((p1[idx], p1[idx + 1]),
								                  (p2[idx], p2[idx + 1]))]
								if not kpath:
									break
							kernel += kpath  # add up kernels of all paths
			else:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						kernel += 1
		try:
			kernel = kernel / (len(spl1) * len(spl2))  # Compute mean average
		except ZeroDivisionError:
			print(spl1, spl2)
			print(g1.nodes(data=True))
			print(g1.edges(data=True))
			raise Exception

		# # ---- exact implementation of the Fast Computation of Shortest Path Kernel (FCSP), reference [2], sadly it is slower than the current implementation
		# # compute vertex kernel matrix
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
		#				 Kmatrix += kn1 + kn2
		return kernel


	def _sp_do_naive_naive(self, g1, g2, spl1, spl2):

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
		# 			# node unlabeled
		# 			else:
		# 				for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
		# 					if e1[2]['cost'] == e2[2]['cost']:
		# 						kernel += 1
		# 				return kernel

		# Define the function to compute kernels between edges in each condition.
		if len(self._edge_labels) > 0:
			# edge symb and non-synb labeled
			if len(self._edge_attrs) > 0:
				def compute_ek(e1, e2):
					ke = self._edge_kernels['mix']
					e1_labels = [g1.edges[e1][el] for el in self._edge_labels]
					e2_labels = [g2.edges[e2][el] for el in self._edge_labels]
					# @TODO: reformat attrs during data processing a priori to save time.
					e1_attrs = np.array(
						[g1.edges[e1][ea] for ea in self._edge_attrs]
					).astype(float)
					e2_attrs = np.array(
						[g2.edges[e2][ea] for ea in self._edge_attrs]
					).astype(float)
					return ke(e1_labels, e2_labels, e1_attrs, e2_attrs)
			# edge symb labeled
			else:
				def compute_ek(e1, e2):
					ke = self._edge_kernels['symb']
					e1_labels = [g1.edges[e1][el] for el in self._edge_labels]
					e2_labels = [g2.edges[e2][el] for el in self._edge_labels]
					return ke(e1_labels, e2_labels)
		else:
			# edge non-synb labeled
			if len(self._edge_attrs) > 0:
				def compute_ek(e1, e2):
					ke = self._edge_kernels['nsymb']
					e1_attrs = np.array(
						[g1.edges[e1][ea] for ea in self._edge_attrs]
					).astype(float)
					e2_attrs = np.array(
						[g2.edges[e2][ea] for ea in self._edge_attrs]
					).astype(float)
					return ke(e1_attrs, e2_attrs)

		# compute graph kernels
		if len(self._node_labels) > 0 or len(self._node_attrs) > 0:
			if len(self._edge_labels) > 0 or len(self._edge_attrs) > 0:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						kpath = compute_vk(p1[0], p2[0])
						if kpath:
							for idx in range(1, len(p1)):
								kpath *= compute_vk(p1[idx], p2[idx]) * \
								         compute_ek(
									         (p1[idx - 1], p1[idx]),
									         (p2[idx - 1], p2[idx])
								         )
								if not kpath:
									break
							kernel += kpath  # add up kernels of all paths
			else:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						kpath = compute_vk(p1[0], p2[0])
						if kpath:
							for idx in range(1, len(p1)):
								kpath *= compute_vk(p1[idx], p2[idx])
								if not kpath:
									break
							kernel += kpath  # add up kernels of all paths
		else:
			if len(self._edge_labels) > 0 or len(self._edge_attrs) > 0:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						if len(p1) == 0:
							kernel += 1
						else:
							kpath = 1
							for idx in range(0, len(p1) - 1):
								kpath *= compute_ek(
									(p1[idx], p1[idx + 1]),
									(p2[idx], p2[idx + 1])
								)
								if not kpath:
									break
							kernel += kpath  # add up kernels of all paths
			else:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						kernel += 1
		try:
			kernel = kernel / (len(spl1) * len(spl2))  # Compute mean average
		except ZeroDivisionError:
			print(spl1, spl2)
			print(g1.nodes(data=True))
			print(g1.edges(data=True))
			raise Exception

		return kernel


	def _wrapper_ssp_do_naive(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._ssp_do_naive(G_gs[i], G_gs[j], G_spl[i], G_spl[j])


	def _get_all_node_kernels(self, g1, g2):
		return compute_vertex_kernels(
			g1, g2, self._node_kernels, node_labels=self._node_labels,
			node_attrs=self._node_attrs
		)


	def _get_all_edge_kernels(self, g1, g2):
		# compute kernels between all pairs of edges, which is an idea of
		# extension of FCSP. It suits sparse graphs, which is the most case we
		# went though. For dense graphs, this would be slow.
		ek_dict = {}  # dict of edge kernels
		if len(self._edge_labels) > 0:
			# edge symb and non-synb labeled
			if len(self._edge_attrs) > 0:
				ke = self._edge_kernels['mix']
				for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
					e1_labels = [e1[2][el] for el in self._edge_labels]
					e2_labels = [e2[2][el] for el in self._edge_labels]
					# @TODO: reformat attrs during data processing a priori to save time.
					e1_attrs = np.array(
						[e1[2][ea] for ea in self._edge_attrs]
					).astype(float)
					e2_attrs = np.array(
						[e2[2][ea] for ea in self._edge_attrs]
					).astype(float)
					ek_temp = ke(e1_labels, e2_labels, e1_attrs, e2_attrs)
					ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
					ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
					ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
					ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
			# edge symb labeled
			else:
				ke = self._edge_kernels['symb']
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						e1_labels = [e1[2][el] for el in self._edge_labels]
						e2_labels = [e2[2][el] for el in self._edge_labels]
						ek_temp = ke(e1_labels, e2_labels)
						ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
		else:
			# edge non-synb labeled
			if len(self._edge_attrs) > 0:
				ke = self._edge_kernels['nsymb']
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						e1_attrs = np.array(
							[e1[2][ea] for ea in self._edge_attrs]
						).astype(float)
						e2_attrs = np.array(
							[e2[2][ea] for ea in self._edge_attrs]
						).astype(float)
						ek_temp = ke(e1_attrs, e2_attrs)
						ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
			# edge unlabeled
			else:
				pass

		return ek_dict
