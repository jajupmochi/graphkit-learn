#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:24:46 2020

@author: ljia

@references:

	[1] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11(Apr):1201–1242, 2010.
"""

import sys
from gklearn.utils import get_iters
import numpy as np
import networkx as nx
from control import dlyap
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.kernels import RandomWalkMeta


class SylvesterEquation(RandomWalkMeta):


	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	def _compute_gm_series(self, graphs):
		self._check_edge_weight(graphs, self.verbose)
		self._check_graphs(graphs)
		if self.verbose >= 2:
			import warnings
			warnings.warn('All labels are ignored.')

		lmda = self._weight

		# compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		if self._q is None:
			# don't normalize adjacency matrices if q is a uniform vector. Note
			# A_wave_list actually contains the transposes of the adjacency matrices.
			iterator = get_iters(graphs, desc='compute adjacency matrices', file=sys.stdout, verbose=(self.verbose >= 2))
			A_wave_list = [nx.adjacency_matrix(G, self._edge_weight).todense().transpose() for G in iterator]
	#		# normalized adjacency matrices
	#		A_wave_list = []
	#		for G in tqdm(Gn, desc='compute adjacency matrices', file=sys.stdout):
	#			A_tilde = nx.adjacency_matrix(G, eweight).todense().transpose()
	#			norm = A_tilde.sum(axis=0)
	#			norm[norm == 0] = 1
	#			A_wave_list.append(A_tilde / norm)

			if self._p is None: # p is uniform distribution as default.
				from itertools import combinations_with_replacement
				itr = combinations_with_replacement(range(0, len(graphs)), 2)
				len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
				iterator = get_iters(itr, desc='Computing kernels', file=sys.stdout, length=len_itr, verbose=(self.verbose >= 2))

				for i, j in iterator:
					kernel = self._kernel_do(A_wave_list[i], A_wave_list[j], lmda)
					gram_matrix[i][j] = kernel
					gram_matrix[j][i] = kernel

			else: # @todo
				pass
		else: # @todo
			pass

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		self._check_edge_weight(self._graphs, self.verbose)
		self._check_graphs(self._graphs)
		if self.verbose >= 2:
			import warnings
			warnings.warn('All labels are ignored.')

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		if self._q is None:
			# don't normalize adjacency matrices if q is a uniform vector. Note
			# A_wave_list actually contains the transposes of the adjacency matrices.
			iterator = get_iters(self._graphs, desc='compute adjacency matrices', file=sys.stdout, verbose=(self.verbose >= 2))
			A_wave_list = [nx.adjacency_matrix(G, self._edge_weight).todense().transpose() for G in iterator] # @todo: parallel?

			if self._p is None: # p is uniform distribution as default.
				def init_worker(A_wave_list_toshare):
					global G_A_wave_list
					G_A_wave_list = A_wave_list_toshare

				do_fun = self._wrapper_kernel_do

				parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=init_worker,
							glbv=(A_wave_list,), n_jobs=self.n_jobs, verbose=self.verbose)

			else: # @todo
				pass
		else: # @todo
			pass

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		self._check_edge_weight(g_list + [g1], self.verbose)
		self._check_graphs(g_list + [g1])
		if self.verbose >= 2:
			import warnings
			warnings.warn('All labels are ignored.')

		lmda = self._weight

		# compute kernel list.
		kernel_list = [None] * len(g_list)

		if self._q is None:
			# don't normalize adjacency matrices if q is a uniform vector. Note
			# A_wave_list actually contains the transposes of the adjacency matrices.
			A_wave_1 = nx.adjacency_matrix(g1, self._edge_weight).todense().transpose()
			iterator = get_iters(g_list, desc='compute adjacency matrices', file=sys.stdout, verbose=(self.verbose >= 2))
			A_wave_list = [nx.adjacency_matrix(G, self._edge_weight).todense().transpose() for G in iterator]

			if self._p is None: # p is uniform distribution as default.
				iterator = get_iters(range(len(g_list)), desc='Computing kernels', file=sys.stdout, length=len(g_list), verbose=(self.verbose >= 2))

				for i in iterator:
					kernel = self._kernel_do(A_wave_1, A_wave_list[i], lmda)
					kernel_list[i] = kernel

			else: # @todo
				pass
		else: # @todo
			pass

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._check_edge_weight(g_list + [g1], self.verbose)
		self._check_graphs(g_list + [g1])
		if self.verbose >= 2:
			import warnings
			warnings.warn('All labels are ignored.')

		# compute kernel list.
		kernel_list = [None] * len(g_list)

		if self._q is None:
			# don't normalize adjacency matrices if q is a uniform vector. Note
			# A_wave_list actually contains the transposes of the adjacency matrices.
			A_wave_1 = nx.adjacency_matrix(g1, self._edge_weight).todense().transpose()
			iterator = get_iters(g_list, desc='compute adjacency matrices', file=sys.stdout, verbose=(self.verbose >= 2))
			A_wave_list = [nx.adjacency_matrix(G, self._edge_weight).todense().transpose() for G in iterator] # @todo: parallel?

			if self._p is None: # p is uniform distribution as default.
				def init_worker(A_wave_1_toshare, A_wave_list_toshare):
					global G_A_wave_1, G_A_wave_list
					G_A_wave_1 = A_wave_1_toshare
					G_A_wave_list = A_wave_list_toshare

				do_fun = self._wrapper_kernel_list_do

				def func_assign(result, var_to_assign):
					var_to_assign[result[0]] = result[1]
				itr = range(len(g_list))
				len_itr = len(g_list)
				parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
					init_worker=init_worker, glbv=(A_wave_1, A_wave_list), method='imap_unordered',
					n_jobs=self.n_jobs, itr_desc='Computing kernels', verbose=self.verbose)

			else: # @todo
				pass
		else: # @todo
			pass

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		return itr, self._kernel_do(G_A_wave_1, G_A_wave_list[itr], self._weight)


	def _compute_single_kernel_series(self, g1, g2):
		self._check_edge_weight([g1] + [g2], self.verbose)
		self._check_graphs([g1] + [g2])
		if self.verbose >= 2:
			import warnings
			warnings.warn('All labels are ignored.')

		lmda = self._weight

		if self._q is None:
			# don't normalize adjacency matrices if q is a uniform vector. Note
			# A_wave_list actually contains the transposes of the adjacency matrices.
			A_wave_1 = nx.adjacency_matrix(g1, self._edge_weight).todense().transpose()
			A_wave_2 = nx.adjacency_matrix(g2, self._edge_weight).todense().transpose()
			if self._p is None: # p is uniform distribution as default.
				kernel = self._kernel_do(A_wave_1, A_wave_2, lmda)
			else: # @todo
				pass
		else: # @todo
			pass

		return kernel


	def _kernel_do(self, A_wave1, A_wave2, lmda):

		S = lmda * A_wave2
		T_t = A_wave1
		# use uniform distribution if there is no prior knowledge.
		nb_pd = len(A_wave1) * len(A_wave2)
		p_times_uni = 1 / nb_pd
		M0 = np.full((len(A_wave2), len(A_wave1)), p_times_uni)
		X = dlyap(S, T_t, M0)
		X = np.reshape(X, (-1, 1), order='F')
		# use uniform distribution if there is no prior knowledge.
		q_times = np.full((1, nb_pd), p_times_uni)
		return np.dot(q_times, X)


	def _wrapper_kernel_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do(G_A_wave_list[i], G_A_wave_list[j], self._weight)