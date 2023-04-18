#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:12:45 2020

@author: ljia

@references:

	[1] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11(Apr):1201–1242, 2010.
"""

import sys
from gklearn.utils import get_iters
import numpy as np
import networkx as nx
from scipy.sparse import kron
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.kernels import RandomWalkMeta


class SpectralDecomposition(RandomWalkMeta):


	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._sub_kernel = kwargs.get('sub_kernel', None)


	def _compute_gm_series(self, graphs):
		self._check_edge_weight(graphs, self.verbose)
		self._check_graphs(graphs)
		if self.verbose >= 2:
			import warnings
			warnings.warn('All labels are ignored. Only works for undirected graphs.')

		# compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		if self._q is None:
			# precompute the spectral decomposition of each graph.
			P_list = []
			D_list = []
			iterator = get_iters(graphs, desc='spectral decompose', file=sys.stdout, verbose=(self.verbose >= 2))
			for G in iterator:
				# don't normalize adjacency matrices if q is a uniform vector. Note
				# A actually is the transpose of the adjacency matrix.
				A = nx.adjacency_matrix(G, self._edge_weight).todense().transpose()
				ew, ev = np.linalg.eig(A)
				D_list.append(ew)
				P_list.append(ev)
#		P_inv_list = [p.T for p in P_list] # @todo: also works for directed graphs?

			if self._p is None: # p is uniform distribution as default.
				q_T_list = [np.full((1, nx.number_of_nodes(G)), 1 / nx.number_of_nodes(G)) for G in graphs]
#			q_T_list = [q.T for q in q_list]

				from itertools import combinations_with_replacement
				itr = combinations_with_replacement(range(0, len(graphs)), 2)
				len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
				iterator = get_iters(itr, desc='Computing kernels', file=sys.stdout, length=len_itr, verbose=(self.verbose >= 2))

				for i, j in iterator:
					kernel = self._kernel_do(q_T_list[i], q_T_list[j], P_list[i], P_list[j], D_list[i], D_list[j], self._weight, self._sub_kernel)
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
			warnings.warn('All labels are ignored. Only works for undirected graphs.')

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		if self._q is None:
			# precompute the spectral decomposition of each graph.
			P_list = []
			D_list = []
			iterator = get_iters(self._graphs, desc='spectral decompose', file=sys.stdout, verbose=(self.verbose >= 2))
			for G in iterator:
				# don't normalize adjacency matrices if q is a uniform vector. Note
				# A actually is the transpose of the adjacency matrix.
				A = nx.adjacency_matrix(G, self._edge_weight).todense().transpose()
				ew, ev = np.linalg.eig(A)
				D_list.append(ew)
				P_list.append(ev) # @todo: parallel?

			if self._p is None: # p is uniform distribution as default.
				q_T_list = [np.full((1, nx.number_of_nodes(G)), 1 / nx.number_of_nodes(G)) for G in self._graphs] # @todo: parallel?

				def init_worker(q_T_list_toshare, P_list_toshare, D_list_toshare):
					global G_q_T_list, G_P_list, G_D_list
					G_q_T_list = q_T_list_toshare
					G_P_list = P_list_toshare
					G_D_list = D_list_toshare

				do_fun = self._wrapper_kernel_do
				parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=init_worker,
							glbv=(q_T_list, P_list, D_list), n_jobs=self.n_jobs, verbose=self.verbose)

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
			warnings.warn('All labels are ignored. Only works for undirected graphs.')

		# compute kernel list.
		kernel_list = [None] * len(g_list)

		if self._q is None:
			# precompute the spectral decomposition of each graph.
			A1 = nx.adjacency_matrix(g1, self._edge_weight).todense().transpose()
			D1, P1 = np.linalg.eig(A1)
			P_list = []
			D_list = []
			iterator = get_iters(g_list, desc='spectral decompose', file=sys.stdout, verbose=(self.verbose >= 2))
			for G in iterator:
				# don't normalize adjacency matrices if q is a uniform vector. Note
				# A actually is the transpose of the adjacency matrix.
				A = nx.adjacency_matrix(G, self._edge_weight).todense().transpose()
				ew, ev = np.linalg.eig(A)
				D_list.append(ew)
				P_list.append(ev)

			if self._p is None: # p is uniform distribution as default.
				q_T1 = 1 / nx.number_of_nodes(g1)
				q_T_list = [np.full((1, nx.number_of_nodes(G)), 1 / nx.number_of_nodes(G)) for G in g_list]
				iterator = get_iters(range(len(g_list)), desc='Computing kernels', file=sys.stdout, length=len(g_list), verbose=(self.verbose >= 2))

				for i in iterator:
					kernel = self._kernel_do(q_T1, q_T_list[i], P1, P_list[i], D1, D_list[i], self._weight, self._sub_kernel)
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
			warnings.warn('All labels are ignored. Only works for undirected graphs.')

		# compute kernel list.
		kernel_list = [None] * len(g_list)

		if self._q is None:
			# precompute the spectral decomposition of each graph.
			A1 = nx.adjacency_matrix(g1, self._edge_weight).todense().transpose()
			D1, P1 = np.linalg.eig(A1)
			P_list = []
			D_list = []
			if self.verbose >= 2:
				iterator = get_iters(g_list, desc='spectral decompose', file=sys.stdout)
			else:
				iterator = g_list
			for G in iterator:
				# don't normalize adjacency matrices if q is a uniform vector. Note
				# A actually is the transpose of the adjacency matrix.
				A = nx.adjacency_matrix(G, self._edge_weight).todense().transpose()
				ew, ev = np.linalg.eig(A)
				D_list.append(ew)
				P_list.append(ev) # @todo: parallel?

			if self._p is None: # p is uniform distribution as default.
				q_T1 = 1 / nx.number_of_nodes(g1)
				q_T_list = [np.full((1, nx.number_of_nodes(G)), 1 / nx.number_of_nodes(G)) for G in g_list] # @todo: parallel?

				def init_worker(q_T1_toshare, P1_toshare, D1_toshare, q_T_list_toshare, P_list_toshare, D_list_toshare):
					global G_q_T1, G_P1, G_D1, G_q_T_list, G_P_list, G_D_list
					G_q_T1 = q_T1_toshare
					G_P1 = P1_toshare
					G_D1 = D1_toshare
					G_q_T_list = q_T_list_toshare
					G_P_list = P_list_toshare
					G_D_list = D_list_toshare

				do_fun = self._wrapper_kernel_list_do

				def func_assign(result, var_to_assign):
					var_to_assign[result[0]] = result[1]
				itr = range(len(g_list))
				len_itr = len(g_list)
				parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
					init_worker=init_worker, glbv=(q_T1, P1, D1, q_T_list, P_list, D_list), method='imap_unordered', n_jobs=self.n_jobs, itr_desc='Computing kernels', verbose=self.verbose)

			else: # @todo
				pass
		else: # @todo
			pass

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		return itr, self._kernel_do(G_q_T1, G_q_T_list[itr], G_P1, G_P_list[itr], G_D1, G_D_list[itr], self._weight, self._sub_kernel)


	def _compute_single_kernel_series(self, g1, g2):
		self._check_edge_weight([g1] + [g2], self.verbose)
		self._check_graphs([g1] + [g2])
		if self.verbose >= 2:
			import warnings
			warnings.warn('All labels are ignored. Only works for undirected graphs.')

		if self._q is None:
			# precompute the spectral decomposition of each graph.
			A1 = nx.adjacency_matrix(g1, self._edge_weight).todense().transpose()
			D1, P1 = np.linalg.eig(A1)
			A2 = nx.adjacency_matrix(g2, self._edge_weight).todense().transpose()
			D2, P2 = np.linalg.eig(A2)

			if self._p is None: # p is uniform distribution as default.
				q_T1 = 1 / nx.number_of_nodes(g1)
				q_T2 = 1 / nx.number_of_nodes(g2)
				kernel = self._kernel_do(q_T1, q_T2, P1, P2, D1, D2, self._weight, self._sub_kernel)
			else: # @todo
				pass
		else: # @todo
			pass

		return kernel


	def _kernel_do(self, q_T1, q_T2, P1, P2, D1, D2, weight, sub_kernel):
		# use uniform distribution if there is no prior knowledge.
		kl = kron(np.dot(q_T1, P1), np.dot(q_T2, P2)).todense()
		# @todo: this is not needed when p = q (kr = kl.T) for undirected graphs.
	#	kr = kron(np.dot(P_inv_list[i], q_list[i]), np.dot(P_inv_list[j], q_list[j])).todense()
		if sub_kernel == 'exp':
			D_diag = np.array([d1 * d2 for d1 in D1 for d2 in D2])
			kmiddle = np.diag(np.exp(weight * D_diag))
		elif sub_kernel == 'geo':
			D_diag = np.array([d1 * d2 for d1 in D1 for d2 in D2])
			kmiddle = np.diag(weight * D_diag)
			kmiddle = np.identity(len(kmiddle)) - weight * kmiddle
			kmiddle = np.linalg.inv(kmiddle)
		return np.dot(np.dot(kl, kmiddle), kl.T)[0, 0]


	def _wrapper_kernel_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do(G_q_T_list[i], G_q_T_list[j], G_P_list[i], G_P_list[j], G_D_list[i], G_D_list[j], self._weight, self._sub_kernel)