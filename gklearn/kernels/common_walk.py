#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:21:31 2020

@author: ljia

@references:

	[1] Thomas Gärtner, Peter Flach, and Stefan Wrobel. On graph kernels:
	Hardness results and efficient alternatives. Learning Theory and Kernel
	Machines, pages 129–143, 2003.
"""

import sys
from gklearn.utils import get_iters
import numpy as np
import networkx as nx
from gklearn.utils import SpecialLabel
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.utils.utils import direct_product_graph
from gklearn.kernels import GraphKernel


class CommonWalk(GraphKernel):

	def __init__(self, **kwargs):
		GraphKernel.__init__(self)
		self._node_labels = kwargs.get('node_labels', [])
		self._edge_labels = kwargs.get('edge_labels', [])
		self._weight = kwargs.get('weight', 1)
		self._compute_method = kwargs.get('compute_method', None)
		self._ds_infos = kwargs.get('ds_infos', {})
		self._compute_method = self._compute_method.lower()


	def _compute_gm_series(self, graphs):
		self._check_graphs(graphs)
		self._add_dummy_labels(graphs)
		if not self._ds_infos['directed']:  #  convert
			graphs = [G.to_directed() for G in graphs]

		# compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(graphs)), 2)
		len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
		iterator = get_iters(itr, desc='Computing kernels', file=sys.stdout,
					length=len_itr, verbose=(self.verbose >= 2))

		# direct product graph method - exponential
		if self._compute_method == 'exp':
			for i, j in iterator:
				kernel = self._kernel_do_exp(graphs[i], graphs[j], self._weight)
				gram_matrix[i][j] = kernel
				gram_matrix[j][i] = kernel
		# direct product graph method - geometric
		elif self._compute_method == 'geo':
			for i, j in iterator:
				kernel = self._kernel_do_geo(graphs[i], graphs[j], self._weight)
				gram_matrix[i][j] = kernel
				gram_matrix[j][i] = kernel

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		self._check_graphs(self._graphs)
		self._add_dummy_labels(self._graphs)
		if not self._ds_infos['directed']:  #  convert
			self._graphs = [G.to_directed() for G in self._graphs]

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

# 		def init_worker(gn_toshare):
# 			global G_gn
# 			G_gn = gn_toshare

		# direct product graph method - exponential
		if self._compute_method == 'exp':
			do_fun = self._wrapper_kernel_do_exp
		# direct product graph method - geometric
		elif self._compute_method == 'geo':
			do_fun = self._wrapper_kernel_do_geo

		parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=_init_worker_gm,
			  glbv=(self._graphs,), n_jobs=self.n_jobs, verbose=self.verbose)

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		self._check_graphs(g_list + [g1])
		self._add_dummy_labels(g_list + [g1])
		if not self._ds_infos['directed']:  #  convert
			g1 = g1.to_directed()
			g_list = [G.to_directed() for G in g_list]

		# compute kernel list.
		kernel_list = [None] * len(g_list)
		if self.verbose >= 2:
			iterator = get_iters(range(len(g_list)), desc='Computing kernels',
						 file=sys.stdout, length=len(g_list), verbose=(self.verbose >= 2))
		else:
			iterator = range(len(g_list))

		# direct product graph method - exponential
		if self._compute_method == 'exp':
			for i in iterator:
				kernel = self._kernel_do_exp(g1, g_list[i], self._weight)
				kernel_list[i] = kernel
		# direct product graph method - geometric
		elif self._compute_method == 'geo':
			for i in iterator:
				kernel = self._kernel_do_geo(g1, g_list[i], self._weight)
				kernel_list[i] = kernel

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._check_graphs(g_list + [g1])
		self._add_dummy_labels(g_list + [g1])
		if not self._ds_infos['directed']:  #  convert
			g1 = g1.to_directed()
			g_list = [G.to_directed() for G in g_list]

		# compute kernel list.
		kernel_list = [None] * len(g_list)

# 		def init_worker(g1_toshare, g_list_toshare):
# 			global G_g1, G_g_list
# 			G_g1 = g1_toshare
# 			G_g_list = g_list_toshare

		# direct product graph method - exponential
		if self._compute_method == 'exp':
			do_fun = self._wrapper_kernel_list_do_exp
		# direct product graph method - geometric
		elif self._compute_method == 'geo':
			do_fun = self._wrapper_kernel_list_do_geo

		def func_assign(result, var_to_assign):
			var_to_assign[result[0]] = result[1]
		itr = range(len(g_list))
		len_itr = len(g_list)
		parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=_init_worker_list, glbv=(g1, g_list), method='imap_unordered',
			n_jobs=self.n_jobs, itr_desc='Computing kernels', verbose=self.verbose)

		return kernel_list


	def _wrapper_kernel_list_do_exp(self, itr):
		return itr, self._kernel_do_exp(G_g1, G_g_list[itr], self._weight)


	def _wrapper_kernel_list_do_geo(self, itr):
		return itr, self._kernel_do_geo(G_g1, G_g_list[itr], self._weight)


	def _compute_single_kernel_series(self, g1, g2):
		self._check_graphs([g1] + [g2])
		self._add_dummy_labels([g1] + [g2])
		if not self._ds_infos['directed']:  #  convert
			g1 = g1.to_directed()
			g2 = g2.to_directed()

		# direct product graph method - exponential
		if self._compute_method == 'exp':
			kernel = self._kernel_do_exp(g1, g2, self._weight)
		# direct product graph method - geometric
		elif self._compute_method == 'geo':
			kernel = self._kernel_do_geo(g1, g2, self._weight)

		return kernel


	def _kernel_do_exp(self, g1, g2, beta):
		"""Compute common walk graph kernel between 2 graphs using exponential
		series.

		Parameters
		----------
		g1, g2 : NetworkX graphs
			Graphs between which the kernels are computed.
		beta : integer
			Weight.

		Return
		------
		kernel : float
			The common walk Kernel between 2 graphs.
		"""
		# get tensor product / direct product
		gp = direct_product_graph(g1, g2, self._node_labels, self._edge_labels)
		# return 0 if the direct product graph have no more than 1 node.
		if nx.number_of_nodes(gp) < 2:
			return 0
		A = nx.adjacency_matrix(gp).todense()

		ew, ev = np.linalg.eig(A)
# 		# remove imaginary part if possible.
# 		# @todo: don't know if it is necessary.
# 		for i in range(len(ew)):
# 			if np.abs(ew[i].imag) < 1e-9:
# 				ew[i] = ew[i].real
# 		for i in range(ev.shape[0]):
# 			for j in range(ev.shape[1]):
# 				if np.abs(ev[i, j].imag) < 1e-9:
# 					ev[i, j] = ev[i, j].real

		D = np.zeros((len(ew), len(ew)), dtype=complex) # @todo: use complex?
		for i in range(len(ew)):
			D[i][i] = np.exp(beta * ew[i])

		exp_D = ev * D * ev.T
		kernel = exp_D.sum()
		if (kernel.real == 0 and np.abs(kernel.imag) < 1e-9) or np.abs(kernel.imag / kernel.real) < 1e-9:
			kernel = kernel.real

		return kernel


	def _wrapper_kernel_do_exp(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do_exp(G_gn[i], G_gn[j], self._weight)


	def _kernel_do_geo(self, g1, g2, gamma):
		"""Compute common walk graph kernel between 2 graphs using geometric
		series.

		Parameters
		----------
		g1, g2 : NetworkX graphs
			Graphs between which the kernels are computed.
		gamma : integer
			Weight.

		Return
		------
		kernel : float
			The common walk Kernel between 2 graphs.
		"""
		# get tensor product / direct product
		gp = direct_product_graph(g1, g2, self._node_labels, self._edge_labels)
		# return 0 if the direct product graph have no more than 1 node.
		if nx.number_of_nodes(gp) < 2:
			return 0
		A = nx.adjacency_matrix(gp).todense()
		mat = np.identity(len(A)) - gamma * A
	#	try:
		return np.linalg.inv(mat).sum()
	#	except np.linalg.LinAlgError:
	#		return np.nan


	def _wrapper_kernel_do_geo(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do_geo(G_gn[i], G_gn[j], self._weight)


	def _check_graphs(self, Gn):
		for g in Gn:
			if nx.number_of_nodes(g) == 1:
				raise Exception('Graphs must contain more than 1 nodes to construct adjacency matrices.')


	def _add_dummy_labels(self, Gn):
		if len(self._node_labels) == 0 or (len(self._node_labels) == 1 and self._node_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self._node_labels = [SpecialLabel.DUMMY]
		if len(self._edge_labels) == 0 or (len(self._edge_labels) == 1 and self._edge_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_edge_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self._edge_labels = [SpecialLabel.DUMMY]


def _init_worker_gm(gn_toshare):
	global G_gn
	G_gn = gn_toshare


def _init_worker_list(g1_toshare, g_list_toshare):
	global G_g1, G_g_list
	G_g1 = g1_toshare
	G_g_list = g_list_toshare