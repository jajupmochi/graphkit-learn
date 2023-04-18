#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:22:57 2020

@author: ljia

@references:

	[1] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between
	labeled graphs. In Proceedings of the 20th International Conference on
	Machine Learning, Washington, DC, United States, 2003.

	[2] Pierre Mahé, Nobuhisa Ueda, Tatsuya Akutsu, Jean-Luc Perret, and
	Jean-Philippe Vert. Extensions of marginalized graph kernels. In
	Proceedings of the twenty-first international conference on Machine
	learning, page 70. ACM, 2004.
"""

import sys
from multiprocessing import Pool
from gklearn.utils import get_iters
import numpy as np
import networkx as nx
from gklearn.utils import SpecialLabel
from gklearn.utils.kernels import deltakernel
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.utils.utils import untotterTransformation
from gklearn.kernels import GraphKernel


class Marginalized(GraphKernel):

	def __init__(self, **kwargs):
		GraphKernel.__init__(self)
		self._node_labels = kwargs.get('node_labels', [])
		self._edge_labels = kwargs.get('edge_labels', [])
		self._p_quit = kwargs.get('p_quit', 0.5)
		self._n_iteration = kwargs.get('n_iteration', 10)
		self._remove_totters = kwargs.get('remove_totters', False)
		self._ds_infos = kwargs.get('ds_infos', {})
		self._n_iteration = int(self._n_iteration)


	def _compute_gm_series(self, graphs):
		self._add_dummy_labels(graphs)

		if self._remove_totters:
			iterator = get_iters(graphs, desc='removing tottering', file=sys.stdout, verbose=(self.verbose >= 2))
			# @todo: this may not work.
			graphs = [untotterTransformation(G, self._node_labels, self._edge_labels) for G in iterator]

		# compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(graphs)), 2)
		len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
		iterator = get_iters(itr, desc='Computing kernels', file=sys.stdout,
					length=len_itr, verbose=(self.verbose >= 2))
		for i, j in iterator:
			kernel = self._kernel_do(graphs[i], graphs[j])
			gram_matrix[i][j] = kernel
			gram_matrix[j][i] = kernel # @todo: no directed graph considered?

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		self._add_dummy_labels(self._graphs)

		if self._remove_totters:
			pool = Pool(self.n_jobs)
			itr = range(0, len(self._graphs))
			if len(self._graphs) < 100 * self.n_jobs:
				chunksize = int(len(self._graphs) / self.n_jobs) + 1
			else:
				chunksize = 100
			remove_fun = self._wrapper_untotter
			iterator = get_iters(pool.imap_unordered(remove_fun, itr, chunksize),
							desc='removing tottering', file=sys.stdout,
							length=len(self._graphs), verbose=(self.verbose >= 2))
			for i, g in iterator:
				self._graphs[i] = g
			pool.close()
			pool.join()

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		def init_worker(gn_toshare):
			global G_gn
			G_gn = gn_toshare
		do_fun = self._wrapper_kernel_do
		parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=init_worker,
					glbv=(self._graphs,), n_jobs=self.n_jobs, verbose=self.verbose)

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		self._add_dummy_labels(g_list + [g1])

		if self._remove_totters:
			g1 = untotterTransformation(g1, self._node_labels, self._edge_labels) # @todo: this may not work.
			iterator = get_iters(g_list, desc='removing tottering', file=sys.stdout, verbose=(self.verbose >= 2))
			# @todo: this may not work.
			g_list = [untotterTransformation(G, self._node_labels, self._edge_labels) for G in iterator]

		# compute kernel list.
		kernel_list = [None] * len(g_list)
		iterator = get_iters(range(len(g_list)), desc='Computing kernels', file=sys.stdout, length=len(g_list), verbose=(self.verbose >= 2))
		for i in iterator:
			kernel = self._kernel_do(g1, g_list[i])
			kernel_list[i] = kernel

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._add_dummy_labels(g_list + [g1])

		if self._remove_totters:
			g1 = untotterTransformation(g1, self._node_labels, self._edge_labels) # @todo: this may not work.
			pool = Pool(self.n_jobs)
			itr = range(0, len(g_list))
			if len(g_list) < 100 * self.n_jobs:
				chunksize = int(len(g_list) / self.n_jobs) + 1
			else:
				chunksize = 100
			remove_fun = self._wrapper_untotter
			iterator = get_iters(pool.imap_unordered(remove_fun, itr, chunksize),
							desc='removing tottering', file=sys.stdout,
							length=len(g_list), verbose=(self.verbose >= 2))
			for i, g in iterator:
				g_list[i] = g
			pool.close()
			pool.join()

		# compute kernel list.
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
		parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(g1, g_list), method='imap_unordered',
			n_jobs=self.n_jobs, itr_desc='Computing kernels', verbose=self.verbose)

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		return itr, self._kernel_do(G_g1, G_g_list[itr])


	def _compute_single_kernel_series(self, g1, g2):
		self._add_dummy_labels([g1] + [g2])
		if self._remove_totters:
			g1 = untotterTransformation(g1, self._node_labels, self._edge_labels) # @todo: this may not work.
			g2 = untotterTransformation(g2, self._node_labels, self._edge_labels)
		kernel = self._kernel_do(g1, g2)
		return kernel


	def _kernel_do(self, g1, g2):
		"""Compute marginalized graph kernel between 2 graphs.

		Parameters
		----------
		g1, g2 : NetworkX graphs
			2 graphs between which the kernel is computed.

		Return
		------
		kernel : float
			Marginalized kernel between 2 graphs.
		"""
		# init parameters
		kernel = 0
		num_nodes_G1 = nx.number_of_nodes(g1)
		num_nodes_G2 = nx.number_of_nodes(g2)
		# the initial probability distribution in the random walks generating step
		# (uniform distribution over |G|)
		p_init_G1 = 1 / num_nodes_G1
		p_init_G2 = 1 / num_nodes_G2

		q = self._p_quit * self._p_quit
		r1 = q

	#	# initial R_inf
	#	# matrix to save all the R_inf for all pairs of nodes
	#	R_inf = np.zeros([num_nodes_G1, num_nodes_G2])
	#
	#	# Compute R_inf with a simple interative method
	#	for i in range(1, n_iteration):
	#		R_inf_new = np.zeros([num_nodes_G1, num_nodes_G2])
	#		R_inf_new.fill(r1)
	#
	#		# Compute R_inf for each pair of nodes
	#		for node1 in g1.nodes(data=True):
	#			neighbor_n1 = g1[node1[0]]
	#			# the transition probability distribution in the random walks
	#			# generating step (uniform distribution over the vertices adjacent
	#			# to the current vertex)
	#			if len(neighbor_n1) > 0:
	#				p_trans_n1 = (1 - p_quit) / len(neighbor_n1)
	#				for node2 in g2.nodes(data=True):
	#					neighbor_n2 = g2[node2[0]]
	#					if len(neighbor_n2) > 0:
	#						p_trans_n2 = (1 - p_quit) / len(neighbor_n2)
	#
	#						for neighbor1 in neighbor_n1:
	#							for neighbor2 in neighbor_n2:
	#								t = p_trans_n1 * p_trans_n2 * \
	#									deltakernel(g1.node[neighbor1][node_label],
	#												g2.node[neighbor2][node_label]) * \
	#									deltakernel(
	#										neighbor_n1[neighbor1][edge_label],
	#										neighbor_n2[neighbor2][edge_label])
	#
	#								R_inf_new[node1[0]][node2[0]] += t * R_inf[neighbor1][
	#									neighbor2]  # ref [1] equation (8)
	#		R_inf[:] = R_inf_new
	#
	#	# add elements of R_inf up and compute kernel
	#	for node1 in g1.nodes(data=True):
	#		for node2 in g2.nodes(data=True):
	#			s = p_init_G1 * p_init_G2 * deltakernel(
	#				node1[1][node_label], node2[1][node_label])
	#			kernel += s * R_inf[node1[0]][node2[0]]  # ref [1] equation (6)


		R_inf = {} # dict to save all the R_inf for all pairs of nodes
		# initial R_inf, the 1st iteration.
		for node1 in g1.nodes():
			for node2 in g2.nodes():
	#			R_inf[(node1[0], node2[0])] = r1
				if len(g1[node1]) > 0:
					if len(g2[node2]) > 0:
						R_inf[(node1, node2)] = r1
					else:
						R_inf[(node1, node2)] = self._p_quit
				else:
					if len(g2[node2]) > 0:
						R_inf[(node1, node2)] = self._p_quit
					else:
						R_inf[(node1, node2)] = 1

		# compute all transition probability first.
		t_dict = {}
		if self._n_iteration > 1:
			for node1 in g1.nodes():
				neighbor_n1 = g1[node1]
				# the transition probability distribution in the random walks
				# generating step (uniform distribution over the vertices adjacent
				# to the current vertex)
				if len(neighbor_n1) > 0:
					p_trans_n1 = (1 - self._p_quit) / len(neighbor_n1)
					for node2 in g2.nodes():
						neighbor_n2 = g2[node2]
						if len(neighbor_n2) > 0:
							p_trans_n2 = (1 - self._p_quit) / len(neighbor_n2)
							for neighbor1 in neighbor_n1:
								for neighbor2 in neighbor_n2:
									t_dict[(node1, node2, neighbor1, neighbor2)] = \
										p_trans_n1 * p_trans_n2 * \
										deltakernel(tuple(g1.nodes[neighbor1][nl] for nl in self._node_labels), tuple(g2.nodes[neighbor2][nl] for nl in self._node_labels)) * \
										deltakernel(tuple(neighbor_n1[neighbor1][el] for el in self._edge_labels), tuple(neighbor_n2[neighbor2][el] for el in self._edge_labels))

		# Compute R_inf with a simple interative method
		for i in range(2, self._n_iteration + 1):
			R_inf_old = R_inf.copy()

			# Compute R_inf for each pair of nodes
			for node1 in g1.nodes():
				neighbor_n1 = g1[node1]
				# the transition probability distribution in the random walks
				# generating step (uniform distribution over the vertices adjacent
				# to the current vertex)
				if len(neighbor_n1) > 0:
					for node2 in g2.nodes():
						neighbor_n2 = g2[node2]
						if len(neighbor_n2) > 0:
							R_inf[(node1, node2)] = r1
							for neighbor1 in neighbor_n1:
								for neighbor2 in neighbor_n2:
									R_inf[(node1, node2)] += \
										(t_dict[(node1, node2, neighbor1, neighbor2)] * \
										R_inf_old[(neighbor1, neighbor2)])  # ref [1] equation (8)

		# add elements of R_inf up and compute kernel.
		for (n1, n2), value in R_inf.items():
			s = p_init_G1 * p_init_G2 * deltakernel(tuple(g1.nodes[n1][nl] for nl in self._node_labels), tuple(g2.nodes[n2][nl] for nl in self._node_labels))
			kernel += s * value  # ref [1] equation (6)

		return kernel


	def _wrapper_kernel_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do(G_gn[i], G_gn[j])


	def _wrapper_untotter(self, i):
		return i, untotterTransformation(self._graphs[i], self._node_labels, self._edge_labels) # @todo: this may not work.


	def _add_dummy_labels(self, Gn):
		if len(self._node_labels) == 0 or (len(self._node_labels) == 1 and self._node_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self._node_labels = [SpecialLabel.DUMMY]
		if len(self._edge_labels) == 0 or (len(self._edge_labels) == 1 and self._edge_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_edge_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self._edge_labels = [SpecialLabel.DUMMY]