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
# from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.utils.utils import getSPGraph
from gklearn.kernels import GraphKernel


class ShortestPath(GraphKernel):
	
	def __init__(self, **kwargs):
		GraphKernel.__init__(self)
		self.__node_labels = kwargs.get('node_labels', [])
		self.__node_attrs = kwargs.get('node_attrs', [])
		self.__edge_weight = kwargs.get('edge_weight', None)
		self.__node_kernels = kwargs.get('node_kernels', None)
		self.__ds_infos = kwargs.get('ds_infos', {})


	def _compute_gm_series(self):
		# get shortest path graph of each graph.
		if self._verbose >= 2:
			iterator = tqdm(self._graphs, desc='getting sp graphs', file=sys.stdout)
		else:
			iterator = self._graphs
		self._graphs = [getSPGraph(g, edge_weight=self.__edge_weight) for g in iterator]
		
		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))
		
		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(self._graphs)), 2)
		if self._verbose >= 2:
			iterator = tqdm(itr, desc='calculating kernels', file=sys.stdout)
		else:
			iterator = itr
		for i, j in iterator:
			kernel = self.__sp_do(self._graphs[i], self._graphs[j])
			gram_matrix[i][j] = kernel
			gram_matrix[j][i] = kernel
				
		return gram_matrix
			
			
	def _compute_gm_imap_unordered(self):
		# get shortest path graph of each graph.
		pool = Pool(self._n_jobs)
		get_sp_graphs_fun = self._wrapper_get_sp_graphs
		itr = zip(self._graphs, range(0, len(self._graphs)))
		if len(self._graphs) < 100 * self._n_jobs:
			chunksize = int(len(self._graphs) / self._n_jobs) + 1
		else:
			chunksize = 100
		if self._verbose >= 2:
			iterator = tqdm(pool.imap_unordered(get_sp_graphs_fun, itr, chunksize),
							desc='getting sp graphs', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(get_sp_graphs_fun, itr, chunksize)
		for i, g in iterator:
			self._graphs[i] = g
		pool.close()
		pool.join()
		
		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))
		
		def init_worker(gs_toshare):
			global G_gs
			G_gs = gs_toshare
		do_fun = self._wrapper_sp_do
		parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=init_worker, 
					glbv=(self._graphs,), n_jobs=self._n_jobs, verbose=self._verbose)
			
		return gram_matrix
	
	
	def _compute_kernel_list_series(self, g1, g_list):
		# get shortest path graphs of g1 and each graph in g_list.
		g1 = getSPGraph(g1, edge_weight=self.__edge_weight)
		if self._verbose >= 2:
			iterator = tqdm(g_list, desc='getting sp graphs', file=sys.stdout)
		else:
			iterator = g_list
		g_list = [getSPGraph(g, edge_weight=self.__edge_weight) for g in iterator]
		
		# compute kernel list.
		kernel_list = [None] * len(g_list)
		if self._verbose >= 2:
			iterator = tqdm(range(len(g_list)), desc='calculating kernels', file=sys.stdout)
		else:
			iterator = range(len(g_list))
		for i in iterator:
			kernel = self.__sp_do(g1, g_list[i])
			kernel_list[i] = kernel
				
		return kernel_list
	
	
	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		# get shortest path graphs of g1 and each graph in g_list.
		g1 = getSPGraph(g1, edge_weight=self.__edge_weight)
		pool = Pool(self._n_jobs)
		get_sp_graphs_fun = self._wrapper_get_sp_graphs
		itr = zip(g_list, range(0, len(g_list)))
		if len(g_list) < 100 * self._n_jobs:
			chunksize = int(len(g_list) / self._n_jobs) + 1
		else:
			chunksize = 100
		if self._verbose >= 2:
			iterator = tqdm(pool.imap_unordered(get_sp_graphs_fun, itr, chunksize),
							desc='getting sp graphs', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(get_sp_graphs_fun, itr, chunksize)
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
		parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(g1, g_list), method='imap_unordered', n_jobs=self._n_jobs, itr_desc='calculating kernels', verbose=self._verbose)
			
		return kernel_list
	
	
	def _wrapper_kernel_list_do(self, itr):
		return itr, self.__sp_do(G_g1, G_gl[itr])
	
	
	def _compute_single_kernel_series(self, g1, g2):
		g1 = getSPGraph(g1, edge_weight=self.__edge_weight)
		g2 = getSPGraph(g2, edge_weight=self.__edge_weight)
		kernel = self.__sp_do(g1, g2)
		return kernel			
		
	
	def _wrapper_get_sp_graphs(self, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, getSPGraph(g, edge_weight=self.__edge_weight)
	
	
	def __sp_do(self, g1, g2):
		
		kernel = 0
	
		# compute shortest path matrices first, method borrowed from FCSP.
		vk_dict = {}  # shortest path matrices dict
		if len(self.__node_labels) > 0:
			# node symb and non-synb labeled
			if len(self.__node_attrs) > 0:
				kn = self.__node_kernels['mix']
				for n1, n2 in product(
						g1.nodes(data=True), g2.nodes(data=True)):
					n1_labels = [n1[1][nl] for nl in self.__node_labels]
					n2_labels = [n2[1][nl] for nl in self.__node_labels]
					n1_attrs = [n1[1][na] for na in self.__node_attrs]
					n2_attrs = [n2[1][na] for na in self.__node_attrs]
					vk_dict[(n1[0], n2[0])] = kn(n1_labels, n2_labels, n1_attrs, n2_attrs)
			# node symb labeled
			else:
				kn = self.__node_kernels['symb']
				for n1 in g1.nodes(data=True):
					for n2 in g2.nodes(data=True):
						n1_labels = [n1[1][nl] for nl in self.__node_labels]
						n2_labels = [n2[1][nl] for nl in self.__node_labels]
						vk_dict[(n1[0], n2[0])] = kn(n1_labels, n2_labels)
		else:
			# node non-synb labeled
			if len(self.__node_attrs) > 0:
				kn = self.__node_kernels['nsymb']
				for n1 in g1.nodes(data=True):
					for n2 in g2.nodes(data=True):
						n1_attrs = [n1[1][na] for na in self.__node_attrs]
						n2_attrs = [n2[1][na] for na in self.__node_attrs]
						vk_dict[(n1[0], n2[0])] = kn(n1_attrs, n2_attrs)
			# node unlabeled
			else:
				for e1, e2 in product(
						g1.edges(data=True), g2.edges(data=True)):
					if e1[2]['cost'] == e2[2]['cost']:
						kernel += 1
				return kernel
	
		# compute graph kernels
		if self.__ds_infos['directed']:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					nk11, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(e1[1], e2[1])]
					kn1 = nk11 * nk22
					kernel += kn1
		else:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					# each edge walk is counted twice, starting from both its extreme nodes.
					nk11, nk12, nk21, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(
						e1[0], e2[1])], vk_dict[(e1[1], e2[0])], vk_dict[(e1[1], e2[1])]
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
	
	
	def _wrapper_sp_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self.__sp_do(G_gs[i], G_gs[j])