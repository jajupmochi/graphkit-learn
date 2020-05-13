#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:52:47 2020

@author: ljia
"""
import numpy as np
import networkx as nx
import multiprocessing
import time

class GraphKernel(object):
	
	def __init__(self):
		self._graphs = None
		self._parallel = ''
		self._n_jobs = 0
		self._verbose = None
		self._normalize = True
		self._run_time = 0
		self._gram_matrix = None
		self._gram_matrix_unnorm = None
	

	def compute(self, *graphs, **kwargs):
		self._parallel = kwargs.get('parallel', 'imap_unordered')
		self._n_jobs = kwargs.get('n_jobs', multiprocessing.cpu_count())
		self._normalize = kwargs.get('normalize', True)
		self._verbose = kwargs.get('verbose', 2)
		
		if len(graphs) == 1:
			if not isinstance(graphs[0], list):
				raise Exception('Cannot detect graphs.')
			elif len(graphs[0]) == 0:
				raise Exception('The graph list given is empty. No computation was performed.')
			else:
				self._graphs = [g.copy() for g in graphs[0]]
				self._gram_matrix = self.__compute_gram_matrix()
				self._gram_matrix_unnorm = np.copy(self._gram_matrix)
				if self._normalize:
					self._gram_matrix = self.normalize_gm(self._gram_matrix)
				return self._gram_matrix, self._run_time
			
		elif len(graphs) == 2:
			if self.is_graph(graphs[0]) and self.is_graph(graphs[1]):
				kernel = self.__compute_single_kernel(graphs[0].copy(), graphs[1].copy())
				return kernel, self._run_time
			elif self.is_graph(graphs[0]) and isinstance(graphs[1], list):
				g1 = graphs[0].copy()
				g_list = [g.copy() for g in graphs[1]]
				kernel_list = self.__compute_kernel_list(g1, g_list)
				return kernel_list, self._run_time
			elif isinstance(graphs[0], list) and self.is_graph(graphs[1]):
				g1 = graphs[1].copy()
				g_list = [g.copy() for g in graphs[0]]
				kernel_list = self.__compute_kernel_list(g1, g_list)
				return kernel_list, self._run_time
			else:
				raise Exception('Cannot detect graphs.')
				
		elif len(graphs) == 0 and self._graphs is None:
			raise Exception('Please add graphs before computing.')
			
		else:
			raise Exception('Cannot detect graphs.')
			
			
	def normalize_gm(self, gram_matrix):
		import warnings
		warnings.warn('gklearn.kernels.graph_kernel.normalize_gm will be deprecated, use gklearn.utils.normalize_gram_matrix instead', DeprecationWarning)

		diag = gram_matrix.diagonal().copy()
		for i in range(len(gram_matrix)):
			for j in range(i, len(gram_matrix)):
				gram_matrix[i][j] /= np.sqrt(diag[i] * diag[j])
				gram_matrix[j][i] = gram_matrix[i][j]
		return gram_matrix
	
	
	def compute_distance_matrix(self):
		if self._gram_matrix is None:
			raise Exception('Please compute the Gram matrix before computing distance matrix.')
		dis_mat = np.empty((len(self._gram_matrix), len(self._gram_matrix)))
		for i in range(len(self._gram_matrix)):
			for j in range(i, len(self._gram_matrix)):
				dis = self._gram_matrix[i, i] + self._gram_matrix[j, j] - 2 * self._gram_matrix[i, j]
				if dis < 0:
					if dis > -1e-10:
						dis = 0
					else:
						raise ValueError('The distance is negative.')
				dis_mat[i, j] = np.sqrt(dis)
				dis_mat[j, i] = dis_mat[i, j]
		dis_max = np.max(np.max(dis_mat))
		dis_min = np.min(np.min(dis_mat[dis_mat != 0]))
		dis_mean = np.mean(np.mean(dis_mat))
		return dis_mat, dis_max, dis_min, dis_mean
			
			
	def __compute_gram_matrix(self):
		start_time = time.time()
		
		if self._parallel == 'imap_unordered':
			gram_matrix = self._compute_gm_imap_unordered()
		elif self._parallel == None:
			gram_matrix = self._compute_gm_series()
		else:
			raise Exception('Parallel mode is not set correctly.')
		
		self._run_time = time.time() - start_time
		if self._verbose:
			print('Gram matrix of size %d built in %s seconds.'
			  % (len(self._graphs), self._run_time))
			
		return gram_matrix
			
			
	def _compute_gm_series(self):
		pass


	def _compute_gm_imap_unordered(self):
		pass
	
	
	def __compute_kernel_list(self, g1, g_list):
		start_time = time.time()
		
		if self._parallel == 'imap_unordered':
			kernel_list = self._compute_kernel_list_imap_unordered(g1, g_list)
		elif self._parallel == None:
			kernel_list = self._compute_kernel_list_series(g1, g_list)
		else:
			raise Exception('Parallel mode is not set correctly.')
		
		self._run_time = time.time() - start_time
		if self._verbose:
			print('Graph kernel bewteen a graph and a list of %d graphs built in %s seconds.'
			  % (len(g_list), self._run_time))
			
		return kernel_list
	

	def _compute_kernel_list_series(self, g1, g_list):
		pass

	
	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		pass
	
	
	def __compute_single_kernel(self, g1, g2):
		start_time = time.time()
		
		kernel = self._compute_single_kernel_series(g1, g2)
		
		self._run_time = time.time() - start_time
		if self._verbose:
			print('Graph kernel bewteen two graphs built in %s seconds.' % (self._run_time))
			
		return kernel
	
	
	def _compute_single_kernel_series(self, g1, g2):
		pass
	
	
	def is_graph(self, graph):
		if isinstance(graph, nx.Graph):
			return True
		if isinstance(graph, nx.DiGraph):
			return True 
		if isinstance(graph, nx.MultiGraph):
			return True 
		if isinstance(graph, nx.MultiDiGraph):
			return True 
		return False
	
	
	@property
	def graphs(self):
		return self._graphs
	
	
	@property
	def parallel(self):
		return self._parallel
	
	
	@property
	def n_jobs(self):
		return self._n_jobs


	@property
	def verbose(self):
		return self._verbose
	
	
	@property
	def normalize(self):
		return self._normalize
	
	
	@property
	def run_time(self):
		return self._run_time
	
	 
	@property
	def gram_matrix(self):
		return self._gram_matrix
	
	@gram_matrix.setter
	def gram_matrix(self, value):
		self._gram_matrix = value
	
	 
	@property
	def gram_matrix_unnorm(self):
		return self._gram_matrix_unnorm 

	@gram_matrix_unnorm.setter
	def gram_matrix_unnorm(self, value):
		self._gram_matrix_unnorm = value