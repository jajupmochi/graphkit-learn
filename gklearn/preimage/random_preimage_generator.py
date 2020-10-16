#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:29:52 2020

@author: ljia
"""

import numpy as np
import time
import sys
from tqdm import tqdm
import multiprocessing
import networkx as nx
from multiprocessing import Pool
from functools import partial
from gklearn.preimage import PreimageGenerator
from gklearn.preimage.utils import compute_k_dis
from gklearn.utils import Timer
from gklearn.utils.utils import get_graph_kernel_by_name
# from gklearn.utils.dataset import Dataset


class RandomPreimageGenerator(PreimageGenerator):
	
	def __init__(self, dataset=None):
		PreimageGenerator.__init__(self, dataset=dataset)
		# arguments to set.
		self._k = 5 # number of nearest neighbors of phi in D_N.
		self._r_max = 10 # maximum number of iterations.
		self._l = 500 # numbers of graphs generated for each graph in D_k U {g_i_hat}.
		self._alphas = None # weights of linear combinations of points in kernel space.
		self._parallel = True
		self._n_jobs = multiprocessing.cpu_count()
		self._time_limit_in_sec = 0
		self._max_itrs = 20
		# values to compute.
		self._runtime_generate_preimage = None
		self._runtime_total = None
		self._preimage = None
		self._best_from_dataset = None
		self._k_dis_preimage = None
		self._k_dis_dataset = None
		self._itrs = 0
		self._converged = False # @todo
		self._num_updates = 0
		# values that can be set or to be computed.
		self._gram_matrix_unnorm = None
		self._runtime_precompute_gm = None

		
	def set_options(self, **kwargs):
		self._kernel_options = kwargs.get('kernel_options', {})
		self._graph_kernel = kwargs.get('graph_kernel', None)
		self._verbose = kwargs.get('verbose', 2)
		self._k = kwargs.get('k', 5)
		self._r_max = kwargs.get('r_max', 10)
		self._l = kwargs.get('l', 500)
		self._alphas = kwargs.get('alphas', None)
		self._parallel = kwargs.get('parallel', True)
		self._n_jobs = kwargs.get('n_jobs', multiprocessing.cpu_count())
		self._time_limit_in_sec = kwargs.get('time_limit_in_sec', 0)
		self._max_itrs = kwargs.get('max_itrs', 20)
		self._gram_matrix_unnorm = kwargs.get('gram_matrix_unnorm', None)
		self._runtime_precompute_gm = kwargs.get('runtime_precompute_gm', None)
		
		
	def run(self):
		self._graph_kernel = get_graph_kernel_by_name(self._kernel_options['name'], 
						  node_labels=self._dataset.node_labels,
						  edge_labels=self._dataset.edge_labels, 
						  node_attrs=self._dataset.node_attrs,
						  edge_attrs=self._dataset.edge_attrs,
						  ds_infos=self._dataset.get_dataset_infos(keys=['directed']),
						  kernel_options=self._kernel_options)
		
		# record start time.
		start = time.time()
		
		# 1. precompute gram matrix.
		if self._gram_matrix_unnorm is None:
			gram_matrix, run_time = self._graph_kernel.compute(self._dataset.graphs, **self._kernel_options)
			self._gram_matrix_unnorm = self._graph_kernel.gram_matrix_unnorm
			end_precompute_gm = time.time()
			self._runtime_precompute_gm = end_precompute_gm - start
		else:
			if self._runtime_precompute_gm is None:
				raise Exception('Parameter "runtime_precompute_gm" must be given when using pre-computed Gram matrix.')
			self._graph_kernel.gram_matrix_unnorm = self._gram_matrix_unnorm
			if self._kernel_options['normalize']:
				self._graph_kernel.gram_matrix = self._graph_kernel.normalize_gm(np.copy(self._gram_matrix_unnorm))
			else:
				self._graph_kernel.gram_matrix = np.copy(self._gram_matrix_unnorm)
			end_precompute_gm = time.time()
			start -= self._runtime_precompute_gm
			
		# 2. compute k nearest neighbors of phi in D_N.
		if self._verbose >= 2:
			print('\nstart computing k nearest neighbors of phi in D_N...\n')
		D_N = self._dataset.graphs
		if self._alphas is None:
			self._alphas = [1 / len(D_N)] * len(D_N)
		k_dis_list = [] # distance between g_star and each graph.
		term3 = 0
		for i1, a1 in enumerate(self._alphas):
			for i2, a2 in enumerate(self._alphas):
				term3 += a1 * a2 * self._graph_kernel.gram_matrix[i1, i2]
		for idx in range(len(D_N)):
			k_dis_list.append(compute_k_dis(idx, range(0, len(D_N)), self._alphas, self._graph_kernel.gram_matrix, term3=term3, withterm3=True))
			
		# sort.
		sort_idx = np.argsort(k_dis_list)
		dis_gs = [k_dis_list[idis] for idis in sort_idx[0:self._k]] # the k shortest distances.
		nb_best = len(np.argwhere(dis_gs == dis_gs[0]).flatten().tolist())
		g0hat_list = [D_N[idx].copy() for idx in sort_idx[0:nb_best]] # the nearest neighbors of phi in D_N
		self._best_from_dataset = g0hat_list[0] # get the first best graph if there are muitlple.
		self._k_dis_dataset = dis_gs[0]
		
		if self._k_dis_dataset == 0: # get the exact pre-image.
			end_generate_preimage = time.time()
			self._runtime_generate_preimage = end_generate_preimage - end_precompute_gm
			self._runtime_total = end_generate_preimage - start
			self._preimage = self._best_from_dataset.copy()	
			self._k_dis_preimage = self._k_dis_dataset
			if self._verbose:
				print()
				print('=============================================================================')
				print('The exact pre-image is found from the input dataset.')
				print('-----------------------------------------------------------------------------')
				print('Distance in kernel space for the best graph from dataset and for preimage:', self._k_dis_dataset)
				print('Time to pre-compute Gram matrix:', self._runtime_precompute_gm)
				print('Time to generate pre-images:', self._runtime_generate_preimage)
				print('Total time:', self._runtime_total)
				print('=============================================================================')
				print()
			return
		
		dhat = dis_gs[0] # the nearest distance
		Gk = [D_N[ig].copy() for ig in sort_idx[0:self._k]] # the k nearest neighbors
		Gs_nearest = [nx.convert_node_labels_to_integers(g) for g in Gk] # [g.copy() for g in Gk]
		
		# 3. start iterations.
		if self._verbose >= 2:
			print('starting iterations...')
		gihat_list = []
		dihat_list = []
		r = 0
		dis_of_each_itr = [dhat]
		if self._parallel:
			self._kernel_options['parallel'] = None
		self._itrs = 0
		self._num_updates = 0
		timer = Timer(self._time_limit_in_sec)
		while not self._termination_criterion_met(timer, self._itrs, r):
			print('\n- r =', r)
			found = False
			dis_bests = dis_gs + dihat_list
			
			# compute numbers of edges to be inserted/deleted.
			# @todo what if the log is negetive? how to choose alpha (scalar)?
			fdgs_list = np.array(dis_bests)
			if np.min(fdgs_list) < 1: # in case the log is negetive.
				fdgs_list /= np.min(fdgs_list)
			fdgs_list = [int(item) for item in np.ceil(np.log(fdgs_list))]
			if np.min(fdgs_list) < 1: # in case the log is smaller than 1.
				fdgs_list = np.array(fdgs_list) + 1
			# expand the number of modifications to increase the possiblity.
			nb_vpairs_list = [nx.number_of_nodes(g) * (nx.number_of_nodes(g) - 1) for g in (Gs_nearest + gihat_list)]
			nb_vpairs_min = np.min(nb_vpairs_list)
			idx_fdgs_max = np.argmax(fdgs_list)
			fdgs_max_old = fdgs_list[idx_fdgs_max]
			fdgs_max = fdgs_max_old
			nb_modif = 1
			for idx, nb in enumerate(range(nb_vpairs_min, nb_vpairs_min - fdgs_max, -1)):
				nb_modif *= nb / (fdgs_max - idx)
			while fdgs_max < nb_vpairs_min and nb_modif < self._l:
				fdgs_max += 1
				nb_modif *= (nb_vpairs_min - fdgs_max + 1) / fdgs_max
			nb_increase = int(fdgs_max - fdgs_max_old)
			if nb_increase > 0:
				fdgs_list += 1
				
				
			for ig, gs in enumerate(Gs_nearest + gihat_list):
				if self._verbose >= 2:
					print('-- computing', ig + 1, 'graphs out of', len(Gs_nearest) + len(gihat_list))
				gnew, dhat, found = self._generate_l_graphs(gs, fdgs_list[ig], dhat, ig, found, term3)
						  
			if found:
				r = 0
				gihat_list = [gnew]
				dihat_list = [dhat]
			else:
				r += 1
				
			dis_of_each_itr.append(dhat)
			self._itrs += 1
			if self._verbose >= 2:
				print('Total number of iterations is', self._itrs, '.')
				print('The preimage is updated', self._num_updates, 'times.')
				print('The shortest distances for previous iterations are', dis_of_each_itr, '.')
			
			
		# get results and print.
		end_generate_preimage = time.time()
		self._runtime_generate_preimage = end_generate_preimage - end_precompute_gm
		self._runtime_total = end_generate_preimage - start
		self._preimage = (g0hat_list[0] if len(gihat_list) == 0 else gihat_list[0])
		self._k_dis_preimage = dhat
		if self._verbose:
			print()
			print('=============================================================================')
			print('Finished generation of preimages.')
			print('-----------------------------------------------------------------------------')
			print('Distance in kernel space for the best graph from dataset:', self._k_dis_dataset)
			print('Distance in kernel space for the preimage:', self._k_dis_preimage)
			print('Total number of iterations for optimizing:', self._itrs)
			print('Total number of updating preimage:', self._num_updates)
			print('Time to pre-compute Gram matrix:', self._runtime_precompute_gm)
			print('Time to generate pre-images:', self._runtime_generate_preimage)
			print('Total time:', self._runtime_total)
			print('=============================================================================')
			print()	
			
			
	def _generate_l_graphs(self, g_init, fdgs, dhat, ig, found, term3):
		if self._parallel:
			gnew, dhat, found = self._generate_l_graphs_parallel(g_init, fdgs, dhat, ig, found, term3)
		else:
			gnew, dhat, found = self._generate_l_graphs_series(g_init, fdgs, dhat, ig, found, term3)
		return gnew, dhat, found
			
			
	def _generate_l_graphs_series(self, g_init, fdgs, dhat, ig, found, term3):
		gnew = None
		updated = False
		for trial in range(0, self._l):
			if self._verbose >= 2:
				print('---', trial + 1, 'trial out of', self._l)

			gtemp, dnew = self._do_trial(g_init, fdgs, term3, trial)

			# get the better graph preimage.
			if dnew <= dhat: # @todo: the new distance is smaller or also equal?
				if dhat - dnew > 1e-6:
					if self._verbose >= 2:
						print('trial =', str(trial))
						print('\nI am smaller!')
						print('index (as in D_k U {gihat} =', str(ig))
						print('distance:', dhat, '->', dnew)
					updated = True
				else:
					if self._verbose >= 2:
						print('I am equal!') 
				dhat = dnew
				gnew = gtemp.copy()
				found = True # found better or equally good graph.
		
		if updated:
			self._num_updates += 1
				
		return gnew, dhat, found
	
	
	def _generate_l_graphs_parallel(self, g_init, fdgs, dhat, ig, found, term3):
		gnew = None
		len_itr = self._l
		gnew_list = [None] * len_itr
		dnew_list = [None] * len_itr
		itr = range(0, len_itr)
		n_jobs = multiprocessing.cpu_count()
		if len_itr < 100 * n_jobs:
			chunksize = int(len_itr / n_jobs) + 1
		else:
			chunksize = 100
		do_fun = partial(self._generate_graph_parallel, g_init, fdgs, term3)
		pool = Pool(processes=n_jobs)
		if self._verbose >= 2:
			iterator = tqdm(pool.imap_unordered(do_fun, itr, chunksize),
						desc='Generating l graphs', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(do_fun, itr, chunksize)
		for idx, gnew, dnew in iterator:
			gnew_list[idx] = gnew
			dnew_list[idx] = dnew
		pool.close()
		pool.join()
		
		# check if get the better graph preimage.
		idx_min = np.argmin(dnew_list)
		dnew = dnew_list[idx_min]
		if dnew <= dhat: # @todo: the new distance is smaller or also equal?
			if dhat - dnew > 1e-6: # @todo: use a proportion and watch out for 0.
				if self._verbose >= 2:
					print('I am smaller!')
					print('index (as in D_k U {gihat}) =', str(ig))
					print('distance:', dhat, '->', dnew, '\n')
				self._num_updates += 1
			else:
				if self._verbose >= 2:
					print('I am equal!') 
			dhat = dnew
			gnew = gnew_list[idx_min]
			found = True # found better graph.
				
		return gnew, dhat, found
		
		
	def _generate_graph_parallel(self, g_init, fdgs, term3, itr):
		trial = itr			
		gtemp, dnew = self._do_trial(g_init, fdgs, term3, trial)		
		return trial, gtemp, dnew
	
	
	def _do_trial(self, g_init, fdgs, term3, trial):
		# add and delete edges.
		gtemp = g_init.copy()
		seed = (trial + int(time.time())) % (2 ** 32 - 1)
		rdm_state = np.random.RandomState(seed=seed)
		# which edges to change.
		# @todo: should we use just half of the adjacency matrix for undirected graphs?
		nb_vpairs = nx.number_of_nodes(g_init) * (nx.number_of_nodes(g_init) - 1)
		# @todo: what if fdgs is bigger than nb_vpairs?
		idx_change = rdm_state.randint(0, high=nb_vpairs, size=(fdgs if 
									   fdgs < nb_vpairs else nb_vpairs))
# 		print(idx_change)
		for item in idx_change:
			node1 = int(item / (nx.number_of_nodes(g_init) - 1))
			node2 = (item - node1 * (nx.number_of_nodes(g_init) - 1))
			if node2 >= node1: # skip the self pair.
				node2 += 1
			# @todo: is the randomness correct?
			if not gtemp.has_edge(node1, node2):
				gtemp.add_edge(node1, node2)
			else:
				gtemp.remove_edge(node1, node2)
				
		# compute new distances.
		kernels_to_gtmp, _ = self._graph_kernel.compute(gtemp, self._dataset.graphs, **self._kernel_options)
		kernel_gtmp, _ = self._graph_kernel.compute(gtemp, gtemp, **self._kernel_options)
		if self._kernel_options['normalize']:
			kernels_to_gtmp = [kernels_to_gtmp[i] / np.sqrt(self._gram_matrix_unnorm[i, i] * kernel_gtmp) for i in range(len(kernels_to_gtmp))] # normalize 
			kernel_gtmp = 1
		# @todo: not correct kernel value
		gram_with_gtmp = np.concatenate((np.array([kernels_to_gtmp]), np.copy(self._graph_kernel.gram_matrix)), axis=0)
		gram_with_gtmp = np.concatenate((np.array([[kernel_gtmp] + kernels_to_gtmp]).T, gram_with_gtmp), axis=1)
		dnew = compute_k_dis(0, range(1, 1 + len(self._dataset.graphs)), self._alphas, gram_with_gtmp, term3=term3, withterm3=True)
		
		return gtemp, dnew
			

	def get_results(self):
		results = {}
		results['runtime_precompute_gm'] = self._runtime_precompute_gm
		results['runtime_generate_preimage'] = self._runtime_generate_preimage
		results['runtime_total'] = self._runtime_total
		results['k_dis_dataset'] = self._k_dis_dataset
		results['k_dis_preimage'] = self._k_dis_preimage
		results['itrs'] = self._itrs
		results['num_updates'] = self._num_updates
		return results


	def _termination_criterion_met(self, timer, itr, r):
		if timer.expired() or (itr >= self._max_itrs if self._max_itrs >= 0 else False):
# 			if self._state == AlgorithmState.TERMINATED:
# 				self._state = AlgorithmState.INITIALIZED
			return True
		return (r >= self._r_max if self._r_max >= 0 else False)
# 		return converged or (itrs_without_update > self._max_itrs_without_update if self._max_itrs_without_update >= 0 else False)
		
	
	@property
	def preimage(self):
		return self._preimage
	
	
	@property
	def best_from_dataset(self):
		return self._best_from_dataset
	
		
	@property
	def gram_matrix_unnorm(self):
		return self._gram_matrix_unnorm
	
	@gram_matrix_unnorm.setter
	def gram_matrix_unnorm(self, value):
		self._gram_matrix_unnorm = value