#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:04:46 2020

@author: ljia
"""
import numpy as np
import time
import random
import multiprocessing
import networkx as nx
from gklearn.preimage import PreimageGenerator
from gklearn.preimage.utils import compute_k_dis
from gklearn.ged.env import GEDEnv
from gklearn.ged.learning import CostMatricesLearner
from gklearn.ged.median import MedianGraphEstimatorCML
from gklearn.ged.median import constant_node_costs, mge_options_to_string
from gklearn.utils.utils import get_graph_kernel_by_name
from gklearn.ged.util import label_costs_to_matrix


class MedianPreimageGeneratorCML(PreimageGenerator):
	"""Generator median preimages by cost matrices learning using the pure Python version of GEDEnv. Works only for symbolic labeled graphs.
	"""
	
	def __init__(self, dataset=None):
		PreimageGenerator.__init__(self, dataset=dataset)
		### arguments to set.
		self.__mge = None
		self.__ged_options = {}
		self.__mge_options = {}
# 		self.__fit_method = 'k-graphs'
		self.__init_method = 'random'
		self.__init_ecc = None
		self.__parallel = True
		self.__n_jobs = multiprocessing.cpu_count()
		self.__ds_name = None
		# for cml.
		self.__time_limit_in_sec = 0
		self.__max_itrs = 100
		self.__max_itrs_without_update = 3
		self.__epsilon_residual = 0.01
		self.__epsilon_ec = 0.1
		self.__allow_zeros = True
# 		self.__triangle_rule = True
		### values to compute.
		self.__runtime_optimize_ec = None
		self.__runtime_generate_preimage = None
		self.__runtime_total = None
		self.__set_median = None
		self.__gen_median = None
		self.__best_from_dataset = None
		self.__sod_set_median = None
		self.__sod_gen_median = None
		self.__k_dis_set_median = None
		self.__k_dis_gen_median = None
		self.__k_dis_dataset = None
		self.__node_label_costs = None
		self.__edge_label_costs = None
		# for cml.
		self.__itrs = 0
		self.__converged = False
		self.__num_updates_ecs = 0
		### values that can be set or to be computed.
		self.__edit_cost_constants = []
		self.__gram_matrix_unnorm = None
		self.__runtime_precompute_gm = None

		
	def set_options(self, **kwargs):
		self._kernel_options = kwargs.get('kernel_options', {})
		self._graph_kernel = kwargs.get('graph_kernel', None)
		self._verbose = kwargs.get('verbose', 2)
		self.__ged_options = kwargs.get('ged_options', {})
		self.__mge_options = kwargs.get('mge_options', {})
# 		self.__fit_method = kwargs.get('fit_method', 'k-graphs')
		self.__init_method = kwargs.get('init_method', 'random')
		self.__init_ecc = kwargs.get('init_ecc', None)
		self.__edit_cost_constants = kwargs.get('edit_cost_constants', [])
		self.__parallel = kwargs.get('parallel', True)
		self.__n_jobs = kwargs.get('n_jobs', multiprocessing.cpu_count())
		self.__ds_name = kwargs.get('ds_name', None)
		self.__time_limit_in_sec = kwargs.get('time_limit_in_sec', 0)
		self.__max_itrs = kwargs.get('max_itrs', 100)
		self.__max_itrs_without_update = kwargs.get('max_itrs_without_update', 3)
		self.__epsilon_residual = kwargs.get('epsilon_residual', 0.01)
		self.__epsilon_ec = kwargs.get('epsilon_ec', 0.1)
		self.__gram_matrix_unnorm = kwargs.get('gram_matrix_unnorm', None)
		self.__runtime_precompute_gm = kwargs.get('runtime_precompute_gm', None)
		self.__allow_zeros = kwargs.get('allow_zeros', True)
# 		self.__triangle_rule = kwargs.get('triangle_rule', True)
		
		
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
		if self.__gram_matrix_unnorm is None:
			gram_matrix, run_time = self._graph_kernel.compute(self._dataset.graphs, **self._kernel_options)
			self.__gram_matrix_unnorm = self._graph_kernel.gram_matrix_unnorm
			end_precompute_gm = time.time()
			self.__runtime_precompute_gm = end_precompute_gm - start
		else:
			if self.__runtime_precompute_gm is None:
				raise Exception('Parameter "runtime_precompute_gm" must be given when using pre-computed Gram matrix.')
			self._graph_kernel.gram_matrix_unnorm = self.__gram_matrix_unnorm
			if self._kernel_options['normalize']:
				self._graph_kernel.gram_matrix = self._graph_kernel.normalize_gm(np.copy(self.__gram_matrix_unnorm))
			else:
				self._graph_kernel.gram_matrix = np.copy(self.__gram_matrix_unnorm)
			end_precompute_gm = time.time()
			start -= self.__runtime_precompute_gm
			
# 		if self.__fit_method != 'k-graphs' and self.__fit_method != 'whole-dataset':
# 			start = time.time()
# 			self.__runtime_precompute_gm = 0
# 			end_precompute_gm = start
		
		# 2. optimize edit cost constants. 
		self.__optimize_edit_cost_vector()
		end_optimize_ec = time.time()
		self.__runtime_optimize_ec = end_optimize_ec - end_precompute_gm
		
		# 3. compute set median and gen median using optimized edit costs.
		if self._verbose >= 2:
			print('\nstart computing set median and gen median using optimized edit costs...\n')
		self.__gmg_bcu()
		end_generate_preimage = time.time()
		self.__runtime_generate_preimage = end_generate_preimage - end_optimize_ec
		self.__runtime_total = end_generate_preimage - start
		if self._verbose >= 2:
			print('medians computed.')
			print('SOD of the set median: ', self.__sod_set_median)
			print('SOD of the generalized median: ', self.__sod_gen_median)
			
		# 4. compute kernel distances to the true median.
		if self._verbose >= 2:
			print('\nstart computing distances to true median....\n')
		self.__compute_distances_to_true_median()

		# 5. print out results.
		if self._verbose:
			print()
			print('================================================================================')
			print('Finished generation of preimages.')
			print('--------------------------------------------------------------------------------')
			print('The optimized edit costs:', self.__edit_cost_constants)
			print('SOD of the set median:', self.__sod_set_median)
			print('SOD of the generalized median:', self.__sod_gen_median)
			print('Distance in kernel space for set median:', self.__k_dis_set_median)
			print('Distance in kernel space for generalized median:', self.__k_dis_gen_median)
			print('Minimum distance in kernel space for each graph in median set:', self.__k_dis_dataset)
			print('Time to pre-compute Gram matrix:', self.__runtime_precompute_gm)
			print('Time to optimize edit costs:', self.__runtime_optimize_ec)
			print('Time to generate pre-images:', self.__runtime_generate_preimage)
			print('Total time:', self.__runtime_total)
			print('Total number of iterations for optimizing:', self.__itrs)
			print('Total number of updating edit costs:', self.__num_updates_ecs)
			print('Is optimization of edit costs converged:', self.__converged)
			print('================================================================================')
			print()


	def get_results(self):
		results = {}
		results['edit_cost_constants'] = self.__edit_cost_constants
		results['runtime_precompute_gm'] = self.__runtime_precompute_gm
		results['runtime_optimize_ec'] = self.__runtime_optimize_ec
		results['runtime_generate_preimage'] = self.__runtime_generate_preimage
		results['runtime_total'] = self.__runtime_total
		results['sod_set_median'] = self.__sod_set_median
		results['sod_gen_median'] = self.__sod_gen_median
		results['k_dis_set_median'] = self.__k_dis_set_median
		results['k_dis_gen_median'] = self.__k_dis_gen_median
		results['k_dis_dataset'] = self.__k_dis_dataset
		results['itrs'] = self.__itrs
		results['converged'] = self.__converged
		results['num_updates_ecc'] = self.__num_updates_ecs
		results['mge'] = {}
		results['mge']['num_decrease_order'] = self.__mge.get_num_times_order_decreased()
		results['mge']['num_increase_order'] = self.__mge.get_num_times_order_increased()
		results['mge']['num_converged_descents'] = self.__mge.get_num_converged_descents()
		return results

		
	def __optimize_edit_cost_vector(self):
		"""Learn edit cost vector.	
		"""
		 # Initialize label costs randomly.
		if self.__init_method == 'random':
			# Initialize label costs.
			self.__initialize_label_costs()
				
			# Optimize edit cost matrices.
			self.__optimize_ecm_by_kernel_distances()
		# Initialize all label costs with the same value.
		elif self.__init_method == 'uniform': # random
			pass
	
		elif self.__fit_method == 'random': # random
			if self.__ged_options['edit_cost'] == 'LETTER':
				self.__edit_cost_constants = random.sample(range(1, 1000), 3)
				self.__edit_cost_constants = [item * 0.001 for item in self.__edit_cost_constants]
			elif self.__ged_options['edit_cost'] == 'LETTER2':
				random.seed(time.time())
				self.__edit_cost_constants = random.sample(range(1, 1000), 5)
				self.__edit_cost_constants = [item * 0.01 for item in self.__edit_cost_constants]
			elif self.__ged_options['edit_cost'] == 'NON_SYMBOLIC':
				self.__edit_cost_constants = random.sample(range(1, 1000), 6)
				self.__edit_cost_constants = [item * 0.01 for item in self.__edit_cost_constants]
				if self._dataset.node_attrs == []:
					self.__edit_cost_constants[2] = 0
				if self._dataset.edge_attrs == []:
					self.__edit_cost_constants[5] = 0
			else:
				self.__edit_cost_constants = random.sample(range(1, 1000), 6)
				self.__edit_cost_constants = [item * 0.01 for item in self.__edit_cost_constants]
			if self._verbose >= 2:
				print('edit cost constants used:', self.__edit_cost_constants)
		elif self.__fit_method == 'expert': # expert
			if self.__init_ecc is None:
				if self.__ged_options['edit_cost'] == 'LETTER':
					self.__edit_cost_constants = [0.9, 1.7, 0.75] 
				elif self.__ged_options['edit_cost'] == 'LETTER2':
					self.__edit_cost_constants = [0.675, 0.675, 0.75, 0.425, 0.425]
				else:
					self.__edit_cost_constants = [3, 3, 1, 3, 3, 1] 
			else:
				self.__edit_cost_constants = self.__init_ecc
		elif self.__fit_method == 'k-graphs':
			if self.__init_ecc is None:
				if self.__ged_options['edit_cost'] == 'LETTER':
					self.__init_ecc = [0.9, 1.7, 0.75] 
				elif self.__ged_options['edit_cost'] == 'LETTER2':
					self.__init_ecc = [0.675, 0.675, 0.75, 0.425, 0.425]
				elif self.__ged_options['edit_cost'] == 'NON_SYMBOLIC':
					self.__init_ecc = [0, 0, 1, 1, 1, 0]
					if self._dataset.node_attrs == []:
						self.__init_ecc[2] = 0
					if self._dataset.edge_attrs == []:
						self.__init_ecc[5] = 0
				else:
					self.__init_ecc = [3, 3, 1, 3, 3, 1] 
			# optimize on the k-graph subset.
			self.__optimize_ecm_by_kernel_distances()
		elif self.__fit_method == 'whole-dataset':
			if self.__init_ecc is None:
				if self.__ged_options['edit_cost'] == 'LETTER':
					self.__init_ecc = [0.9, 1.7, 0.75] 
				elif self.__ged_options['edit_cost'] == 'LETTER2':
					self.__init_ecc = [0.675, 0.675, 0.75, 0.425, 0.425]
				else:
					self.__init_ecc = [3, 3, 1, 3, 3, 1] 
			# optimizeon the whole set.
			self.__optimize_ecc_by_kernel_distances()
		elif self.__fit_method == 'precomputed':
			pass
		
		
	def __initialize_label_costs(self):
		self.__initialize_node_label_costs()
		self.__initialize_edge_label_costs()
				
				
	def __initialize_node_label_costs(self):
		# Get list of node labels.
		nls = self._dataset.get_all_node_labels()
		# Generate random costs.
		nb_nl = int((len(nls) * (len(nls) - 1)) / 2 + 2 * len(nls))
		rand_costs = random.sample(range(1, 10 * nb_nl + 1), nb_nl)
		rand_costs /= np.max(rand_costs) # @todo: maybe not needed.
		self.__node_label_costs = rand_costs


	def __initialize_edge_label_costs(self):
		# Get list of edge labels.
		els = self._dataset.get_all_edge_labels()
		# Generate random costs.
		nb_el = int((len(els) * (len(els) - 1)) / 2 + 2 * len(els))
		rand_costs = random.sample(range(1, 10 * nb_el + 1), nb_el)
		rand_costs /= np.max(rand_costs) # @todo: maybe not needed.
		self.__edge_label_costs = rand_costs
		
		
	def __optimize_ecm_by_kernel_distances(self):		
		# compute distances in feature space.
		dis_k_mat, _, _, _ = self._graph_kernel.compute_distance_matrix()
		dis_k_vec = []
		for i in range(len(dis_k_mat)):
	#		for j in range(i, len(dis_k_mat)):
			for j in range(i + 1, len(dis_k_mat)):
				dis_k_vec.append(dis_k_mat[i, j])
		dis_k_vec = np.array(dis_k_vec)
		
		# Set GEDEnv options.
# 		graphs = [self.__clean_graph(g) for g in self._dataset.graphs]
# 		self.__edit_cost_constants = self.__init_ecc
		options = self.__ged_options.copy()
		options['edit_cost_constants'] = self.__edit_cost_constants # @todo: not needed.
		options['node_labels'] = self._dataset.node_labels
		options['edge_labels'] = self._dataset.edge_labels
# 		options['node_attrs'] = self._dataset.node_attrs
# 		options['edge_attrs'] = self._dataset.edge_attrs
		options['node_label_costs'] = self.__node_label_costs
		options['edge_label_costs'] = self.__edge_label_costs
		
		# Learner cost matrices.
		# Initialize cost learner.
		cml = CostMatricesLearner(edit_cost='CONSTANT', triangle_rule=False, allow_zeros=True, parallel=self.__parallel, verbose=self._verbose) # @todo
		cml.set_update_params(time_limit_in_sec=self.__time_limit_in_sec, max_itrs=self.__max_itrs, max_itrs_without_update=self.__max_itrs_without_update, epsilon_residual=self.__epsilon_residual, epsilon_ec=self.__epsilon_ec)
		# Run cost learner.
		cml.update(dis_k_vec, self._dataset.graphs, options)
		
		# Get results.
		results = cml.get_results()
		self.__converged = results['converged']
		self.__itrs = results['itrs']
		self.__num_updates_ecs = results['num_updates_ecs']
		cost_list = results['cost_list']
		self.__node_label_costs = cost_list[-1][0:len(self.__node_label_costs)]
		self.__edge_label_costs = cost_list[-1][len(self.__node_label_costs):]

	
	def __gmg_bcu(self):
		"""
		The local search algorithm based on block coordinate update (BCU) for estimating a generalized median graph (GMG).

		Returns
		-------
		None.

		"""
		# Set up the ged environment.
		ged_env = GEDEnv() # @todo: maybe create a ged_env as a private varible.
		# gedlibpy.restart_env()
		ged_env.set_edit_cost(self.__ged_options['edit_cost'], edit_cost_constants=self.__edit_cost_constants)
		graphs = [self.__clean_graph(g) for g in self._dataset.graphs]
		for g in graphs:
			ged_env.add_nx_graph(g, '')
		graph_ids = ged_env.get_all_graph_ids()
	
		node_labels = ged_env.get_all_node_labels()
		edge_labels =  ged_env.get_all_edge_labels()
		node_label_costs = label_costs_to_matrix(self.__node_label_costs, len(node_labels))
		edge_label_costs = label_costs_to_matrix(self.__edge_label_costs, len(edge_labels))
		ged_env.set_label_costs(node_label_costs, edge_label_costs)
	
		set_median_id = ged_env.add_graph('set_median')
		gen_median_id = ged_env.add_graph('gen_median')
		ged_env.init(init_type=self.__ged_options['init_option'])
		
		# Set up the madian graph estimator.
		self.__mge = MedianGraphEstimatorCML(ged_env, constant_node_costs(self.__ged_options['edit_cost']))
		self.__mge.set_refine_method(self.__ged_options['method'], self.__ged_options)
		options = self.__mge_options.copy()
		if not 'seed' in options:
			options['seed'] = int(round(time.time() * 1000)) # @todo: may not work correctly for possible parallel usage.
		options['parallel'] = self.__parallel
		
		# Select the GED algorithm.
		self.__mge.set_options(mge_options_to_string(options))
		self.__mge.set_label_names(node_labels=self._dataset.node_labels, 
					  edge_labels=self._dataset.edge_labels, 
					  node_attrs=self._dataset.node_attrs, 
					  edge_attrs=self._dataset.edge_attrs)
		ged_options = self.__ged_options.copy()
		if self.__parallel:
			ged_options['threads'] = 1
		self.__mge.set_init_method(ged_options['method'], ged_options)
		self.__mge.set_descent_method(ged_options['method'], ged_options)
		
		# Run the estimator.
		self.__mge.run(graph_ids, set_median_id, gen_median_id)
		
		# Get SODs.
		self.__sod_set_median = self.__mge.get_sum_of_distances('initialized')
		self.__sod_gen_median = self.__mge.get_sum_of_distances('converged')
		
		# Get median graphs.
		self.__set_median = ged_env.get_nx_graph(set_median_id)
		self.__gen_median = ged_env.get_nx_graph(gen_median_id)
		
		
	def __compute_distances_to_true_median(self):		
		# compute distance in kernel space for set median.
		kernels_to_sm, _ = self._graph_kernel.compute(self.__set_median, self._dataset.graphs, **self._kernel_options)
		kernel_sm, _ = self._graph_kernel.compute(self.__set_median, self.__set_median, **self._kernel_options)
		if self._kernel_options['normalize']:
			kernels_to_sm = [kernels_to_sm[i] / np.sqrt(self.__gram_matrix_unnorm[i, i] * kernel_sm) for i in range(len(kernels_to_sm))] # normalize 
			kernel_sm = 1
		# @todo: not correct kernel value
		gram_with_sm = np.concatenate((np.array([kernels_to_sm]), np.copy(self._graph_kernel.gram_matrix)), axis=0)
		gram_with_sm = np.concatenate((np.array([[kernel_sm] + kernels_to_sm]).T, gram_with_sm), axis=1)
		self.__k_dis_set_median = compute_k_dis(0, range(1, 1+len(self._dataset.graphs)), 
										  [1 / len(self._dataset.graphs)] * len(self._dataset.graphs),
										  gram_with_sm, withterm3=False)
		
		# compute distance in kernel space for generalized median.
		kernels_to_gm, _ = self._graph_kernel.compute(self.__gen_median, self._dataset.graphs, **self._kernel_options)
		kernel_gm, _ = self._graph_kernel.compute(self.__gen_median, self.__gen_median, **self._kernel_options)
		if self._kernel_options['normalize']:
			kernels_to_gm = [kernels_to_gm[i] / np.sqrt(self.__gram_matrix_unnorm[i, i] * kernel_gm) for i in range(len(kernels_to_gm))] # normalize
			kernel_gm = 1
		gram_with_gm = np.concatenate((np.array([kernels_to_gm]), np.copy(self._graph_kernel.gram_matrix)), axis=0)
		gram_with_gm = np.concatenate((np.array([[kernel_gm] + kernels_to_gm]).T, gram_with_gm), axis=1)
		self.__k_dis_gen_median = compute_k_dis(0, range(1, 1+len(self._dataset.graphs)), 
										  [1 / len(self._dataset.graphs)] * len(self._dataset.graphs),
										  gram_with_gm, withterm3=False)
				
		# compute distance in kernel space for each graph in median set.
		k_dis_median_set = []
		for idx in range(len(self._dataset.graphs)):
			k_dis_median_set.append(compute_k_dis(idx+1, range(1, 1+len(self._dataset.graphs)), 
								 [1 / len(self._dataset.graphs)] * len(self._dataset.graphs), 
								 gram_with_gm, withterm3=False))
		idx_k_dis_median_set_min = np.argmin(k_dis_median_set)
		self.__k_dis_dataset = k_dis_median_set[idx_k_dis_median_set_min]
		self.__best_from_dataset = self._dataset.graphs[idx_k_dis_median_set_min].copy()
			
		if self._verbose >= 2:
			print()
			print('distance in kernel space for set median:', self.__k_dis_set_median)
			print('distance in kernel space for generalized median:', self.__k_dis_gen_median)
			print('minimum distance in kernel space for each graph in median set:', self.__k_dis_dataset)
			print('distance in kernel space for each graph in median set:', k_dis_median_set)	
			
			
# 	def __clean_graph(self, G, node_labels=[], edge_labels=[], node_attrs=[], edge_attrs=[]):
	def __clean_graph(self, G): # @todo: this may not be needed when datafile is updated.
		"""
		Cleans node and edge labels and attributes of the given graph.
		"""
		G_new = nx.Graph(**G.graph)
		for nd, attrs in G.nodes(data=True):
			G_new.add_node(str(nd)) # @todo: should we keep this as str()?
			for l_name in self._dataset.node_labels:
				G_new.nodes[str(nd)][l_name] = str(attrs[l_name])
			for a_name in self._dataset.node_attrs:
				G_new.nodes[str(nd)][a_name] = str(attrs[a_name])
		for nd1, nd2, attrs in G.edges(data=True):
			G_new.add_edge(str(nd1), str(nd2))
			for l_name in self._dataset.edge_labels:
				G_new.edges[str(nd1), str(nd2)][l_name] = str(attrs[l_name])		
			for a_name in self._dataset.edge_attrs:
				G_new.edges[str(nd1), str(nd2)][a_name] = str(attrs[a_name])		
		return G_new
			
			
	@property
	def mge(self):
		return self.__mge
	
	@property
	def ged_options(self):
		return self.__ged_options

	@ged_options.setter
	def ged_options(self, value):
		self.__ged_options = value		

	
	@property
	def mge_options(self):
		return self.__mge_options

	@mge_options.setter
	def mge_options(self, value):
		self.__mge_options = value		


	@property
	def fit_method(self):
		return self.__fit_method

	@fit_method.setter
	def fit_method(self, value):
		self.__fit_method = value
		
		
	@property
	def init_ecc(self):
		return self.__init_ecc

	@init_ecc.setter
	def init_ecc(self, value):
		self.__init_ecc = value
		
	
	@property
	def set_median(self):
		return self.__set_median


	@property
	def gen_median(self):
		return self.__gen_median
	
	
	@property
	def best_from_dataset(self):
		return self.__best_from_dataset
	
	
	@property
	def gram_matrix_unnorm(self):
		return self.__gram_matrix_unnorm
	
	@gram_matrix_unnorm.setter
	def gram_matrix_unnorm(self, value):
		self.__gram_matrix_unnorm = value