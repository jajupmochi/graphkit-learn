#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:27:22 2020

@author: ljia
"""
import numpy as np
import time
import random
import multiprocessing
import networkx as nx
import cvxpy as cp
from gklearn.preimage import PreimageGenerator
from gklearn.preimage.utils import compute_k_dis
from gklearn.ged.util import compute_geds, ged_options_to_string
from gklearn.ged.median import MedianGraphEstimator
from gklearn.ged.median import constant_node_costs,mge_options_to_string
from gklearn.gedlib import librariesImport, gedlibpy
from gklearn.utils import Timer
from gklearn.utils.utils import get_graph_kernel_by_name


class MedianPreimageGenerator(PreimageGenerator):
	
	def __init__(self, dataset=None):
		PreimageGenerator.__init__(self, dataset=dataset)
		# arguments to set.
		self._mge = None
		self._ged_options = {}
		self._mge_options = {}
		self._fit_method = 'k-graphs'
		self._init_ecc = None
		self._parallel = True
		self._n_jobs = multiprocessing.cpu_count()
		self._ds_name = None
		self._time_limit_in_sec = 0
		self._max_itrs = 100
		self._max_itrs_without_update = 3
		self._epsilon_residual = 0.01
		self._epsilon_ec = 0.1
		self._allow_zeros = False
		self._triangle_rule = True
		# values to compute.
		self._runtime_optimize_ec = None
		self._runtime_generate_preimage = None
		self._runtime_total = None
		self._set_median = None
		self._gen_median = None
		self._best_from_dataset = None
		self._sod_set_median = None
		self._sod_gen_median = None
		self._k_dis_set_median = None
		self._k_dis_gen_median = None
		self._k_dis_dataset = None
		self._itrs = 0
		self._converged = False
		self._num_updates_ecc = 0
		# values that can be set or to be computed.
		self._edit_cost_constants = []
		self._gram_matrix_unnorm = None
		self._runtime_precompute_gm = None

		
	def set_options(self, **kwargs):
		self._kernel_options = kwargs.get('kernel_options', {})
		self._graph_kernel = kwargs.get('graph_kernel', None)
		self._verbose = kwargs.get('verbose', 2)
		self._ged_options = kwargs.get('ged_options', {})
		self._mge_options = kwargs.get('mge_options', {})
		self._fit_method = kwargs.get('fit_method', 'k-graphs')
		self._init_ecc = kwargs.get('init_ecc', None)
		self._edit_cost_constants = kwargs.get('edit_cost_constants', [])
		self._parallel = kwargs.get('parallel', True)
		self._n_jobs = kwargs.get('n_jobs', multiprocessing.cpu_count())
		self._ds_name = kwargs.get('ds_name', None)
		self._time_limit_in_sec = kwargs.get('time_limit_in_sec', 0)
		self._max_itrs = kwargs.get('max_itrs', 100)
		self._max_itrs_without_update = kwargs.get('max_itrs_without_update', 3)
		self._epsilon_residual = kwargs.get('epsilon_residual', 0.01)
		self._epsilon_ec = kwargs.get('epsilon_ec', 0.1)
		self._gram_matrix_unnorm = kwargs.get('gram_matrix_unnorm', None)
		self._runtime_precompute_gm = kwargs.get('runtime_precompute_gm', None)
		self._allow_zeros = kwargs.get('allow_zeros', False)
		self._triangle_rule = kwargs.get('triangle_rule', True)
		
		
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
			
		if self._fit_method != 'k-graphs' and self._fit_method != 'whole-dataset':
			start = time.time()
			self._runtime_precompute_gm = 0
			end_precompute_gm = start
		
		# 2. optimize edit cost constants. 
		self._optimize_edit_cost_constants()
		end_optimize_ec = time.time()
		self._runtime_optimize_ec = end_optimize_ec - end_precompute_gm
		
		# 3. compute set median and gen median using optimized edit costs.
		if self._verbose >= 2:
			print('\nstart computing set median and gen median using optimized edit costs...\n')
		self._gmg_bcu()
		end_generate_preimage = time.time()
		self._runtime_generate_preimage = end_generate_preimage - end_optimize_ec
		self._runtime_total = end_generate_preimage - start
		if self._verbose >= 2:
			print('medians computed.')
			print('SOD of the set median: ', self._sod_set_median)
			print('SOD of the generalized median: ', self._sod_gen_median)
			
		# 4. compute kernel distances to the true median.
		if self._verbose >= 2:
			print('\nstart computing distances to true median....\n')
		self._compute_distances_to_true_median()

		# 5. print out results.
		if self._verbose:
			print()
			print('================================================================================')
			print('Finished generation of preimages.')
			print('--------------------------------------------------------------------------------')
			print('The optimized edit cost constants:', self._edit_cost_constants)
			print('SOD of the set median:', self._sod_set_median)
			print('SOD of the generalized median:', self._sod_gen_median)
			print('Distance in kernel space for set median:', self._k_dis_set_median)
			print('Distance in kernel space for generalized median:', self._k_dis_gen_median)
			print('Minimum distance in kernel space for each graph in median set:', self._k_dis_dataset)
			print('Time to pre-compute Gram matrix:', self._runtime_precompute_gm)
			print('Time to optimize edit costs:', self._runtime_optimize_ec)
			print('Time to generate pre-images:', self._runtime_generate_preimage)
			print('Total time:', self._runtime_total)
			print('Total number of iterations for optimizing:', self._itrs)
			print('Total number of updating edit costs:', self._num_updates_ecc)
			print('Is optimization of edit costs converged:', self._converged)
			print('================================================================================')
			print()


	def get_results(self):
		results = {}
		results['edit_cost_constants'] = self._edit_cost_constants
		results['runtime_precompute_gm'] = self._runtime_precompute_gm
		results['runtime_optimize_ec'] = self._runtime_optimize_ec
		results['runtime_generate_preimage'] = self._runtime_generate_preimage
		results['runtime_total'] = self._runtime_total
		results['sod_set_median'] = self._sod_set_median
		results['sod_gen_median'] = self._sod_gen_median
		results['k_dis_set_median'] = self._k_dis_set_median
		results['k_dis_gen_median'] = self._k_dis_gen_median
		results['k_dis_dataset'] = self._k_dis_dataset
		results['itrs'] = self._itrs
		results['converged'] = self._converged
		results['num_updates_ecc'] = self._num_updates_ecc
		results['mge'] = {}
		results['mge']['num_decrease_order'] = self._mge.get_num_times_order_decreased()
		results['mge']['num_increase_order'] = self._mge.get_num_times_order_increased()
		results['mge']['num_converged_descents'] = self._mge.get_num_converged_descents()
# 		results['ged_matrix_set_median'] = self._mge.ged_matrix_set_median_tmp
		return results

		
	def _optimize_edit_cost_constants(self):
		"""fit edit cost constants.	
		"""
		if self._fit_method == 'random': # random
			if self._ged_options['edit_cost'] == 'LETTER':
				self._edit_cost_constants = random.sample(range(1, 1000), 3)
				self._edit_cost_constants = [item * 0.001 for item in self._edit_cost_constants]
			elif self._ged_options['edit_cost'] == 'LETTER2':
				random.seed(time.time())
				self._edit_cost_constants = random.sample(range(1, 1000), 5)
				self._edit_cost_constants = [item * 0.01 for item in self._edit_cost_constants]
			elif self._ged_options['edit_cost'] == 'NON_SYMBOLIC':
				self._edit_cost_constants = random.sample(range(1, 1000), 6)
				self._edit_cost_constants = [item * 0.01 for item in self._edit_cost_constants]
				if self._dataset.node_attrs == []:
					self._edit_cost_constants[2] = 0
				if self._dataset.edge_attrs == []:
					self._edit_cost_constants[5] = 0
			else:
				self._edit_cost_constants = random.sample(range(1, 1000), 6)
				self._edit_cost_constants = [item * 0.01 for item in self._edit_cost_constants]
			if self._verbose >= 2:
				print('edit cost constants used:', self._edit_cost_constants)
		elif self._fit_method == 'expert': # expert
			if self._init_ecc is None:
				if self._ged_options['edit_cost'] == 'LETTER':
					self._edit_cost_constants = [0.9, 1.7, 0.75] 
				elif self._ged_options['edit_cost'] == 'LETTER2':
					self._edit_cost_constants = [0.675, 0.675, 0.75, 0.425, 0.425]
				else:
					self._edit_cost_constants = [3, 3, 1, 3, 3, 1] 
			else:
				self._edit_cost_constants = self._init_ecc
		elif self._fit_method == 'k-graphs':
			if self._init_ecc is None:
				if self._ged_options['edit_cost'] == 'LETTER':
					self._init_ecc = [0.9, 1.7, 0.75] 
				elif self._ged_options['edit_cost'] == 'LETTER2':
					self._init_ecc = [0.675, 0.675, 0.75, 0.425, 0.425]
				elif self._ged_options['edit_cost'] == 'NON_SYMBOLIC':
					self._init_ecc = [0, 0, 1, 1, 1, 0]
					if self._dataset.node_attrs == []:
						self._init_ecc[2] = 0
					if self._dataset.edge_attrs == []:
						self._init_ecc[5] = 0
				else:
					self._init_ecc = [3, 3, 1, 3, 3, 1] 
			# optimize on the k-graph subset.
			self._optimize_ecc_by_kernel_distances()
		elif self._fit_method == 'whole-dataset':
			if self._init_ecc is None:
				if self._ged_options['edit_cost'] == 'LETTER':
					self._init_ecc = [0.9, 1.7, 0.75] 
				elif self._ged_options['edit_cost'] == 'LETTER2':
					self._init_ecc = [0.675, 0.675, 0.75, 0.425, 0.425]
				else:
					self._init_ecc = [3, 3, 1, 3, 3, 1] 
			# optimizeon the whole set.
			self._optimize_ecc_by_kernel_distances()
		elif self._fit_method == 'precomputed':
			pass
		
		
	def _optimize_ecc_by_kernel_distances(self):		
		# compute distances in feature space.
		dis_k_mat, _, _, _ = self._graph_kernel.compute_distance_matrix()
		dis_k_vec = []
		for i in range(len(dis_k_mat)):
	#		for j in range(i, len(dis_k_mat)):
			for j in range(i + 1, len(dis_k_mat)):
				dis_k_vec.append(dis_k_mat[i, j])
		dis_k_vec = np.array(dis_k_vec)
		
		# init ged.
		if self._verbose >= 2:
			print('\ninitial:')
		time0 = time.time()
		graphs = [self._clean_graph(g) for g in self._dataset.graphs]
		self._edit_cost_constants = self._init_ecc
		options = self._ged_options.copy()
		options['edit_cost_constants'] = self._edit_cost_constants # @todo
		options['node_labels'] = self._dataset.node_labels
		options['edge_labels'] = self._dataset.edge_labels
		options['node_attrs'] = self._dataset.node_attrs
		options['edge_attrs'] = self._dataset.edge_attrs
		ged_vec_init, ged_mat, n_edit_operations = compute_geds(graphs, options=options, parallel=self._parallel, verbose=(self._verbose > 1))
		residual_list = [np.sqrt(np.sum(np.square(np.array(ged_vec_init) - dis_k_vec)))]	
		time_list = [time.time() - time0]
		edit_cost_list = [self._init_ecc]  
		nb_cost_mat = np.array(n_edit_operations)
		nb_cost_mat_list = [nb_cost_mat]
		if self._verbose >= 2:
			print('Current edit cost constants:', self._edit_cost_constants)
			print('Residual list:', residual_list)
		
		# run iteration from initial edit costs.
		self._converged = False
		itrs_without_update = 0
		self._itrs = 0
		self._num_updates_ecc = 0
		timer = Timer(self._time_limit_in_sec)
		while not self._termination_criterion_met(self._converged, timer, self._itrs, itrs_without_update):
			if self._verbose >= 2:
				print('\niteration', self._itrs + 1)
			time0 = time.time()
			# "fit" geds to distances in feature space by tuning edit costs using theLeast Squares Method.
# 			np.savez('results/xp_fit_method/fit_data_debug' + str(self._itrs) + '.gm', 
# 					 nb_cost_mat=nb_cost_mat, dis_k_vec=dis_k_vec, 
# 					 n_edit_operations=n_edit_operations, ged_vec_init=ged_vec_init,
# 					 ged_mat=ged_mat)
			self._edit_cost_constants, _ = self._update_ecc(nb_cost_mat, dis_k_vec)
			for i in range(len(self._edit_cost_constants)):
				if -1e-9 <= self._edit_cost_constants[i] <= 1e-9:
					self._edit_cost_constants[i] = 0
				if self._edit_cost_constants[i] < 0:
					raise ValueError('The edit cost is negative.')
	#		for i in range(len(self._edit_cost_constants)):
	#			if self._edit_cost_constants[i] < 0:
	#				self._edit_cost_constants[i] = 0
	
			# compute new GEDs and numbers of edit operations.
			options = self._ged_options.copy() # np.array([self._edit_cost_constants[0], self._edit_cost_constants[1], 0.75])
			options['edit_cost_constants'] = self._edit_cost_constants # @todo
			options['node_labels'] = self._dataset.node_labels
			options['edge_labels'] = self._dataset.edge_labels
			options['node_attrs'] = self._dataset.node_attrs
			options['edge_attrs'] = self._dataset.edge_attrs
			ged_vec, ged_mat, n_edit_operations = compute_geds(graphs, options=options, parallel=self._parallel, verbose=(self._verbose > 1))
			residual_list.append(np.sqrt(np.sum(np.square(np.array(ged_vec) - dis_k_vec))))
			time_list.append(time.time() - time0)
			edit_cost_list.append(self._edit_cost_constants)
			nb_cost_mat = np.array(n_edit_operations)
			nb_cost_mat_list.append(nb_cost_mat)	
				
			# check convergency.
			ec_changed = False
			for i, cost in enumerate(self._edit_cost_constants):
				if cost == 0:
 					if edit_cost_list[-2][i] > self._epsilon_ec:
						 ec_changed = True
						 break
				elif abs(cost - edit_cost_list[-2][i]) / cost > self._epsilon_ec:
 					ec_changed = True
 					break
# 				if abs(cost - edit_cost_list[-2][i]) > self._epsilon_ec:
#  					ec_changed = True
#  					break
			residual_changed = False
			if residual_list[-1] == 0:
				if residual_list[-2] > self._epsilon_residual:
					residual_changed = True
			elif abs(residual_list[-1] - residual_list[-2]) / residual_list[-1] > self._epsilon_residual:
				residual_changed = True
			self._converged = not (ec_changed or residual_changed)
			if self._converged:
				itrs_without_update += 1
			else:
				itrs_without_update = 0
				self._num_updates_ecc += 1
				
			# print current states.
			if self._verbose >= 2:
				print()
				print('-------------------------------------------------------------------------')
				print('States of iteration', self._itrs + 1)
				print('-------------------------------------------------------------------------')
# 				print('Time spend:', self._runtime_optimize_ec)
				print('Total number of iterations for optimizing:', self._itrs + 1)
				print('Total number of updating edit costs:', self._num_updates_ecc)
				print('Was optimization of edit costs converged:', self._converged)
				print('Did edit costs change:', ec_changed)
				print('Did residual change:', residual_changed)
				print('Iterations without update:', itrs_without_update)
				print('Current edit cost constants:', self._edit_cost_constants)
				print('Residual list:', residual_list)
				print('-------------------------------------------------------------------------')
			
			self._itrs += 1


	def _termination_criterion_met(self, converged, timer, itr, itrs_without_update):
		if timer.expired() or (itr >= self._max_itrs if self._max_itrs >= 0 else False):
# 			if self._state == AlgorithmState.TERMINATED:
# 				self._state = AlgorithmState.INITIALIZED
			return True
		return converged or (itrs_without_update > self._max_itrs_without_update if self._max_itrs_without_update >= 0 else False)


	def _update_ecc(self, nb_cost_mat, dis_k_vec, rw_constraints='inequality'):
	#	if self._ds_name == 'Letter-high':
		if self._ged_options['edit_cost'] == 'LETTER':
			raise Exception('Cannot compute for cost "LETTER".')
			pass
	#		# method 1: set alpha automatically, just tune c_vir and c_eir by 
	#		# LMS using cvxpy.
	#		alpha = 0.5
	#		coeff = 100 # np.max(alpha * nb_cost_mat[:,4] / dis_k_vec)
	##		if np.count_nonzero(nb_cost_mat[:,4]) == 0:
	##			alpha = 0.75
	##		else:
	##			alpha = np.min([dis_k_vec / c_vs for c_vs in nb_cost_mat[:,4] if c_vs != 0])
	##		alpha = alpha * 0.99
	#		param_vir = alpha * (nb_cost_mat[:,0] + nb_cost_mat[:,1])
	#		param_eir = (1 - alpha) * (nb_cost_mat[:,4] + nb_cost_mat[:,5])
	#		nb_cost_mat_new = np.column_stack((param_vir, param_eir))
	#		dis_new = coeff * dis_k_vec - alpha * nb_cost_mat[:,3]
	#		
	#		x = cp.Variable(nb_cost_mat_new.shape[1])
	#		cost = cp.sum_squares(nb_cost_mat_new * x - dis_new)
	#		constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])]]
	#		prob = cp.Problem(cp.Minimize(cost), constraints)
	#		prob.solve()
	#		edit_costs_new = x.value
	#		edit_costs_new = np.array([edit_costs_new[0], edit_costs_new[1], alpha])
	#		residual = np.sqrt(prob.value)
		
	#		# method 2: tune c_vir, c_eir and alpha by nonlinear programming by 
	#		# scipy.optimize.minimize.
	#		w0 = nb_cost_mat[:,0] + nb_cost_mat[:,1]
	#		w1 = nb_cost_mat[:,4] + nb_cost_mat[:,5]
	#		w2 = nb_cost_mat[:,3]
	#		w3 = dis_k_vec
	#		func_min = lambda x: np.sum((w0 * x[0] * x[3] + w1 * x[1] * (1 - x[2]) \
	#							 + w2 * x[2] - w3 * x[3]) ** 2)
	#		bounds = ((0, None), (0., None), (0.5, 0.5), (0, None))
	#		res = minimize(func_min, [0.9, 1.7, 0.75, 10], bounds=bounds)
	#		edit_costs_new = res.x[0:3]
	#		residual = res.fun
		
		# method 3: tune c_vir, c_eir and alpha by nonlinear programming using cvxpy.
		
		
	#		# method 4: tune c_vir, c_eir and alpha by QP function
	#		# scipy.optimize.least_squares. An initial guess is required.
	#		w0 = nb_cost_mat[:,0] + nb_cost_mat[:,1]
	#		w1 = nb_cost_mat[:,4] + nb_cost_mat[:,5]
	#		w2 = nb_cost_mat[:,3]
	#		w3 = dis_k_vec
	#		func = lambda x: (w0 * x[0] * x[3] + w1 * x[1] * (1 - x[2]) \
	#							 + w2 * x[2] - w3 * x[3]) ** 2
	#		res = optimize.root(func, [0.9, 1.7, 0.75, 100])
	#		edit_costs_new = res.x
	#		residual = None
		elif self._ged_options['edit_cost'] == 'LETTER2':
	#			# 1. if c_vi != c_vr, c_ei != c_er.
	#			nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
	#			x = cp.Variable(nb_cost_mat_new.shape[1])
	#			cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
	##			# 1.1 no constraints.
	##			constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])]]
	#			# 1.2 c_vs <= c_vi + c_vr.
	#			constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
	#						   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]			
	##			# 2. if c_vi == c_vr, c_ei == c_er.
	##			nb_cost_mat_new = nb_cost_mat[:,[0,3,4]]
	##			nb_cost_mat_new[:,0] += nb_cost_mat[:,1]
	##			nb_cost_mat_new[:,2] += nb_cost_mat[:,5]
	##			x = cp.Variable(nb_cost_mat_new.shape[1])
	##			cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
	##			# 2.1 no constraints.
	##			constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])]]
	###			# 2.2 c_vs <= c_vi + c_vr.
	###			constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
	###						   np.array([2.0, -1.0, 0.0]).T@x >= 0.0]	 
	#			
	#			prob = cp.Problem(cp.Minimize(cost_fun), constraints)
	#			prob.solve()
	#			edit_costs_new = [x.value[0], x.value[0], x.value[1], x.value[2], x.value[2]]
	#			edit_costs_new = np.array(edit_costs_new)
	#			residual = np.sqrt(prob.value)
			if not self._triangle_rule and self._allow_zeros:
				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
				constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
							   np.array([1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 1.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 0.0, 1.0]).T@x >= 0.01]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif self._triangle_rule and self._allow_zeros:
				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
				constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
							   np.array([1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 1.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 0.0, 1.0]).T@x >= 0.01,
							   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif not self._triangle_rule and not self._allow_zeros:
				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				prob.solve()
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
	#			elif method == 'inequality_modified':
	#				# c_vs <= c_vi + c_vr.
	#				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
	#				x = cp.Variable(nb_cost_mat_new.shape[1])
	#				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
	#				constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
	#							   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
	#				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
	#				prob.solve()
	#				# use same costs for insertion and removal rather than the fitted costs.
	#				edit_costs_new = [x.value[0], x.value[0], x.value[1], x.value[2], x.value[2]]
	#				edit_costs_new = np.array(edit_costs_new)
	#				residual = np.sqrt(prob.value)
			elif self._triangle_rule and not self._allow_zeros:
				# c_vs <= c_vi + c_vr.
				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
							   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)				
			elif rw_constraints == '2constraints': # @todo: rearrange it later.
				# c_vs <= c_vi + c_vr and c_vi == c_vr, c_ei == c_er.
				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
							   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0,
							   np.array([1.0, -1.0, 0.0, 0.0, 0.0]).T@x == 0.0,
							   np.array([0.0, 0.0, 0.0, 1.0, -1.0]).T@x == 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				prob.solve()
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)

		elif self._ged_options['edit_cost'] == 'NON_SYMBOLIC':
			is_n_attr = np.count_nonzero(nb_cost_mat[:,2])
			is_e_attr = np.count_nonzero(nb_cost_mat[:,5])
			
			if self._ds_name == 'SYNTHETICnew': # @todo: rearrenge this later.
	#			nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
				nb_cost_mat_new = nb_cost_mat[:,[2,3,4]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
	#			constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
	#						   np.array([0.0, 0.0, 0.0, 1.0, -1.0]).T@x == 0.0]
	#			constraints = [x >= [0.0001 for i in range(nb_cost_mat_new.shape[1])]]
				constraints = [x >= [0.0001 for i in range(nb_cost_mat_new.shape[1])],
					   np.array([0.0, 1.0, -1.0]).T@x == 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				prob.solve()
	#			print(x.value)
				edit_costs_new = np.concatenate((np.array([0.0, 0.0]), x.value, 
												 np.array([0.0])))
				residual = np.sqrt(prob.value)
				
			elif not self._triangle_rule and self._allow_zeros:
				if is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = x.value
					residual = np.sqrt(prob.value)
				elif is_n_attr and not is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 1.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 0.0, 1.0]).T@x >= 0.01]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value, np.array([0.0])))
					residual = np.sqrt(prob.value)
				elif not is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 1.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 1.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)
			elif self._triangle_rule and self._allow_zeros:
				if is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
								   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
								   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = x.value
					residual = np.sqrt(prob.value)
				elif is_n_attr and not is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 1.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 0.0, 1.0]).T@x >= 0.01,
								   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value, np.array([0.0])))
					residual = np.sqrt(prob.value)
				elif not is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 1.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 1.0, 0.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
								   np.array([0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)
			elif not self._triangle_rule and not self._allow_zeros:
				if is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = x.value
					residual = np.sqrt(prob.value)
				elif is_n_attr and not is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value, np.array([0.0])))
					residual = np.sqrt(prob.value)
				elif not is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)
			elif self._triangle_rule and not self._allow_zeros:
				# c_vs <= c_vi + c_vr.
				if is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
								   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = x.value
					residual = np.sqrt(prob.value)
				elif is_n_attr and not is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value, np.array([0.0])))
					residual = np.sqrt(prob.value)
				elif not is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self._execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)

		elif self._ged_options['edit_cost'] == 'CONSTANT': # @todo: node/edge may not labeled.
			if not self._triangle_rule and self._allow_zeros:
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.0 for i in range(nb_cost_mat.shape[1])],
							   np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif self._triangle_rule and self._allow_zeros:
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.0 for i in range(nb_cost_mat.shape[1])],
							   np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
							   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
							   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif not self._triangle_rule and not self._allow_zeros:
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat.shape[1])]]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif self._triangle_rule and not self._allow_zeros:
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat.shape[1])],
							   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
							   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
		else:
			raise Exception('The edit cost "', self._ged_options['edit_cost'], '" is not supported for update progress.')
	#	# method 1: simple least square method.
	#	edit_costs_new, residual, _, _ = np.linalg.lstsq(nb_cost_mat, dis_k_vec,
	#													 rcond=None)
		
	#	# method 2: least square method with x_i >= 0.
	#	edit_costs_new, residual = optimize.nnls(nb_cost_mat, dis_k_vec)
		
		# method 3: solve as a quadratic program with constraints.
	#	P = np.dot(nb_cost_mat.T, nb_cost_mat)
	#	q_T = -2 * np.dot(dis_k_vec.T, nb_cost_mat)
	#	G = -1 * np.identity(nb_cost_mat.shape[1])
	#	h = np.array([0 for i in range(nb_cost_mat.shape[1])])
	#	A = np.array([1 for i in range(nb_cost_mat.shape[1])])
	#	b = 1
	#	x = cp.Variable(nb_cost_mat.shape[1])
	#	prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q_T@x),
	#					  [G@x <= h])
	#	prob.solve()
	#	edit_costs_new = x.value
	#	residual = prob.value - np.dot(dis_k_vec.T, dis_k_vec)
		
	#	G = -1 * np.identity(nb_cost_mat.shape[1])
	#	h = np.array([0 for i in range(nb_cost_mat.shape[1])])
			x = cp.Variable(nb_cost_mat.shape[1])
			cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
			constraints = [x >= [0.0 for i in range(nb_cost_mat.shape[1])],
		#				   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
						   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
						   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
			prob = cp.Problem(cp.Minimize(cost_fun), constraints)
			self._execute_cvx(prob)
			edit_costs_new = x.value
			residual = np.sqrt(prob.value)
		
		# method 4: 
		
		return edit_costs_new, residual
	
	
	def _execute_cvx(self, prob):
		try:
			prob.solve(verbose=(self._verbose>=2))
		except MemoryError as error0:
			if self._verbose >= 2:
				print('\nUsing solver "OSQP" caused a memory error.')
				print('the original error message is\n', error0)
				print('solver status: ', prob.status)
				print('trying solver "CVXOPT" instead...\n')
			try:
				prob.solve(solver=cp.CVXOPT, verbose=(self._verbose>=2))
			except Exception as error1:
				if self._verbose >= 2:
					print('\nAn error occured when using solver "CVXOPT".')
					print('the original error message is\n', error1)
					print('solver status: ', prob.status)
					print('trying solver "MOSEK" instead. Notice this solver is commercial and a lisence is required.\n')
				prob.solve(solver=cp.MOSEK, verbose=(self._verbose>=2))
			else:
				if self._verbose >= 2:
					print('solver status: ', prob.status)					
		else:
			if self._verbose >= 2:
				print('solver status: ', prob.status)
		if self._verbose >= 2:				
			print()

	
	def _gmg_bcu(self):
		"""
		The local search algorithm based on block coordinate update (BCU) for estimating a generalized median graph (GMG).

		Returns
		-------
		None.

		"""
		# Set up the ged environment.
		ged_env = gedlibpy.GEDEnv() # @todo: maybe create a ged_env as a private varible.
		# gedlibpy.restart_env()
		ged_env.set_edit_cost(self._ged_options['edit_cost'], edit_cost_constant=self._edit_cost_constants)
		graphs = [self._clean_graph(g) for g in self._dataset.graphs]
		for g in graphs:
			ged_env.add_nx_graph(g, '')
		graph_ids = ged_env.get_all_graph_ids()
		set_median_id = ged_env.add_graph('set_median')
		gen_median_id = ged_env.add_graph('gen_median')
		ged_env.init(init_option=self._ged_options['init_option'])
		
		# Set up the madian graph estimator.
		self._mge = MedianGraphEstimator(ged_env, constant_node_costs(self._ged_options['edit_cost']))
		self._mge.set_refine_method(self._ged_options['method'], ged_options_to_string(self._ged_options))
		options = self._mge_options.copy()
		if not 'seed' in options:
			options['seed'] = int(round(time.time() * 1000)) # @todo: may not work correctly for possible parallel usage.
		options['parallel'] = self._parallel
		
		# Select the GED algorithm.
		self._mge.set_options(mge_options_to_string(options))
		self._mge.set_label_names(node_labels=self._dataset.node_labels, 
					  edge_labels=self._dataset.edge_labels, 
					  node_attrs=self._dataset.node_attrs, 
					  edge_attrs=self._dataset.edge_attrs)
		ged_options = self._ged_options.copy()
		if self._parallel:
			ged_options['threads'] = 1
		self._mge.set_init_method(ged_options['method'], ged_options_to_string(ged_options))
		self._mge.set_descent_method(ged_options['method'], ged_options_to_string(ged_options))
		
		# Run the estimator.
		self._mge.run(graph_ids, set_median_id, gen_median_id)
		
		# Get SODs.
		self._sod_set_median = self._mge.get_sum_of_distances('initialized')
		self._sod_gen_median = self._mge.get_sum_of_distances('converged')
		
		# Get median graphs.
		self._set_median = ged_env.get_nx_graph(set_median_id)
		self._gen_median = ged_env.get_nx_graph(gen_median_id)
		
		
	def _compute_distances_to_true_median(self):		
		# compute distance in kernel space for set median.
		kernels_to_sm, _ = self._graph_kernel.compute(self._set_median, self._dataset.graphs, **self._kernel_options)
		kernel_sm, _ = self._graph_kernel.compute(self._set_median, self._set_median, **self._kernel_options)
		if self._kernel_options['normalize']:
			kernels_to_sm = [kernels_to_sm[i] / np.sqrt(self._gram_matrix_unnorm[i, i] * kernel_sm) for i in range(len(kernels_to_sm))] # normalize 
			kernel_sm = 1
		# @todo: not correct kernel value
		gram_with_sm = np.concatenate((np.array([kernels_to_sm]), np.copy(self._graph_kernel.gram_matrix)), axis=0)
		gram_with_sm = np.concatenate((np.array([[kernel_sm] + kernels_to_sm]).T, gram_with_sm), axis=1)
		self._k_dis_set_median = compute_k_dis(0, range(1, 1+len(self._dataset.graphs)), 
										  [1 / len(self._dataset.graphs)] * len(self._dataset.graphs),
										  gram_with_sm, withterm3=False)
		
		# compute distance in kernel space for generalized median.
		kernels_to_gm, _ = self._graph_kernel.compute(self._gen_median, self._dataset.graphs, **self._kernel_options)
		kernel_gm, _ = self._graph_kernel.compute(self._gen_median, self._gen_median, **self._kernel_options)
		if self._kernel_options['normalize']:
			kernels_to_gm = [kernels_to_gm[i] / np.sqrt(self._gram_matrix_unnorm[i, i] * kernel_gm) for i in range(len(kernels_to_gm))] # normalize
			kernel_gm = 1
		gram_with_gm = np.concatenate((np.array([kernels_to_gm]), np.copy(self._graph_kernel.gram_matrix)), axis=0)
		gram_with_gm = np.concatenate((np.array([[kernel_gm] + kernels_to_gm]).T, gram_with_gm), axis=1)
		self._k_dis_gen_median = compute_k_dis(0, range(1, 1+len(self._dataset.graphs)), 
										  [1 / len(self._dataset.graphs)] * len(self._dataset.graphs),
										  gram_with_gm, withterm3=False)
				
		# compute distance in kernel space for each graph in median set.
		k_dis_median_set = []
		for idx in range(len(self._dataset.graphs)):
			k_dis_median_set.append(compute_k_dis(idx+1, range(1, 1+len(self._dataset.graphs)), 
								 [1 / len(self._dataset.graphs)] * len(self._dataset.graphs), 
								 gram_with_gm, withterm3=False))
		idx_k_dis_median_set_min = np.argmin(k_dis_median_set)
		self._k_dis_dataset = k_dis_median_set[idx_k_dis_median_set_min]
		self._best_from_dataset = self._dataset.graphs[idx_k_dis_median_set_min].copy()
			
		if self._verbose >= 2:
			print()
			print('distance in kernel space for set median:', self._k_dis_set_median)
			print('distance in kernel space for generalized median:', self._k_dis_gen_median)
			print('minimum distance in kernel space for each graph in median set:', self._k_dis_dataset)
			print('distance in kernel space for each graph in median set:', k_dis_median_set)	
			
			
# 	def _clean_graph(self, G, node_labels=[], edge_labels=[], node_attrs=[], edge_attrs=[]):
	def _clean_graph(self, G): # @todo: this may not be needed when datafile is updated.
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
		return self._mge
	
	@property
	def ged_options(self):
		return self._ged_options

	@ged_options.setter
	def ged_options(self, value):
		self._ged_options = value		

	
	@property
	def mge_options(self):
		return self._mge_options

	@mge_options.setter
	def mge_options(self, value):
		self._mge_options = value		


	@property
	def fit_method(self):
		return self._fit_method

	@fit_method.setter
	def fit_method(self, value):
		self._fit_method = value
		
		
	@property
	def init_ecc(self):
		return self._init_ecc

	@init_ecc.setter
	def init_ecc(self, value):
		self._init_ecc = value
		
	
	@property
	def set_median(self):
		return self._set_median


	@property
	def gen_median(self):
		return self._gen_median
	
	
	@property
	def best_from_dataset(self):
		return self._best_from_dataset
	
	
	@property
	def gram_matrix_unnorm(self):
		return self._gram_matrix_unnorm
	
	@gram_matrix_unnorm.setter
	def gram_matrix_unnorm(self, value):
		self._gram_matrix_unnorm = value