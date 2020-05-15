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
# from gklearn.utils.dataset import Dataset

class MedianPreimageGenerator(PreimageGenerator):
	
	def __init__(self, dataset=None):
		PreimageGenerator.__init__(self, dataset=dataset)
		# arguments to set.
		self.__mge = None
		self.__ged_options = {}
		self.__mge_options = {}
		self.__fit_method = 'k-graphs'
		self.__init_ecc = None
		self.__parallel = True
		self.__n_jobs = multiprocessing.cpu_count()
		self.__ds_name = None
		self.__time_limit_in_sec = 0
		self.__max_itrs = 100
		self.__max_itrs_without_update = 3
		self.__epsilon_residual = 0.01
		self.__epsilon_ec = 0.1
		self.__allow_zeros = False
		self.__triangle_rule = True
		# values to compute.
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
		self.__itrs = 0
		self.__converged = False
		self.__num_updates_ecc = 0
		# values that can be set or to be computed.
		self.__edit_cost_constants = []
		self.__gram_matrix_unnorm = None
		self.__runtime_precompute_gm = None

		
	def set_options(self, **kwargs):
		self._kernel_options = kwargs.get('kernel_options', {})
		self._graph_kernel = kwargs.get('graph_kernel', None)
		self._verbose = kwargs.get('verbose', 2)
		self.__ged_options = kwargs.get('ged_options', {})
		self.__mge_options = kwargs.get('mge_options', {})
		self.__fit_method = kwargs.get('fit_method', 'k-graphs')
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
		self.__allow_zeros = kwargs.get('allow_zeros', False)
		self.__triangle_rule = kwargs.get('triangle_rule', True)
		
		
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
			
		if self.__fit_method != 'k-graphs' and self.__fit_method != 'whole-dataset':
			start = time.time()
			self.__runtime_precompute_gm = 0
			end_precompute_gm = start
		
		# 2. optimize edit cost constants. 
		self.__optimize_edit_cost_constants()
		end_optimize_ec = time.time()
		self.__runtime_optimize_ec = end_optimize_ec - end_precompute_gm
		
		# 3. compute set median and gen median using optimized edit costs.
		if self._verbose >= 2:
			print('\nstart computing set median and gen median using optimized edit costs...\n')
# 		group_fnames = [Gn[g].graph['filename'] for g in group_min]
		self.__generate_preimage_iam()
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
# 		Gn_median = [Gn[g].copy() for g in group_min]
		self.__compute_distances_to_true_median()
# 		dis_k_sm, dis_k_gm, dis_k_gi, dis_k_gi_min, idx_dis_k_gi_min = 
# 		idx_dis_k_gi_min = group_min[idx_dis_k_gi_min]
# 		print('index min dis_k_gi:', idx_dis_k_gi_min)
# 		print('sod_sm:', sod_sm)
# 		print('sod_gm:', sod_gm)

		# 5. print out results.
		if self._verbose:
			print()
			print('================================================================================')
			print('Finished generalization of preimages.')
			print('--------------------------------------------------------------------------------')
			print('The optimized edit cost constants:', self.__edit_cost_constants)
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
			print('Total number of updating edit costs:', self.__num_updates_ecc)
			print('Is optimization of edit costs converged:', self.__converged)
			print('================================================================================')
			print()
			
	# collect return values.
# 	return (sod_sm, sod_gm), \
# 		   (dis_k_sm, dis_k_gm, dis_k_gi, dis_k_gi_min, idx_dis_k_gi_min), \
# 		   (time_fitting, time_generating)


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
		results['num_updates_ecc'] = self.__num_updates_ecc
		results['mge'] = {}
		results['mge']['num_decrease_order'] = self.__mge.get_num_times_order_decreased()
		results['mge']['num_increase_order'] = self.__mge.get_num_times_order_increased()
		results['mge']['num_converged_descents'] = self.__mge.get_num_converged_descents()
		return results

		
	def __optimize_edit_cost_constants(self):
		"""fit edit cost constants.	
		"""
		if self.__fit_method == 'random': # random
			if self.__ged_options['edit_cost'] == 'LETTER':
				self.__edit_cost_constants = random.sample(range(1, 10), 3)
				self.__edit_cost_constants = [item * 0.1 for item in self.__edit_cost_constants]
			elif self.__ged_options['edit_cost'] == 'LETTER2':
				random.seed(time.time())
				self.__edit_cost_constants = random.sample(range(1, 10), 5)
	#			self.__edit_cost_constants = [item * 0.1 for item in self.__edit_cost_constants]
			elif self.__ged_options['edit_cost'] == 'NON_SYMBOLIC':
				self.__edit_cost_constants = random.sample(range(1, 10), 6)
				if self._dataset.node_attrs == []:
					self.__edit_cost_constants[2] = 0
				if self._dataset.edge_attrs == []:
					self.__edit_cost_constants[5] = 0
			else:
				self.__edit_cost_constants = random.sample(range(1, 10), 6)
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
			self.__optimize_ecc_by_kernel_distances()
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
		
		
	def __optimize_ecc_by_kernel_distances(self):		
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
		graphs = [self.__clean_graph(g) for g in self._dataset.graphs]
		self.__edit_cost_constants = self.__init_ecc
		options = self.__ged_options.copy()
		options['edit_cost_constants'] = self.__edit_cost_constants # @todo
		options['node_labels'] = self._dataset.node_labels
		options['edge_labels'] = self._dataset.edge_labels
		options['node_attrs'] = self._dataset.node_attrs
		options['edge_attrs'] = self._dataset.edge_attrs
		ged_vec_init, ged_mat, n_edit_operations = compute_geds(graphs, options=options, parallel=self.__parallel, verbose=(self._verbose > 1))
		residual_list = [np.sqrt(np.sum(np.square(np.array(ged_vec_init) - dis_k_vec)))]	
		time_list = [time.time() - time0]
		edit_cost_list = [self.__init_ecc]  
		nb_cost_mat = np.array(n_edit_operations)
		nb_cost_mat_list = [nb_cost_mat]
		if self._verbose >= 2:
			print('Current edit cost constants:', self.__edit_cost_constants)
			print('Residual list:', residual_list)
		
		# run iteration from initial edit costs.
		self.__converged = False
		itrs_without_update = 0
		self.__itrs = 0
		self.__num_updates_ecc = 0
		timer = Timer(self.__time_limit_in_sec)
		while not self.__termination_criterion_met(self.__converged, timer, self.__itrs, itrs_without_update):
			if self._verbose >= 2:
				print('\niteration', self.__itrs + 1)
			time0 = time.time()
			# "fit" geds to distances in feature space by tuning edit costs using theLeast Squares Method.
# 			np.savez('results/xp_fit_method/fit_data_debug' + str(self.__itrs) + '.gm', 
# 					 nb_cost_mat=nb_cost_mat, dis_k_vec=dis_k_vec, 
# 					 n_edit_operations=n_edit_operations, ged_vec_init=ged_vec_init,
# 					 ged_mat=ged_mat)
			self.__edit_cost_constants, _ = self.__update_ecc(nb_cost_mat, dis_k_vec)
			for i in range(len(self.__edit_cost_constants)):
				if -1e-9 <= self.__edit_cost_constants[i] <= 1e-9:
					self.__edit_cost_constants[i] = 0
				if self.__edit_cost_constants[i] < 0:
					raise ValueError('The edit cost is negative.')
	#		for i in range(len(self.__edit_cost_constants)):
	#			if self.__edit_cost_constants[i] < 0:
	#				self.__edit_cost_constants[i] = 0
	
			# compute new GEDs and numbers of edit operations.
			options = self.__ged_options.copy() # np.array([self.__edit_cost_constants[0], self.__edit_cost_constants[1], 0.75])
			options['edit_cost_constants'] = self.__edit_cost_constants # @todo
			options['node_labels'] = self._dataset.node_labels
			options['edge_labels'] = self._dataset.edge_labels
			options['node_attrs'] = self._dataset.node_attrs
			options['edge_attrs'] = self._dataset.edge_attrs
			ged_vec, ged_mat, n_edit_operations = compute_geds(graphs, options=options, parallel=self.__parallel, verbose=(self._verbose > 1))
			residual_list.append(np.sqrt(np.sum(np.square(np.array(ged_vec) - dis_k_vec))))
			time_list.append(time.time() - time0)
			edit_cost_list.append(self.__edit_cost_constants)
			nb_cost_mat = np.array(n_edit_operations)
			nb_cost_mat_list.append(nb_cost_mat)	
				
			# check convergency.
			ec_changed = False
			for i, cost in enumerate(self.__edit_cost_constants):
				if cost == 0:
 					if edit_cost_list[-2][i] > self.__epsilon_ec:
						 ec_changed = True
						 break
				elif abs(cost - edit_cost_list[-2][i]) / cost > self.__epsilon_ec:
 					ec_changed = True
 					break
# 				if abs(cost - edit_cost_list[-2][i]) > self.__epsilon_ec:
#  					ec_changed = True
#  					break
			residual_changed = False
			if residual_list[-1] == 0:
				if residual_list[-2] > self.__epsilon_residual:
					residual_changed = True
			elif abs(residual_list[-1] - residual_list[-2]) / residual_list[-1] > self.__epsilon_residual:
				residual_changed = True
			self.__converged = not (ec_changed or residual_changed)
			if self.__converged:
				itrs_without_update += 1
			else:
				itrs_without_update = 0
				self.__num_updates_ecc += 1
				
			# print current states.
			if self._verbose >= 2:
				print()
				print('-------------------------------------------------------------------------')
				print('States of iteration', self.__itrs + 1)
				print('-------------------------------------------------------------------------')
# 				print('Time spend:', self.__runtime_optimize_ec)
				print('Total number of iterations for optimizing:', self.__itrs + 1)
				print('Total number of updating edit costs:', self.__num_updates_ecc)
				print('Was optimization of edit costs converged:', self.__converged)
				print('Did edit costs change:', ec_changed)
				print('Did residual change:', residual_changed)
				print('Iterations without update:', itrs_without_update)
				print('Current edit cost constants:', self.__edit_cost_constants)
				print('Residual list:', residual_list)
				print('-------------------------------------------------------------------------')
			
			self.__itrs += 1


	def __termination_criterion_met(self, converged, timer, itr, itrs_without_update):
		if timer.expired() or (itr >= self.__max_itrs if self.__max_itrs >= 0 else False):
# 			if self.__state == AlgorithmState.TERMINATED:
# 				self.__state = AlgorithmState.INITIALIZED
			return True
		return converged or (itrs_without_update > self.__max_itrs_without_update if self.__max_itrs_without_update >= 0 else False)


	def __update_ecc(self, nb_cost_mat, dis_k_vec, rw_constraints='inequality'):
	#	if self.__ds_name == 'Letter-high':
		if self.__ged_options['edit_cost'] == 'LETTER':
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
		elif self.__ged_options['edit_cost'] == 'LETTER2':
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
			if not self.__triangle_rule and self.__allow_zeros:
				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
				constraints = [x >= [0.0 for i in range(nb_cost_mat_new.shape[1])],
							   np.array([1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 1.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 0.0, 1.0]).T@x >= 0.01]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self.__execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif self.__triangle_rule and self.__allow_zeros:
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
				self.__execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif not self.__triangle_rule and not self.__allow_zeros:
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
			elif self.__triangle_rule and not self.__allow_zeros:
				# c_vs <= c_vi + c_vr.
				nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
				x = cp.Variable(nb_cost_mat_new.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
							   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self.__execute_cvx(prob)
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

		elif self.__ged_options['edit_cost'] == 'NON_SYMBOLIC':
			is_n_attr = np.count_nonzero(nb_cost_mat[:,2])
			is_e_attr = np.count_nonzero(nb_cost_mat[:,5])
			
			if self.__ds_name == 'SYNTHETICnew': # @todo: rearrenge this later.
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
				
			elif not self.__triangle_rule and self.__allow_zeros:
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
					self.__execute_cvx(prob)
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
					self.__execute_cvx(prob)
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
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)
			elif self.__triangle_rule and self.__allow_zeros:
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
					self.__execute_cvx(prob)
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
					self.__execute_cvx(prob)
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
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)
			elif not self.__triangle_rule and not self.__allow_zeros:
				if is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = x.value
					residual = np.sqrt(prob.value)
				elif is_n_attr and not is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value, np.array([0.0])))
					residual = np.sqrt(prob.value)
				elif not is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)
			elif self.__triangle_rule and not self.__allow_zeros:
				# c_vs <= c_vi + c_vr.
				if is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
								   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = x.value
					residual = np.sqrt(prob.value)
				elif is_n_attr and not is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,2,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([1.0, 1.0, -1.0, 0.0, 0.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value, np.array([0.0])))
					residual = np.sqrt(prob.value)
				elif not is_n_attr and is_e_attr:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4,5]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])],
								   np.array([0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), x.value[2:]))
					residual = np.sqrt(prob.value)
				else:
					nb_cost_mat_new = nb_cost_mat[:,[0,1,3,4]]
					x = cp.Variable(nb_cost_mat_new.shape[1])
					cost_fun = cp.sum_squares(nb_cost_mat_new @ x - dis_k_vec)
					constraints = [x >= [0.01 for i in range(nb_cost_mat_new.shape[1])]]
					prob = cp.Problem(cp.Minimize(cost_fun), constraints)
					self.__execute_cvx(prob)
					edit_costs_new = np.concatenate((x.value[0:2], np.array([0.0]), 
													 x.value[2:], np.array([0.0])))
					residual = np.sqrt(prob.value)

		elif self.__ged_options['edit_cost'] == 'CONSTANT': # @todo: node/edge may not labeled.
			if not self.__triangle_rule and self.__allow_zeros:
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.0 for i in range(nb_cost_mat.shape[1])],
							   np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).T@x >= 0.01,
							   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).T@x >= 0.01]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self.__execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif self.__triangle_rule and self.__allow_zeros:
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
				self.__execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif not self.__triangle_rule and not self.__allow_zeros:
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat.shape[1])]]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self.__execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif self.__triangle_rule and not self.__allow_zeros:
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat.shape[1])],
							   np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T@x >= 0.0,
							   np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T@x >= 0.0]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self.__execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
		else:
			raise Exception('The edit cost "', self.__ged_options['edit_cost'], '" is not supported for update progress.')
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
			self.__execute_cvx(prob)
			edit_costs_new = x.value
			residual = np.sqrt(prob.value)
		
		# method 4: 
		
		return edit_costs_new, residual
	
	
	def __execute_cvx(self, prob):
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

	
	def __generate_preimage_iam(self):
		# Set up the ged environment.
		ged_env = gedlibpy.GEDEnv() # @todo: maybe create a ged_env as a private varible.
		# gedlibpy.restart_env()
		ged_env.set_edit_cost(self.__ged_options['edit_cost'], edit_cost_constant=self.__edit_cost_constants)
		graphs = [self.__clean_graph(g) for g in self._dataset.graphs]
		for g in graphs:
			ged_env.add_nx_graph(g, '')
		graph_ids = ged_env.get_all_graph_ids()
		set_median_id = ged_env.add_graph('set_median')
		gen_median_id = ged_env.add_graph('gen_median')
		ged_env.init(init_option=self.__ged_options['init_option'])
		
		# Set up the madian graph estimator.
		self.__mge = MedianGraphEstimator(ged_env, constant_node_costs(self.__ged_options['edit_cost']))
		self.__mge.set_refine_method(self.__ged_options['method'], ged_options_to_string(self.__ged_options))
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
		self.__mge.set_init_method(ged_options['method'], ged_options_to_string(ged_options))
		self.__mge.set_descent_method(ged_options['method'], ged_options_to_string(ged_options))
		
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
		kernels_to_sm = [kernels_to_sm[i] / np.sqrt(self.__gram_matrix_unnorm[i, i] * kernel_sm) for i in range(len(kernels_to_sm))] # normalize 
		# @todo: not correct kernel value
		gram_with_sm = np.concatenate((np.array([kernels_to_sm]), np.copy(self._graph_kernel.gram_matrix)), axis=0)
		gram_with_sm = np.concatenate((np.array([[1] + kernels_to_sm]).T, gram_with_sm), axis=1)
		self.__k_dis_set_median = compute_k_dis(0, range(1, 1+len(self._dataset.graphs)), 
										  [1 / len(self._dataset.graphs)] * len(self._dataset.graphs),
										  gram_with_sm, withterm3=False)
	#	print(gen_median.nodes(data=True))
	#	print(gen_median.edges(data=True))
	#	print(set_median.nodes(data=True))
	#	print(set_median.edges(data=True))
		
		# compute distance in kernel space for generalized median.
		kernels_to_gm, _ = self._graph_kernel.compute(self.__gen_median, self._dataset.graphs, **self._kernel_options)
		kernel_gm, _ = self._graph_kernel.compute(self.__gen_median, self.__gen_median, **self._kernel_options)
		kernels_to_gm = [kernels_to_gm[i] / np.sqrt(self.__gram_matrix_unnorm[i, i] * kernel_gm) for i in range(len(kernels_to_gm))] # normalize
		gram_with_gm = np.concatenate((np.array([kernels_to_gm]), np.copy(self._graph_kernel.gram_matrix)), axis=0)
		gram_with_gm = np.concatenate((np.array([[1] + kernels_to_gm]).T, gram_with_gm), axis=1)
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