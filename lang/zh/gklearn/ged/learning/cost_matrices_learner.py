#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:42:48 2020

@author: ljia
"""
import numpy as np
import cvxpy as cp
import time
from gklearn.ged.learning.costs_learner import CostsLearner
from gklearn.ged.util import compute_geds_cml


class CostMatricesLearner(CostsLearner):
	
	
	def __init__(self, edit_cost='CONSTANT', triangle_rule=False, allow_zeros=True, parallel=False, verbose=2):
		super().__init__(parallel, verbose)
		self._edit_cost = edit_cost
		self._triangle_rule = triangle_rule
		self._allow_zeros = allow_zeros
	
	
	def fit(self, X, y):
		if self._edit_cost == 'LETTER':
			raise Exception('Cannot compute for cost "LETTER".')
		elif self._edit_cost == 'LETTER2':
			raise Exception('Cannot compute for cost "LETTER2".')
		elif self._edit_cost == 'NON_SYMBOLIC':
			raise Exception('Cannot compute for cost "NON_SYMBOLIC".')
		elif self._edit_cost == 'CONSTANT': # @todo: node/edge may not labeled.
			if not self._triangle_rule and self._allow_zeros:
				w = cp.Variable(X.shape[1])
				cost_fun = cp.sum_squares(X @ w - y)
				constraints = [w >= [0.0 for i in range(X.shape[1])]]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self.execute_cvx(prob)
				edit_costs_new = w.value
				residual = np.sqrt(prob.value)
			elif self._triangle_rule and self._allow_zeros: # @todo
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
			elif not self._triangle_rule and not self._allow_zeros: # @todo
				x = cp.Variable(nb_cost_mat.shape[1])
				cost_fun = cp.sum_squares(nb_cost_mat @ x - dis_k_vec)
				constraints = [x >= [0.01 for i in range(nb_cost_mat.shape[1])]]
				prob = cp.Problem(cp.Minimize(cost_fun), constraints)
				self._execute_cvx(prob)
				edit_costs_new = x.value
				residual = np.sqrt(prob.value)
			elif self._triangle_rule and not self._allow_zeros: # @todo
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
		
		self._cost_list.append(edit_costs_new)
		
		
	def init_geds_and_nb_eo(self, y, graphs):
		time0 = time.time()
		self._cost_list.append(np.concatenate((self._ged_options['node_label_costs'], 
										 self._ged_options['edge_label_costs'])))
		ged_vec, self._nb_eo = self.compute_geds_and_nb_eo(graphs)
		self._residual_list.append(np.sqrt(np.sum(np.square(np.array(ged_vec) - y))))
		self._runtime_list.append(time.time() - time0)

		if self._verbose >= 2:
			print('Current node label costs:', self._cost_list[-1][0:len(self._ged_options['node_label_costs'])])
			print('Current edge label costs:', self._cost_list[-1][len(self._ged_options['node_label_costs']):])
			print('Residual list:', self._residual_list) 
	
	
	def update_geds_and_nb_eo(self, y, graphs, time0):
		self._ged_options['node_label_costs'] = self._cost_list[-1][0:len(self._ged_options['node_label_costs'])]
		self._ged_options['edge_label_costs'] = self._cost_list[-1][len(self._ged_options['node_label_costs']):]
		ged_vec, self._nb_eo = self.compute_geds_and_nb_eo(graphs)
		self._residual_list.append(np.sqrt(np.sum(np.square(np.array(ged_vec) - y))))
		self._runtime_list.append(time.time() - time0)
		
	
	def compute_geds_and_nb_eo(self, graphs):
		ged_vec, ged_mat, n_edit_operations = compute_geds_cml(graphs, options=self._ged_options, parallel=self._parallel, verbose=(self._verbose > 1))
		return ged_vec, np.array(n_edit_operations)
	
	
	def check_convergency(self):
		self._ec_changed = False
		for i, cost in enumerate(self._cost_list[-1]):
			if cost == 0:
				if self._cost_list[-2][i] > self._epsilon_ec:
					self._ec_changed = True
					break
			elif abs(cost - self._cost_list[-2][i]) / cost > self._epsilon_ec:
				self._ec_changed = True
				break
# 				if abs(cost - edit_cost_list[-2][i]) > self._epsilon_ec:
#  					ec_changed = True
#  					break
		self._residual_changed = False
		if self._residual_list[-1] == 0:
			if self._residual_list[-2] > self._epsilon_residual:
				self._residual_changed = True
		elif abs(self._residual_list[-1] - self._residual_list[-2]) / self._residual_list[-1] > self._epsilon_residual:
			self._residual_changed = True
		self._converged = not (self._ec_changed or self._residual_changed)
		if self._converged:
			self._itrs_without_update += 1
		else:
			self._itrs_without_update = 0
			self._num_updates_ecs += 1
	
	
	def print_current_states(self):
		print()
		print('-------------------------------------------------------------------------')
		print('States of iteration', self._itrs + 1)
		print('-------------------------------------------------------------------------')
# 				print('Time spend:', self._runtime_optimize_ec)
		print('Total number of iterations for optimizing:', self._itrs + 1)
		print('Total number of updating edit costs:', self._num_updates_ecs)
		print('Was optimization of edit costs converged:', self._converged)
		print('Did edit costs change:', self._ec_changed)
		print('Did residual change:', self._residual_changed)
		print('Iterations without update:', self._itrs_without_update)
		print('Current node label costs:', self._cost_list[-1][0:len(self._ged_options['node_label_costs'])])
		print('Current edge label costs:', self._cost_list[-1][len(self._ged_options['node_label_costs']):])
		print('Residual list:', self._residual_list)
		print('-------------------------------------------------------------------------')