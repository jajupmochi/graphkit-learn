#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:30:31 2020

@author: ljia
"""
import numpy as np
import cvxpy as cp
import time
from gklearn.utils import Timer


class CostsLearner(object):
	
	
	def __init__(self, parallel, verbose):
		### To set.
		self._parallel = parallel
		self._verbose = verbose
		# For update().
		self._time_limit_in_sec = 0
		self._max_itrs = 100
		self._max_itrs_without_update = 3
		self._epsilon_residual = 0.01
		self._epsilon_ec = 0.1
		### To compute.
		self._residual_list = []
		self._runtime_list = []
		self._cost_list = []
		self._nb_eo = None
		# For update().
		self._itrs = 0
		self._converged = False
		self._num_updates_ecs = 0
		self._ec_changed = None
		self._residual_changed = None
		self._itrs_without_update = 0
		### Both set and get.
		self._ged_options = None


	def fit(self, X, y):
		pass
	
	
	def preprocess(self):
		pass # @todo: remove the zero numbers of edit costs.
	
	
	def postprocess(self):
		for i in range(len(self._cost_list[-1])):
			if -1e-9 <= self._cost_list[-1][i] <= 1e-9:
				self._cost_list[-1][i] = 0
			if self._cost_list[-1][i] < 0:
				raise ValueError('The edit cost is negative.')
	
	
	def set_update_params(self, **kwargs):
		self._time_limit_in_sec = kwargs.get('time_limit_in_sec', self._time_limit_in_sec)
		self._max_itrs = kwargs.get('max_itrs', self._max_itrs)
		self._max_itrs_without_update = kwargs.get('max_itrs_without_update', self._max_itrs_without_update)
		self._epsilon_residual = kwargs.get('epsilon_residual', self._epsilon_residual)
		self._epsilon_ec = kwargs.get('epsilon_ec', self._epsilon_ec)
		

	def update(self, y, graphs, ged_options, **kwargs):
		# Set parameters.
		self._ged_options = ged_options
		if kwargs != {}:
			self.set_update_params(**kwargs)
		
		# The initial iteration.
		if self._verbose >= 2:
			print('\ninitial:')
		self.init_geds_and_nb_eo(y, graphs)

		self._converged = False
		self._itrs_without_update = 0
		self._itrs = 0
		self._num_updates_ecs = 0
		timer = Timer(self._time_limit_in_sec)
		# Run iterations from initial edit costs.
		while not self.termination_criterion_met(self._converged, timer, self._itrs, self._itrs_without_update):
			if self._verbose >= 2:
				print('\niteration', self._itrs + 1)
			time0 = time.time()
				
			# Fit GED space to the target space.
			self.preprocess()
			self.fit(self._nb_eo, y)
			self.postprocess()
			
			# Compute new GEDs and numbers of edit operations.
			self.update_geds_and_nb_eo(y, graphs, time0)
			
			# Check convergency.
			self.check_convergency()
			
			# Print current states.
			if self._verbose >= 2:
				self.print_current_states()
				
			self._itrs += 1
			
			
	def init_geds_and_nb_eo(self, y, graphs):
		pass
	
	
	def update_geds_and_nb_eo(self, y, graphs, time0):
		pass
			
			
	def compute_geds_and_nb_eo(self, graphs):
		pass
	
	
	def check_convergency(self):
		pass
	
	
	def print_current_states(self):
		pass


	def termination_criterion_met(self, converged, timer, itr, itrs_without_update):
		if timer.expired() or (itr >= self._max_itrs if self._max_itrs >= 0 else False):
# 			if self._state == AlgorithmState.TERMINATED:
# 				self._state = AlgorithmState.INITIALIZED
			return True
		return converged or (itrs_without_update > self._max_itrs_without_update if self._max_itrs_without_update >= 0 else False)
	
	
	def execute_cvx(self, prob):
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
			
			
	def get_results(self):
		results = {}
		results['residual_list'] = self._residual_list
		results['runtime_list'] = self._runtime_list
		results['cost_list'] = self._cost_list
		results['nb_eo'] = self._nb_eo
		results['itrs'] = self._itrs
		results['converged'] = self._converged
		results['num_updates_ecs'] = self._num_updates_ecs
		results['ec_changed'] = self._ec_changed
		results['residual_changed'] = self._residual_changed
		results['itrs_without_update'] = self._itrs_without_update
		return results