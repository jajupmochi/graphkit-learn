#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:52:35 2020

@author: ljia
"""
import numpy as np
import time
import networkx as nx


class GEDMethod(object):
	
	
	def __init__(self, ged_data):
		self._initialized = False
		self._ged_data = ged_data
		self._options = None
		self._lower_bound = 0
		self._upper_bound = np.inf
		self._node_map = [0, 0] # @todo
		self._runtime = None
		self._init_time = None
		
	
	def init(self):
		"""Initializes the method with options specified by set_options().
		"""
		start = time.time()
		self._ged_init()
		end = time.time()
		self._init_time = end - start
		self._initialized = True
		
		
	def set_options(self, options):
		"""
	/*!
	 * @brief Sets the options of the method.
	 * @param[in] options String of the form <tt>[--@<option@> @<arg@>] [...]</tt>, where @p option contains neither spaces nor single quotes,
	 * and @p arg contains neither spaces nor single quotes or is of the form <tt>'[--@<sub-option@> @<sub-arg@>] [...]'</tt>,
	 * where both @p sub-option and @p sub-arg contain neither spaces nor single quotes.
	 */
		"""
		self._ged_set_default_options()
		for key, val in options.items():
			if not self._ged_parse_option(key, val):
				raise Exception('Invalid option "', key, '". Usage: options = "' + self._ged_valid_options_string() + '".') # @todo: not implemented.
		self._initialized = False
		
		
	def run(self, g_id, h_id):
		"""
	/*!
	 * @brief Runs the method with options specified by set_options().
	 * @param[in] g_id ID of input graph.
	 * @param[in] h_id ID of input graph.
	 */
		"""
		start = time.time()
		result = self.run_as_util(self._ged_data._graphs[g_id], self._ged_data._graphs[h_id])
		end = time.time()
		self._lower_bound = result['lower_bound']
		self._upper_bound = result['upper_bound']
		if len(result['node_maps']) > 0:
			self._node_map = result['node_maps'][0]
		self._runtime = end - start
		
		
	def run_as_util(self, g, h):
		"""
	/*!
	 * @brief Runs the method with options specified by set_options().
	 * @param[in] g Input graph.
	 * @param[in] h Input graph.
	 * @param[out] result Result variable.
	 */
		"""
		# Compute optimal solution and return if at least one of the two graphs is empty.
		if nx.number_of_nodes(g) == 0 or nx.number_of_nodes(h) == 0:
			print('This is not implemented.')
			pass # @todo:
			
		# Run the method.
		return self._ged_run(g, h)
		
		
	def get_upper_bound(self):
		"""
	/*!
	 * @brief Returns an upper bound.
	 * @return Upper bound for graph edit distance provided by last call to run() or -1 if the method does not yield an upper bound.
	 */
		"""
		return self._upper_bound
		
		
	def get_lower_bound(self):
		"""
	/*!
	 * @brief Returns a lower bound.
	 * @return Lower bound for graph edit distance provided by last call to run() or -1 if the method does not yield a lower bound.
	 */
		"""
		return self._lower_bound
		
		
	def get_runtime(self):
		"""
	/*!
	 * @brief Returns the runtime.
	 * @return Runtime of last call to run() in seconds.
	 */
		"""
		return self._runtime
	

	def get_init_time(self):
		"""
	/*!
	 * @brief Returns the initialization time.
	 * @return Runtime of last call to init() in seconds.
	 */
		"""		
		return self._init_time


	def get_node_map(self):
		"""
	/*!
	 * @brief Returns a graph matching.
	 * @return Constant reference to graph matching provided by last call to run() or to an empty matching if the method does not yield a matching.
	 */
		"""
		return self._node_map
		
		
	def _ged_init(self):
		"""
	/*!
	 * @brief Initializes the method.
	 * @note Must be overridden by derived classes that require initialization.
	 */
		"""
		pass
			
			
	def _ged_parse_option(self, option, arg):
		"""
	/*!
	 * @brief Parses one option.
	 * @param[in] option The name of the option.
	 * @param[in] arg The argument of the option.
	 * @return Boolean @p true if @p option is a valid option name for the method and @p false otherwise.
	 * @note Must be overridden by derived classes that have options.
	 */
		"""
		return False
	
	
	def _ged_run(self, g, h):
		"""
	/*!
	 * @brief Runs the method with options specified by set_options().
	 * @param[in] g Input graph.
	 * @param[in] h Input graph.
	 * @param[out] result Result variable.
	 * @note Must be overridden by derived classes.
	 */
		"""
		return {}
		
	
	
	def _ged_valid_options_string(self):
		"""
	/*!
	 * @brief Returns string of all valid options.
	 * @return String of the form <tt>[--@<option@> @<arg@>] [...]</tt>.
	 * @note Must be overridden by derived classes that have options.
	 */
		"""
		return ''
		
		
	def _ged_set_default_options(self):
		"""
	/*!
	 * @brief Sets all options to default values.
	 * @note Must be overridden by derived classes that have options.
	 */
		"""
		pass
		