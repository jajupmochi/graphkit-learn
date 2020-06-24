#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:01:24 2020

@author: ljia
"""
import numpy as np
import networkx as nx
from gklearn.ged.methods import GEDMethod
from gklearn.ged.util import LSAPESolver, misc
from gklearn.ged.env import NodeMap
	

class LSAPEBasedMethod(GEDMethod):
	
	
	def __init__(self, ged_data):
		super().__init__(ged_data)
		self._lsape_model = None # @todo: LSAPESolver::ECBP
		self._greedy_method = None # @todo: LSAPESolver::BASIC
		self._compute_lower_bound = True
		self._solve_optimally = True
		self._num_threads = 1
		self._centrality_method = 'NODE' # @todo
		self._centrality_weight = 0.7
		self._centralities = {}
		self._max_num_solutions = 1
		
		
	def populate_instance_and_run_as_util(self, g, h): #, lsape_instance):
		"""
	/*!
	 * @brief Runs the method with options specified by set_options() and provides access to constructed LSAPE instance.
	 * @param[in] g Input graph.
	 * @param[in] h Input graph.
	 * @param[out] result Result variable.
	 * @param[out] lsape_instance LSAPE instance.
	 */
		"""
		result = {'node_maps': [], 'lower_bound': 0, 'upper_bound': np.inf}
		
		# Populate the LSAPE instance and set up the solver.
		nb1, nb2 = nx.number_of_nodes(g), nx.number_of_nodes(h)
		lsape_instance = np.ones((nb1 + nb2, nb1 + nb2)) * np.inf
# 		lsape_instance = np.empty((nx.number_of_nodes(g) + 1, nx.number_of_nodes(h) + 1))
		self.populate_instance(g, h, lsape_instance)
		
# 		nb1, nb2 = nx.number_of_nodes(g), nx.number_of_nodes(h)
# 		lsape_instance_new = np.empty((nb1 + nb2, nb1 + nb2)) * np.inf
# 		lsape_instance_new[nb1:, nb2:] = 0
# 		lsape_instance_new[0:nb1, 0:nb2] = lsape_instance[0:nb1, 0:nb2]
# 		for i in range(nb1): # all u's neighbor
# 			lsape_instance_new[i, nb2 + i] = lsape_instance[i, nb2]
# 		for i in range(nb2): # all u's neighbor
# 			lsape_instance_new[nb1 + i, i] = lsape_instance[nb2, i]
# 		lsape_solver = LSAPESolver(lsape_instance_new)
		
		lsape_solver = LSAPESolver(lsape_instance)
		
		# Solve the LSAPE instance.
		if self._solve_optimally:
			lsape_solver.set_model(self._lsape_model)
		else:
			lsape_solver.set_greedy_method(self._greedy_method)
		lsape_solver.solve(self._max_num_solutions)
		
		# Compute and store lower and upper bound.
		if self._compute_lower_bound and self._solve_optimally:
			result['lower_bound'] = lsape_solver.minimal_cost() * self._lsape_lower_bound_scaling_factor(g, h) # @todo: test
		
		for solution_id in range(0, lsape_solver.num_solutions()):
			result['node_maps'].append(NodeMap(nx.number_of_nodes(g), nx.number_of_nodes(h)))
			misc.construct_node_map_from_solver(lsape_solver, result['node_maps'][-1], solution_id)
			self._ged_data.compute_induced_cost(g, h, result['node_maps'][-1])
			
		# Add centralities and reoptimize.
		if self._centrality_weight > 0 and self._centrality_method != 'NODE':
			print('This is not implemented.')
			pass # @todo
			
		# Sort the node maps and set the upper bound.
		if len(result['node_maps']) > 1 or len(result['node_maps']) > self._max_num_solutions:
			print('This is not implemented.') # @todo:
			pass
		if len(result['node_maps']) == 0:
			result['upper_bound'] = np.inf
		else:
			result['upper_bound'] = result['node_maps'][0].induced_cost()
					
		return result
			
		
	
	def populate_instance(self, g, h, lsape_instance):
		"""
	/*!
	 * @brief Populates the LSAPE instance.
	 * @param[in] g Input graph.
	 * @param[in] h Input graph.
	 * @param[out] lsape_instance LSAPE instance.
	 */
		"""
		if not self._initialized:
			pass
		# @todo: if (not this->initialized_) {
		self._lsape_populate_instance(g, h, lsape_instance)
		lsape_instance[nx.number_of_nodes(g):, nx.number_of_nodes(h):] = 0
# 		lsape_instance[nx.number_of_nodes(g), nx.number_of_nodes(h)] = 0
		
	
	###########################################################################
	# Member functions inherited from GEDMethod.
	###########################################################################
	
	
	def _ged_init(self):
		self._lsape_pre_graph_init(False)
		for graph in self._ged_data._graphs:
			self._init_graph(graph)
		self._lsape_init()
		
		
	def _ged_run(self, g, h):
# 		lsape_instance = np.empty((0, 0))
		result = self.populate_instance_and_run_as_util(g, h) # , lsape_instance)
		return result
		
		
	def _ged_parse_option(self, option, arg):
		is_valid_option = False
		
		if option == 'threads': # @todo: try.. catch...
			self._num_threads = arg
			is_valid_option = True
		elif option == 'lsape_model':
			self._lsape_model = arg # @todo
			is_valid_option = True
		elif option == 'greedy_method':
			self._greedy_method = arg # @todo
			is_valid_option = True
		elif option == 'optimal':
			self._solve_optimally = arg # @todo
			is_valid_option = True
		elif option == 'centrality_method':
			self._centrality_method = arg # @todo
			is_valid_option = True
		elif option == 'centrality_weight':
			self._centrality_weight = arg # @todo
			is_valid_option = True
		elif option == 'max_num_solutions':
			if arg == 'ALL':
				self._max_num_solutions = -1
			else:				
				self._max_num_solutions = arg # @todo
			is_valid_option = True
			
		is_valid_option = is_valid_option or self._lsape_parse_option(option, arg)
		is_valid_option = True # @todo: this is not in the C++ code.
		return is_valid_option
		
		
	def _ged_set_default_options(self):
		self._lsape_model = None # @todo: LSAPESolver::ECBP
		self._greedy_method = None # @todo: LSAPESolver::BASIC
		self._solve_optimally = True
		self._num_threads = 1
		self._centrality_method = 'NODE' # @todo
		self._centrality_weight = 0.7
		self._max_num_solutions = 1
		
		
	###########################################################################
	# Private helper member functions.
	###########################################################################
	
	
	def _init_graph(self, graph):
		if self._centrality_method != 'NODE':
			self._init_centralities(graph) # @todo
		self._lsape_init_graph(graph)
	
	
	###########################################################################
	# Virtual member functions to be overridden by derived classes.
	###########################################################################
	
	
	def _lsape_init(self):
		"""
	/*!
	 * @brief Initializes the method after initializing the global variables for the graphs.
	 * @note Must be overridden by derived classes of ged::LSAPEBasedMethod that require custom initialization.
	 */
		"""
		pass
		
		
	def _lsape_parse_option(self, option, arg):
		"""
	/*!
	 * @brief Parses one option that is not among the ones shared by all derived classes of ged::LSAPEBasedMethod.
	 * @param[in] option The name of the option.
	 * @param[in] arg The argument of the option.
	 * @return Returns true if @p option is a valid option name for the method and false otherwise.
	 * @note Must be overridden by derived classes of ged::LSAPEBasedMethod that have options that are not among the ones shared by all derived classes of ged::LSAPEBasedMethod.
	 */
		"""
		return False
		
		
	def _lsape_set_default_options(self):
		"""
	/*!
	 * @brief Sets all options that are not among the ones shared by all derived classes of ged::LSAPEBasedMethod to default values.
	 * @note Must be overridden by derived classes of ged::LSAPEBasedMethod that have options that are not among the ones shared by all derived classes of ged::LSAPEBasedMethod.
	 */
		"""
		pass
	
	
	def _lsape_populate_instance(self, g, h, lsape_instance):
		"""
	/*!
	 * @brief Populates the LSAPE instance.
	 * @param[in] g Input graph.
	 * @param[in] h Input graph.
	 * @param[out] lsape_instance LSAPE instance of size (n + 1) x (m + 1), where n and m are the number of nodes in @p g and @p h. The last row and the last column represent insertion and deletion.
	 * @note Must be overridden by derived classes of ged::LSAPEBasedMethod.
	 */
		"""
		pass
	
	
	def _lsape_init_graph(self, graph):
		"""
	/*!
	 * @brief Initializes global variables for one graph.
	 * @param[in] graph Graph for which the global variables have to be initialized.
	 * @note Must be overridden by derived classes of ged::LSAPEBasedMethod that require to initialize custom global variables.
	 */
		"""
		pass
	
	
	def _lsape_pre_graph_init(self, called_at_runtime):
		"""
	/*!
	 * @brief Initializes the method at runtime or during initialization before initializing the global variables for the graphs.
	 * @param[in] called_at_runtime Equals @p true if called at runtime and @p false if called during initialization.
	 * @brief Must be overridden by derived classes of ged::LSAPEBasedMethod that require default initialization at runtime before initializing the global variables for the graphs.
	 */
		"""
		pass