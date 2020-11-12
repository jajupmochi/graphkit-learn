#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:04:55 2020

@author: ljia
"""
import numpy as np
from gklearn.ged.env import AlgorithmState, NodeMap
from gklearn.ged.util import misc
from gklearn.utils import Timer
import time
from tqdm import tqdm
import sys
import networkx as nx
import multiprocessing
from multiprocessing import Pool
from functools import partial


class MedianGraphEstimator(object): # @todo: differ dummy_node from undifined node?
	
	def __init__(self, ged_env, constant_node_costs):
		"""Constructor.
		
		Parameters
		----------
		ged_env : gklearn.gedlib.gedlibpy.GEDEnv
			Initialized GED environment. The edit costs must be set by the user.
			
		constant_node_costs : Boolean
			Set to True if the node relabeling costs are constant.
		"""
		self._ged_env = ged_env
		self._init_method = 'BRANCH_FAST'
		self._init_options = ''
		self._descent_method = 'BRANCH_FAST'
		self._descent_options = ''
		self._refine_method = 'IPFP'
		self._refine_options = ''
		self._constant_node_costs = constant_node_costs
		self._labeled_nodes = (ged_env.get_num_node_labels() > 1)
		self._node_del_cost = ged_env.get_node_del_cost(ged_env.get_node_label(1))
		self._node_ins_cost = ged_env.get_node_ins_cost(ged_env.get_node_label(1))
		self._labeled_edges = (ged_env.get_num_edge_labels() > 1)
		self._edge_del_cost = ged_env.get_edge_del_cost(ged_env.get_edge_label(1))
		self._edge_ins_cost = ged_env.get_edge_ins_cost(ged_env.get_edge_label(1))
		self._init_type = 'RANDOM'
		self._num_random_inits = 10
		self._desired_num_random_inits = 10
		self._use_real_randomness = True
		self._seed = 0
		self._parallel = True
		self._update_order = True
		self._sort_graphs = True # sort graphs by size when computing GEDs.
		self._refine = True
		self._time_limit_in_sec = 0
		self._epsilon = 0.0001
		self._max_itrs = 100
		self._max_itrs_without_update = 3
		self._num_inits_increase_order = 10
		self._init_type_increase_order = 'K-MEANS++'
		self._max_itrs_increase_order = 10
		self._print_to_stdout = 2
		self._median_id = np.inf # @todo: check
		self._node_maps_from_median = {}
		self._sum_of_distances = 0
		self._best_init_sum_of_distances = np.inf
		self._converged_sum_of_distances = np.inf
		self._runtime = None
		self._runtime_initialized = None
		self._runtime_converged = None
		self._itrs = [] # @todo: check: {} ?
		self._num_decrease_order = 0
		self._num_increase_order = 0
		self._num_converged_descents = 0
		self._state = AlgorithmState.TERMINATED
		self._label_names = {}
		
		if ged_env is None:
			raise Exception('The GED environment pointer passed to the constructor of MedianGraphEstimator is null.')
		elif not ged_env.is_initialized():
			raise Exception('The GED environment is uninitialized. Call gedlibpy.GEDEnv.init() before passing it to the constructor of MedianGraphEstimator.')
	
	
	def set_options(self, options):
		"""Sets the options of the estimator.

		Parameters
		----------
		options : string
			String that specifies with which options to run the estimator.
		"""
		self._set_default_options()
		options_map = misc.options_string_to_options_map(options)
		for opt_name, opt_val in options_map.items():
			if opt_name == 'init-type':
				self._init_type = opt_val
				if opt_val != 'MEDOID' and opt_val != 'RANDOM' and opt_val != 'MIN' and opt_val != 'MAX' and opt_val != 'MEAN':
					raise Exception('Invalid argument ' + opt_val + ' for option init-type. Usage: options = "[--init-type RANDOM|MEDOID|EMPTY|MIN|MAX|MEAN] [...]"')
			elif opt_name == 'random-inits':
				try:
					self._num_random_inits = int(opt_val)
					self._desired_num_random_inits = self._num_random_inits
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option random-inits. Usage: options = "[--random-inits <convertible to int greater 0>]"')

				if self._num_random_inits <= 0:
					raise Exception('Invalid argument "' + opt_val + '" for option random-inits. Usage: options = "[--random-inits <convertible to int greater 0>]"')
	
			elif opt_name == 'randomness':
				if opt_val == 'PSEUDO':
					self._use_real_randomness = False
	
				elif opt_val == 'REAL':
					self._use_real_randomness = True
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option randomness. Usage: options = "[--randomness REAL|PSEUDO] [...]"')
	
			elif opt_name == 'stdout':
				if opt_val == '0':
					self._print_to_stdout = 0
	
				elif opt_val == '1':
					self._print_to_stdout = 1
	
				elif opt_val == '2':
					self._print_to_stdout = 2
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option stdout. Usage: options = "[--stdout 0|1|2] [...]"')

			elif opt_name == 'parallel':
				if opt_val == 'TRUE':
					self._parallel = True
	
				elif opt_val == 'FALSE':
					self._parallel = False
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option parallel. Usage: options = "[--parallel TRUE|FALSE] [...]"')
	
			elif opt_name == 'update-order':
				if opt_val == 'TRUE':
					self._update_order = True
	
				elif opt_val == 'FALSE':
					self._update_order = False
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option update-order. Usage: options = "[--update-order TRUE|FALSE] [...]"')
					
			elif opt_name == 'sort-graphs':
				if opt_val == 'TRUE':
					self._sort_graphs = True
	
				elif opt_val == 'FALSE':
					self._sort_graphs = False
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option sort-graphs. Usage: options = "[--sort-graphs TRUE|FALSE] [...]"')
					
			elif opt_name == 'refine':
				if opt_val == 'TRUE':
					self._refine = True
	
				elif opt_val == 'FALSE':
					self._refine = False
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option refine. Usage: options = "[--refine TRUE|FALSE] [...]"')
	
			elif opt_name == 'time-limit':
				try:
					self._time_limit_in_sec = float(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option time-limit.  Usage: options = "[--time-limit <convertible to double>] [...]')
	
			elif opt_name == 'max-itrs':
				try:
					self._max_itrs = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option max-itrs. Usage: options = "[--max-itrs <convertible to int>] [...]')
	
			elif opt_name == 'max-itrs-without-update':
				try:
					self._max_itrs_without_update = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option max-itrs-without-update. Usage: options = "[--max-itrs-without-update <convertible to int>] [...]')
	
			elif opt_name == 'seed':
				try:
					self._seed = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option seed. Usage: options = "[--seed <convertible to int greater equal 0>] [...]')
	
			elif opt_name == 'epsilon':
				try:
					self._epsilon = float(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option epsilon. Usage: options = "[--epsilon <convertible to double greater 0>] [...]')
	
				if self._epsilon <= 0:
					raise Exception('Invalid argument "' + opt_val + '" for option epsilon. Usage: options = "[--epsilon <convertible to double greater 0>] [...]')
	
			elif opt_name == 'inits-increase-order':
				try:
					self._num_inits_increase_order = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option inits-increase-order. Usage: options = "[--inits-increase-order <convertible to int greater 0>]"')
	
				if self._num_inits_increase_order <= 0:
					raise Exception('Invalid argument "' + opt_val + '" for option inits-increase-order. Usage: options = "[--inits-increase-order <convertible to int greater 0>]"')

			elif opt_name == 'init-type-increase-order':
				self._init_type_increase_order = opt_val
				if opt_val != 'CLUSTERS' and opt_val != 'K-MEANS++':
					raise Exception('Invalid argument ' + opt_val + ' for option init-type-increase-order. Usage: options = "[--init-type-increase-order CLUSTERS|K-MEANS++] [...]"')
	
			elif opt_name == 'max-itrs-increase-order':
				try:
					self._max_itrs_increase_order = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option max-itrs-increase-order. Usage: options = "[--max-itrs-increase-order <convertible to int>] [...]')

			else:
				valid_options = '[--init-type <arg>] [--random-inits <arg>] [--randomness <arg>] [--seed <arg>] [--stdout <arg>] '
				valid_options += '[--time-limit <arg>] [--max-itrs <arg>] [--epsilon <arg>] '
				valid_options += '[--inits-increase-order <arg>] [--init-type-increase-order <arg>] [--max-itrs-increase-order <arg>]'
				raise Exception('Invalid option "' + opt_name + '". Usage: options = "' + valid_options + '"')
 
		
	def set_init_method(self, init_method, init_options=''):
		"""Selects method to be used for computing the initial medoid graph.
		
		Parameters
		----------
		init_method : string
			The selected method. Default: ged::Options::GEDMethod::BRANCH_UNIFORM.
		
		init_options : string
			The options for the selected method. Default: "".
		
		Notes
		-----
		Has no effect unless "--init-type MEDOID" is passed to set_options().
		"""
		self._init_method = init_method;
		self._init_options = init_options;
	
	
	def set_descent_method(self, descent_method, descent_options=''):
		"""Selects method to be used for block gradient descent..
		
		Parameters
		----------
		descent_method : string
			The selected method. Default: ged::Options::GEDMethod::BRANCH_FAST.
		
		descent_options : string
			The options for the selected method. Default: "".
		
		Notes
		-----
		Has no effect unless "--init-type MEDOID" is passed to set_options().
		"""
		self._descent_method = descent_method;
		self._descent_options = descent_options;

	
	def set_refine_method(self, refine_method, refine_options):
		"""Selects method to be used for improving the sum of distances and the node maps for the converged median.
		
		Parameters
		----------
		refine_method : string
			The selected method. Default: "IPFP".
			
		refine_options : string 
			The options for the selected method. Default: "".
					
		Notes
		-----
		Has no effect if "--refine FALSE" is passed to set_options().
		"""
		self._refine_method = refine_method
		self._refine_options = refine_options

	
	def run(self, graph_ids, set_median_id, gen_median_id):
		"""Computes a generalized median graph.
		
		Parameters
		----------
		graph_ids : list[integer]
			The IDs of the graphs for which the median should be computed. Must have been added to the environment passed to the constructor.
		
		set_median_id : integer
			The ID of the computed set-median. A dummy graph with this ID must have been added to the environment passed to the constructor. Upon termination, the computed median can be obtained via gklearn.gedlib.gedlibpy.GEDEnv.get_graph().


		gen_median_id : integer
			The ID of the computed generalized median. Upon termination, the computed median can be obtained via gklearn.gedlib.gedlibpy.GEDEnv.get_graph().
		"""
		# Sanity checks.
		if len(graph_ids) == 0:
			raise Exception('Empty vector of graph IDs, unable to compute median.')
		all_graphs_empty = True
		for graph_id in graph_ids:
			if self._ged_env.get_graph_num_nodes(graph_id) > 0:
				all_graphs_empty = False
				break
		if all_graphs_empty:
			raise Exception('All graphs in the collection are empty.')
			
		# Start timer and record start time.
		start = time.time()
		timer = Timer(self._time_limit_in_sec)
		self._median_id = gen_median_id
		self._state = AlgorithmState.TERMINATED
		
		# Get NetworkX graph representations of the input graphs.
		graphs = {}
		for graph_id in graph_ids:
			# @todo: get_nx_graph() function may need to be modified according to the coming code.
			graphs[graph_id] = self._ged_env.get_nx_graph(graph_id, True, True, False)
#		print(self._ged_env.get_graph_internal_id(0))
#		print(graphs[0].graph)
#		print(graphs[0].nodes(data=True))
#		print(graphs[0].edges(data=True))
#		print(nx.adjacency_matrix(graphs[0]))
			
		# Construct initial medians.
		medians = []
		self._construct_initial_medians(graph_ids, timer, medians)
		end_init = time.time()
		self._runtime_initialized = end_init - start
		print(medians[0].graph)
		print(medians[0].nodes(data=True))
		print(medians[0].edges(data=True))
		print(nx.adjacency_matrix(medians[0]))
		
		# Reset information about iterations and number of times the median decreases and increases.
		self._itrs = [0] * len(medians)
		self._num_decrease_order = 0
		self._num_increase_order = 0
		self._num_converged_descents = 0
		
		# Initialize the best median.
		best_sum_of_distances = np.inf
		self._best_init_sum_of_distances = np.inf
		node_maps_from_best_median = {}
		
		# Run block gradient descent from all initial medians.
		self._ged_env.set_method(self._descent_method, self._descent_options)
		for median_pos in range(0, len(medians)):
			
			# Terminate if the timer has expired and at least one SOD has been computed.
			if timer.expired() and median_pos > 0:
				break
			
			# Print information about current iteration.
			if self._print_to_stdout == 2:
				print('\n===========================================================')
				print('Block gradient descent for initial median', str(median_pos + 1), 'of', str(len(medians)), '.')
				print('-----------------------------------------------------------')
				
			# Get reference to the median.
			median = medians[median_pos]
			
			# Load initial median into the environment.
			self._ged_env.load_nx_graph(median, gen_median_id)
			self._ged_env.init(self._ged_env.get_init_type())
			
			# Compute node maps and sum of distances for initial median.
			xxx = self._node_maps_from_median
			self._compute_init_node_maps(graph_ids, gen_median_id)
			yyy = self._node_maps_from_median
			
			self._best_init_sum_of_distances = min(self._best_init_sum_of_distances, self._sum_of_distances)
			self._ged_env.load_nx_graph(median, set_median_id)
			print(self._best_init_sum_of_distances)
				
			# Run block gradient descent from initial median.
			converged = False
			itrs_without_update = 0
			while not self._termination_criterion_met(converged, timer, self._itrs[median_pos], itrs_without_update):
				
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					print('\n===========================================================')
					print('Iteration', str(self._itrs[median_pos] + 1), 'for initial median', str(median_pos + 1), 'of', str(len(medians)), '.')
					print('-----------------------------------------------------------')
					
				# Initialize flags that tell us what happened in the iteration.
				median_modified = False
				node_maps_modified = False
				decreased_order = False
				increased_order = False
				
				# Update the median.
				median_modified = self._update_median(graphs, median)
				if self._update_order:
					if not median_modified or self._itrs[median_pos] == 0:
						decreased_order = self._decrease_order(graphs, median)
						if not decreased_order or self._itrs[median_pos] == 0:
							increased_order = self._increase_order(graphs, median)
						
				# Update the number of iterations without update of the median.
				if median_modified or decreased_order or increased_order:
					itrs_without_update = 0
				else:
					itrs_without_update += 1
					
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					print('Loading median to environment: ... ', end='')
					
				# Load the median into the environment.
				# @todo: should this function use the original node label?
				self._ged_env.load_nx_graph(median, gen_median_id)
				self._ged_env.init(self._ged_env.get_init_type())
					
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					print('done.')					
					
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					print('Updating induced costs: ... ', end='')

				# Compute induced costs of the old node maps w.r.t. the updated median.
				for graph_id in graph_ids:
# 					print(self._node_maps_from_median[graph_id].induced_cost())
# 					xxx = self._node_maps_from_median[graph_id]					   
					self._ged_env.compute_induced_cost(gen_median_id, graph_id, self._node_maps_from_median[graph_id])
# 					print('---------------------------------------')
# 					print(self._node_maps_from_median[graph_id].induced_cost())
					# @todo:!!!!!!!!!!!!!!!!!!!!!!!!!!!!This value is a slight different from the c++ program, which might be a bug! Use it very carefully!
					
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					print('done.')					
					
				# Update the node maps.
				node_maps_modified = self._update_node_maps()

				# Update the order of the median if no improvement can be found with the current order.
				
				# Update the sum of distances.
				old_sum_of_distances = self._sum_of_distances
				self._sum_of_distances = 0
				for graph_id, node_map in self._node_maps_from_median.items():
					self._sum_of_distances += node_map.induced_cost()
# 					print(self._sum_of_distances)
					
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					print('Old local SOD: ', old_sum_of_distances)
					print('New local SOD: ', self._sum_of_distances)
					print('Best converged SOD: ', best_sum_of_distances)
					print('Modified median: ', median_modified)
					print('Modified node maps: ', node_maps_modified)
					print('Decreased order: ', decreased_order)
					print('Increased order: ', increased_order)
					print('===========================================================\n')
					
				converged = not (median_modified or node_maps_modified or decreased_order or increased_order)
				
				self._itrs[median_pos] += 1
				
			# Update the best median.
			if self._sum_of_distances < best_sum_of_distances:
				best_sum_of_distances = self._sum_of_distances
				node_maps_from_best_median = self._node_maps_from_median.copy() # @todo: this is a shallow copy, not sure if it is enough.
				best_median = median
				
			# Update the number of converged descents.
			if converged:
				self._num_converged_descents += 1
				
		# Store the best encountered median.
		self._sum_of_distances = best_sum_of_distances
		self._node_maps_from_median = node_maps_from_best_median
		self._ged_env.load_nx_graph(best_median, gen_median_id)
		self._ged_env.init(self._ged_env.get_init_type())
		end_descent = time.time()
		self._runtime_converged = end_descent - start
		
		# Refine the sum of distances and the node maps for the converged median.
		self._converged_sum_of_distances = self._sum_of_distances
		if self._refine:
			self._improve_sum_of_distances(timer)
		
		# Record end time, set runtime and reset the number of initial medians.
		end = time.time()
		self._runtime = end - start
		self._num_random_inits = self._desired_num_random_inits
		
		# Print global information.
		if self._print_to_stdout != 0:
			print('\n===========================================================')
			print('Finished computation of generalized median graph.')
			print('-----------------------------------------------------------')
			print('Best SOD after initialization: ', self._best_init_sum_of_distances)
			print('Converged SOD: ', self._converged_sum_of_distances)
			if self._refine:
				print('Refined SOD: ', self._sum_of_distances)
			print('Overall runtime: ', self._runtime)
			print('Runtime of initialization: ', self._runtime_initialized)
			print('Runtime of block gradient descent: ', self._runtime_converged - self._runtime_initialized)
			if self._refine:
				print('Runtime of refinement: ', self._runtime - self._runtime_converged)
			print('Number of initial medians: ', len(medians))
			total_itr = 0
			num_started_descents = 0
			for itr in self._itrs:
				total_itr += itr
				if itr > 0:
					num_started_descents += 1
			print('Size of graph collection: ', len(graph_ids))
			print('Number of started descents: ', num_started_descents)
			print('Number of converged descents: ', self._num_converged_descents)
			print('Overall number of iterations: ', total_itr)
			print('Overall number of times the order decreased: ', self._num_decrease_order)
			print('Overall number of times the order increased: ', self._num_increase_order)
			print('===========================================================\n')
			
			
	def _improve_sum_of_distances(self, timer): # @todo: go through and test
		# Use method selected for refinement phase.
		self._ged_env.set_method(self._refine_method, self._refine_options)
		
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			progress = tqdm(desc='Improving node maps', total=len(self._node_maps_from_median), file=sys.stdout)
			print('\n===========================================================')
			print('Improving node maps and SOD for converged median.')
			print('-----------------------------------------------------------')
			progress.update(1)
			
		# Improving the node maps.
		nb_nodes_median = self._ged_env.get_graph_num_nodes(self._gen_median_id)
		for graph_id, node_map in self._node_maps_from_median.items():
			if time.expired():
				if self._state == AlgorithmState.TERMINATED:
					self._state = AlgorithmState.CONVERGED
				break

			nb_nodes_g = self._ged_env.get_graph_num_nodes(graph_id)
			if nb_nodes_median <= nb_nodes_g or not self._sort_graphs:			
				self._ged_env.run_method(self._gen_median_id, graph_id)
				if self._ged_env.get_upper_bound(self._gen_median_id, graph_id) < node_map.induced_cost():
					self._node_maps_from_median[graph_id] = self._ged_env.get_node_map(self._gen_median_id, graph_id)
			else:
				self._ged_env.run_method(graph_id, self._gen_median_id)
				if self._ged_env.get_upper_bound(graph_id, self._gen_median_id) < node_map.induced_cost():
						node_map_tmp = self._ged_env.get_node_map(graph_id, self._gen_median_id)
						node_map_tmp.forward_map, node_map_tmp.backward_map = node_map_tmp.backward_map, node_map_tmp.forward_map
						self._node_maps_from_median[graph_id] = node_map_tmp	
			
			self._sum_of_distances += self._node_maps_from_median[graph_id].induced_cost()				

			# Print information.
			if self._print_to_stdout == 2:
				progress.update(1)

		self._sum_of_distances = 0.0
		for key, val in self._node_maps_from_median.items():
			self._sum_of_distances += val.induced_cost()
			
		# Print information.
		if self._print_to_stdout == 2:
			print('===========================================================\n')
			
	
	def _median_available(self):
		return self._median_id != np.inf
	
	
	def get_state(self):
		if not self._median_available():
			raise Exception('No median has been computed. Call run() before calling get_state().')
		return self._state
		
			
	def get_sum_of_distances(self, state=''):
		"""Returns the sum of distances.
		
		Parameters
		----------
		state : string
			The state of the estimator. Can be 'initialized' or 'converged'. Default: ""
			
		Returns
		-------
		float
			The sum of distances (SOD) of the median when the estimator was in the state `state` during the last call to run(). If `state` is not given, the converged SOD (without refinement) or refined SOD (with refinement) is returned.
		"""
		if not self._median_available():
			raise Exception('No median has been computed. Call run() before calling get_sum_of_distances().')
		if state == 'initialized':
			return self._best_init_sum_of_distances
		if state == 'converged':
			return self._converged_sum_of_distances
		return self._sum_of_distances


	def get_runtime(self, state):
		if not self._median_available():
			raise Exception('No median has been computed. Call run() before calling get_runtime().')
		if state == AlgorithmState.INITIALIZED:
			return self._runtime_initialized
		if state == AlgorithmState.CONVERGED:
			return self._runtime_converged
		return self._runtime
	

	def get_num_itrs(self):
		if not self._median_available():
			raise Exception('No median has been computed. Call run() before calling get_num_itrs().')
		return self._itrs


	def get_num_times_order_decreased(self):
		if not self._median_available():
			raise Exception('No median has been computed. Call run() before calling get_num_times_order_decreased().')
		return self._num_decrease_order		
	
	
	def get_num_times_order_increased(self):
		if not self._median_available():
			raise Exception('No median has been computed. Call run() before calling get_num_times_order_increased().')
		return self._num_increase_order		
	
	
	def get_num_converged_descents(self):
		if not self._median_available():
			raise Exception('No median has been computed. Call run() before calling get_num_converged_descents().')
		return self._num_converged_descents
	
	
	def get_ged_env(self):
		return self._ged_env
	
	
	def _set_default_options(self):
		self._init_type = 'RANDOM'
		self._num_random_inits = 10
		self._desired_num_random_inits = 10
		self._use_real_randomness = True
		self._seed = 0
		self._parallel = True
		self._update_order = True
		self._sort_graphs = True
		self._refine = True
		self._time_limit_in_sec = 0
		self._epsilon = 0.0001
		self._max_itrs = 100
		self._max_itrs_without_update = 3
		self._num_inits_increase_order = 10
		self._init_type_increase_order = 'K-MEANS++'
		self._max_itrs_increase_order = 10
		self._print_to_stdout = 2
		self._label_names = {}
		
		
	def _construct_initial_medians(self, graph_ids, timer, initial_medians):
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('\n===========================================================')
			print('Constructing initial median(s).')
			print('-----------------------------------------------------------')
			
		# Compute or sample the initial median(s).
		initial_medians.clear()
		if self._init_type == 'MEDOID':
			self._compute_medoid(graph_ids, timer, initial_medians)
		elif self._init_type == 'MAX':
			pass # @todo
#			compute_max_order_graph_(graph_ids, initial_medians)
		elif self._init_type == 'MIN':
			pass # @todo
#			compute_min_order_graph_(graph_ids, initial_medians)
		elif self._init_type == 'MEAN':
			pass # @todo
#			compute_mean_order_graph_(graph_ids, initial_medians)
		else:
			pass # @todo
#			sample_initial_medians_(graph_ids, initial_medians)

		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('===========================================================')
			
			
	def _compute_medoid(self, graph_ids, timer, initial_medians):
		# Use method selected for initialization phase.
		self._ged_env.set_method(self._init_method, self._init_options)
					
		# Compute the medoid.
		if self._parallel:
			# @todo: notice when parallel self._ged_env is not modified.
			sum_of_distances_list = [np.inf] * len(graph_ids)
			len_itr = len(graph_ids)
			itr = zip(graph_ids, range(0, len(graph_ids)))
			n_jobs = multiprocessing.cpu_count()
			if len_itr < 100 * n_jobs:
				chunksize = int(len_itr / n_jobs) + 1
			else:
				chunksize = 100
			def init_worker(ged_env_toshare):
				global G_ged_env
				G_ged_env = ged_env_toshare
			do_fun = partial(_compute_medoid_parallel, graph_ids, self._sort_graphs)
			pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(self._ged_env,))
			if self._print_to_stdout == 2:
				iterator = tqdm(pool.imap_unordered(do_fun, itr, chunksize),
							desc='Computing medoid', file=sys.stdout)
			else:
				iterator = pool.imap_unordered(do_fun, itr, chunksize)
			for i, dis in iterator:
				sum_of_distances_list[i] = dis
			pool.close()
			pool.join()
			
			medoid_id = np.argmin(sum_of_distances_list)
			best_sum_of_distances = sum_of_distances_list[medoid_id]
			
			initial_medians.append(self._ged_env.get_nx_graph(medoid_id, True, True, False)) # @todo

		else:
			# Print information about current iteration.
			self.ged_matrix_set_median_tmp = np.ones((len(graph_ids), len(graph_ids))) * np.inf
			if self._print_to_stdout == 2:
				progress = tqdm(desc='Computing medoid', total=len(graph_ids), file=sys.stdout)
		
			medoid_id = graph_ids[0]
			best_sum_of_distances = np.inf
			for g_id in graph_ids:
				if timer.expired():
					self._state = AlgorithmState.CALLED
					break
				nb_nodes_g = self._ged_env.get_graph_num_nodes(g_id)
				sum_of_distances = 0
				for h_id in graph_ids: # @todo: can this be faster?				
					nb_nodes_h = self._ged_env.get_graph_num_nodes(h_id)
					if nb_nodes_g <= nb_nodes_h or not self._sort_graphs:
						self._ged_env.run_method(g_id, h_id)
						sum_of_distances += self._ged_env.get_upper_bound(g_id, h_id)
						self.ged_matrix_set_median_tmp[g_id, h_id] = self._ged_env.get_upper_bound(g_id, h_id)
					else:
						# @todo: is this correct?
						self._ged_env.run_method(h_id, g_id)
						sum_of_distances += self._ged_env.get_upper_bound(h_id, g_id)
						self.ged_matrix_set_median_tmp[g_id, h_id] = self._ged_env.get_upper_bound(h_id, g_id)
					print(sum_of_distances)
				if sum_of_distances < best_sum_of_distances:
					best_sum_of_distances = sum_of_distances
					medoid_id = g_id
					
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					progress.update(1)
					
			initial_medians.append(self._ged_env.get_nx_graph(medoid_id, True, True, False)) # @todo
			
			# Print information about current iteration.
			if self._print_to_stdout == 2:
				print('\n')
			
			
	def _compute_init_node_maps(self, graph_ids, gen_median_id):
		# Compute node maps and sum of distances for initial median.
		if self._parallel:
			# @todo: notice when parallel self._ged_env is not modified.
			self._sum_of_distances = 0
			self._node_maps_from_median.clear()
			sum_of_distances_list = [0] * len(graph_ids)
			
			len_itr = len(graph_ids)
			itr = graph_ids
			n_jobs = multiprocessing.cpu_count()
			if len_itr < 100 * n_jobs:
				chunksize = int(len_itr / n_jobs) + 1
			else:
				chunksize = 100
			def init_worker(ged_env_toshare):
				global G_ged_env
				G_ged_env = ged_env_toshare
			nb_nodes_median = self._ged_env.get_graph_num_nodes(gen_median_id)
			do_fun = partial(_compute_init_node_maps_parallel, gen_median_id, self._sort_graphs, nb_nodes_median)
			pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(self._ged_env,))
			if self._print_to_stdout == 2:
				iterator = tqdm(pool.imap_unordered(do_fun, itr, chunksize),
							desc='Computing initial node maps', file=sys.stdout)
			else:
				iterator = pool.imap_unordered(do_fun, itr, chunksize)
			for g_id, sod, node_maps in iterator:
				sum_of_distances_list[g_id] = sod
				self._node_maps_from_median[g_id] = node_maps
			pool.close()
			pool.join()
			
			self._sum_of_distances = np.sum(sum_of_distances_list)
# 			xxx = self._node_maps_from_median
			
		else:
			# Print information about current iteration.
			if self._print_to_stdout == 2:
				progress = tqdm(desc='Computing initial node maps', total=len(graph_ids), file=sys.stdout)
				
			self._sum_of_distances = 0
			self._node_maps_from_median.clear()
			nb_nodes_median = self._ged_env.get_graph_num_nodes(gen_median_id)
			for graph_id in graph_ids:
				nb_nodes_g = self._ged_env.get_graph_num_nodes(graph_id)
				if nb_nodes_median <= nb_nodes_g or not self._sort_graphs:
					self._ged_env.run_method(gen_median_id, graph_id)
					self._node_maps_from_median[graph_id] = self._ged_env.get_node_map(gen_median_id, graph_id)
				else:
					self._ged_env.run_method(graph_id, gen_median_id)
					node_map_tmp = self._ged_env.get_node_map(graph_id, gen_median_id)
					node_map_tmp.forward_map, node_map_tmp.backward_map = node_map_tmp.backward_map, node_map_tmp.forward_map
					self._node_maps_from_median[graph_id] = node_map_tmp
	# 				print(self._node_maps_from_median[graph_id])
				self._sum_of_distances += self._node_maps_from_median[graph_id].induced_cost()
	# 				print(self._sum_of_distances)
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					progress.update(1)
			
			# Print information about current iteration.
			if self._print_to_stdout == 2:
				print('\n')

		
	def _termination_criterion_met(self, converged, timer, itr, itrs_without_update):
		if timer.expired() or (itr >= self._max_itrs if self._max_itrs >= 0 else False):
			if self._state == AlgorithmState.TERMINATED:
				self._state = AlgorithmState.INITIALIZED
			return True
		return converged or (itrs_without_update > self._max_itrs_without_update if self._max_itrs_without_update >= 0 else False)
	
	
	def _update_median(self, graphs, median):
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('Updating median: ', end='')
			
		# Store copy of the old median.
		old_median = median.copy() # @todo: this is just a shallow copy.
		
		# Update the node labels.
		if self._labeled_nodes:
			self._update_node_labels(graphs, median)
			
		# Update the edges and their labels.
		self._update_edges(graphs, median)
		
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('done.')
			
		return not self._are_graphs_equal(median, old_median)
		
		
	def _update_node_labels(self, graphs, median):
# 		print('----------------------------')
		
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('nodes ... ', end='')
			
		# Iterate through all nodes of the median.
		for i in range(0, nx.number_of_nodes(median)):
# 			print('i: ', i)
			# Collect the labels of the substituted nodes.
			node_labels = []
			for graph_id, graph in graphs.items():
# 				print('graph_id: ', graph_id)
# 				print(self._node_maps_from_median[graph_id])
# 				print(self._node_maps_from_median[graph_id].forward_map, self._node_maps_from_median[graph_id].backward_map)
				k = self._node_maps_from_median[graph_id].image(i)
# 				print('k: ', k)
				if k != np.inf:
					node_labels.append(graph.nodes[k])
					
			# Compute the median label and update the median.
			if len(node_labels) > 0:
#				median_label = self._ged_env.get_median_node_label(node_labels)
				median_label = self._get_median_node_label(node_labels)
				if self._ged_env.get_node_rel_cost(median.nodes[i], median_label) > self._epsilon:
					nx.set_node_attributes(median, {i: median_label})
					
					
	def _update_edges(self, graphs, median):
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('edges ... ', end='')
			
#		# Clear the adjacency lists of the median and reset number of edges to 0.
# 		median_edges = list(median.edges)		
# 		for (head, tail) in median_edges:
# 			median.remove_edge(head, tail)
		
		# @todo: what if edge is not labeled?
		# Iterate through all possible edges (i,j) of the median.
		for i in range(0, nx.number_of_nodes(median)):
			for j in range(i + 1, nx.number_of_nodes(median)):
				
				# Collect the labels of the edges to which (i,j) is mapped by the node maps.
				edge_labels = []
				for graph_id, graph in graphs.items():
					k = self._node_maps_from_median[graph_id].image(i)
					l = self._node_maps_from_median[graph_id].image(j)
					if k != np.inf and l != np.inf:
						if graph.has_edge(k, l):
							edge_labels.append(graph.edges[(k, l)])
							
				# Compute the median edge label and the overall edge relabeling cost.
				rel_cost = 0
				median_label = self._ged_env.get_edge_label(1)
				if median.has_edge(i, j):
					median_label = median.edges[(i, j)]
				if self._labeled_edges and len(edge_labels) > 0:
					new_median_label = self._get_median_edge_label(edge_labels)
					if self._ged_env.get_edge_rel_cost(median_label, new_median_label) > self._epsilon:
						median_label = new_median_label
					for edge_label in edge_labels:
						rel_cost += self._ged_env.get_edge_rel_cost(median_label, edge_label)
						
				# Update the median.
				if median.has_edge(i, j):
					median.remove_edge(i, j)
				if rel_cost < (self._edge_ins_cost + self._edge_del_cost) * len(edge_labels) - self._edge_del_cost * len(graphs):
					median.add_edge(i, j, **median_label)
# 				else:
# 					if median.has_edge(i, j):
# 						median.remove_edge(i, j)


	def _update_node_maps(self):
		# Update the node maps.
		if self._parallel:
			# @todo: notice when parallel self._ged_env is not modified.
			node_maps_were_modified = False
# 			xxx = self._node_maps_from_median.copy()
			
			len_itr = len(self._node_maps_from_median)
			itr = [item for item in self._node_maps_from_median.items()]
			n_jobs = multiprocessing.cpu_count()
			if len_itr < 100 * n_jobs:
				chunksize = int(len_itr / n_jobs) + 1
			else:
				chunksize = 100
			def init_worker(ged_env_toshare):
				global G_ged_env
				G_ged_env = ged_env_toshare
			nb_nodes_median = self._ged_env.get_graph_num_nodes(self._median_id)
			do_fun = partial(_update_node_maps_parallel, self._median_id, self._epsilon, self._sort_graphs, nb_nodes_median)
			pool = Pool(processes=n_jobs, initializer=init_worker, initargs=(self._ged_env,))
			if self._print_to_stdout == 2:
				iterator = tqdm(pool.imap_unordered(do_fun, itr, chunksize),
							desc='Updating node maps', file=sys.stdout)
			else:
				iterator = pool.imap_unordered(do_fun, itr, chunksize)
			for g_id, node_map, nm_modified in iterator:
				self._node_maps_from_median[g_id] = node_map
				if nm_modified:
					node_maps_were_modified = True
			pool.close()
			pool.join()
# 			yyy = self._node_maps_from_median.copy()

		else:
			# Print information about current iteration.
			if self._print_to_stdout == 2:
				progress = tqdm(desc='Updating node maps', total=len(self._node_maps_from_median), file=sys.stdout)
				
			node_maps_were_modified = False
			nb_nodes_median = self._ged_env.get_graph_num_nodes(self._median_id)
			for graph_id, node_map in self._node_maps_from_median.items():
				nb_nodes_g = self._ged_env.get_graph_num_nodes(graph_id)
				
				if nb_nodes_median <= nb_nodes_g or not self._sort_graphs:
					self._ged_env.run_method(self._median_id, graph_id)
					if self._ged_env.get_upper_bound(self._median_id, graph_id) < node_map.induced_cost() - self._epsilon:
		# 				xxx = self._node_maps_from_median[graph_id]
						self._node_maps_from_median[graph_id] = self._ged_env.get_node_map(self._median_id, graph_id)
						node_maps_were_modified = True
						
				else:
					self._ged_env.run_method(graph_id, self._median_id)
					if self._ged_env.get_upper_bound(graph_id, self._median_id) < node_map.induced_cost() - self._epsilon:
						node_map_tmp = self._ged_env.get_node_map(graph_id, self._median_id)
						node_map_tmp.forward_map, node_map_tmp.backward_map = node_map_tmp.backward_map, node_map_tmp.forward_map
						self._node_maps_from_median[graph_id] = node_map_tmp	
						node_maps_were_modified = True
					
				# Print information about current iteration.
				if self._print_to_stdout == 2:
					progress.update(1)
				
			# Print information about current iteration.
			if self._print_to_stdout == 2:
				print('\n')
			
		# Return true if the node maps were modified.
		return node_maps_were_modified
	
	
	def _decrease_order(self, graphs, median):
		# Print information about current iteration
		if self._print_to_stdout == 2:
			print('Trying to decrease order: ... ', end='')
			
		if nx.number_of_nodes(median) <= 1:
			if self._print_to_stdout == 2:
				print('median graph has only 1 node, skip decrease.')
			return False
			
		# Initialize ID of the node that is to be deleted.
		id_deleted_node = [None] # @todo: or np.inf
		decreased_order = False
		
		# Decrease the order as long as the best deletion delta is negative.
		while self._compute_best_deletion_delta(graphs, median, id_deleted_node) < -self._epsilon:
			decreased_order = True
			self._delete_node_from_median(id_deleted_node[0], median)
			if nx.number_of_nodes(median) <= 1:
				if self._print_to_stdout == 2:
					print('decrease stopped because median graph remains only 1 node. ', end='')
				break
			
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('done.')
			
		# Return true iff the order was decreased.
		return decreased_order
	
	
	def _compute_best_deletion_delta(self, graphs, median, id_deleted_node):
		best_delta = 0.0
		
		# Determine node that should be deleted (if any).
		for i in range(0, nx.number_of_nodes(median)):
			# Compute cost delta.
			delta = 0.0
			for graph_id, graph in graphs.items():
				k = self._node_maps_from_median[graph_id].image(i)
				if k == np.inf:
					delta -= self._node_del_cost
				else:
					delta += self._node_ins_cost - self._ged_env.get_node_rel_cost(median.nodes[i], graph.nodes[k])
				for j, j_label in median[i].items():
					l = self._node_maps_from_median[graph_id].image(j)
					if k == np.inf or l == np.inf:
						delta -= self._edge_del_cost
					elif not graph.has_edge(k, l):
						delta -= self._edge_del_cost
					else:
						delta += self._edge_ins_cost - self._ged_env.get_edge_rel_cost(j_label, graph.edges[(k, l)])
						
			# Update best deletion delta.
			if delta < best_delta - self._epsilon:
				best_delta = delta
				id_deleted_node[0] = i
# 			id_deleted_node[0] = 3 # @todo: 
				
		return best_delta
	
	
	def _delete_node_from_median(self, id_deleted_node, median):
		# Update the median.
		mapping = {}
		for i in range(0, nx.number_of_nodes(median)):
			if i != id_deleted_node:
				new_i = (i if i < id_deleted_node else (i - 1))
				mapping[i] = new_i
		median.remove_node(id_deleted_node)
		nx.relabel_nodes(median, mapping, copy=False)
		
		# Update the node maps.
# 		xxx = self._node_maps_from_median
		for key, node_map in self._node_maps_from_median.items():
			new_node_map = NodeMap(nx.number_of_nodes(median), node_map.num_target_nodes())
			is_unassigned_target_node = [True] * node_map.num_target_nodes()
			for i in range(0, nx.number_of_nodes(median) + 1):
				if i != id_deleted_node:
					new_i = (i if i < id_deleted_node else (i - 1))
					k = node_map.image(i)
					new_node_map.add_assignment(new_i, k)
					if k != np.inf:
						is_unassigned_target_node[k] = False
			for k in range(0, node_map.num_target_nodes()):
				if is_unassigned_target_node[k]:
					new_node_map.add_assignment(np.inf, k)
# 			print(self._node_maps_from_median[key].forward_map, self._node_maps_from_median[key].backward_map)
# 			print(new_node_map.forward_map, new_node_map.backward_map
			self._node_maps_from_median[key] = new_node_map
			
		# Increase overall number of decreases.
		self._num_decrease_order += 1
	
	
	def _increase_order(self, graphs, median):
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('Trying to increase order: ... ', end='')
			
		# Initialize the best configuration and the best label of the node that is to be inserted.
		best_config = {}
		best_label = self._ged_env.get_node_label(1)
		increased_order = False
		
		# Increase the order as long as the best insertion delta is negative.
		while self._compute_best_insertion_delta(graphs, best_config, best_label) < - self._epsilon:
			increased_order = True
			self._add_node_to_median(best_config, best_label, median)
			
		# Print information about current iteration.
		if self._print_to_stdout == 2:
			print('done.')
			
		# Return true iff the order was increased.
		return increased_order
	
	
	def _compute_best_insertion_delta(self, graphs, best_config, best_label):
		# Construct sets of inserted nodes.
		no_inserted_node = True
		inserted_nodes = {}
		for graph_id, graph in graphs.items():
			inserted_nodes[graph_id] = []
			best_config[graph_id] = np.inf
			for k in range(nx.number_of_nodes(graph)):
				if self._node_maps_from_median[graph_id].pre_image(k) == np.inf:
					no_inserted_node = False
					inserted_nodes[graph_id].append((k, tuple(item for item in graph.nodes[k].items()))) # @todo: can order of label names be garantteed?
					
		# Return 0.0 if no node is inserted in any of the graphs.
		if no_inserted_node:
			return 0.0
		
		# Compute insertion configuration, label, and delta.
		best_delta = 0.0 # @todo
		if len(self._label_names['node_labels']) == 0 and len(self._label_names['node_attrs']) == 0: # @todo
			best_delta = self._compute_insertion_delta_unlabeled(inserted_nodes, best_config, best_label)
		elif len(self._label_names['node_labels']) > 0: # self._constant_node_costs:
			best_delta = self._compute_insertion_delta_constant(inserted_nodes, best_config, best_label)
		else:
			best_delta = self._compute_insertion_delta_generic(inserted_nodes, best_config, best_label)
			
		# Return the best delta.
		return best_delta
	
	
	def _compute_insertion_delta_unlabeled(self, inserted_nodes, best_config, best_label): # @todo: go through and test.
		# Construct the nest configuration and compute its insertion delta.
		best_delta = 0.0
		best_config.clear()
		for graph_id, node_set in inserted_nodes.items():
			if len(node_set) == 0:
				best_config[graph_id] = np.inf
				best_delta += self._node_del_cost
			else:
				best_config[graph_id] = node_set[0][0]
				best_delta -= self._node_ins_cost
				
		# Return the best insertion delta.
		return best_delta
	
	
	def _compute_insertion_delta_constant(self, inserted_nodes, best_config, best_label):
		# Construct histogram and inverse label maps.
		hist = {}
		inverse_label_maps = {}
		for graph_id, node_set in inserted_nodes.items():
			inverse_label_maps[graph_id] = {}
			for node in node_set:
				k = node[0]
				label = node[1]
				if label not in inverse_label_maps[graph_id]:
					inverse_label_maps[graph_id][label] = k
					if label not in hist:
						hist[label] = 1
					else:
						hist[label] += 1
				
		# Determine the best label.
		best_count = 0
		for key, val in hist.items():
			if val > best_count:
				best_count = val
				best_label_tuple = key
		
		# get best label.
		best_label.clear()
		for key, val in best_label_tuple:
			best_label[key] = val
				
		# Construct the best configuration and compute its insertion delta.
		best_config.clear()
		best_delta = 0.0
		node_rel_cost = self._ged_env.get_node_rel_cost(self._ged_env.get_node_label(1), self._ged_env.get_node_label(2))
		triangle_ineq_holds = (node_rel_cost <= self._node_del_cost + self._node_ins_cost)
		for graph_id, _ in inserted_nodes.items():
			if best_label_tuple in inverse_label_maps[graph_id]:
				best_config[graph_id] = inverse_label_maps[graph_id][best_label_tuple]
				best_delta -= self._node_ins_cost
			elif triangle_ineq_holds and not len(inserted_nodes[graph_id]) == 0:
				best_config[graph_id] = inserted_nodes[graph_id][0][0]
				best_delta += node_rel_cost - self._node_ins_cost
			else:
				best_config[graph_id] = np.inf
				best_delta += self._node_del_cost
				
		# Return the best insertion delta.
		return best_delta
	
	
	def _compute_insertion_delta_generic(self, inserted_nodes, best_config, best_label):
		# Collect all node labels of inserted nodes.
		node_labels = []
		for _, node_set in inserted_nodes.items():
			for node in node_set:
				node_labels.append(node[1])
				
		# Compute node label medians that serve as initial solutions for block gradient descent.
		initial_node_labels = []
		self._compute_initial_node_labels(node_labels, initial_node_labels)
		
		# Determine best insertion configuration, label, and delta via parallel block gradient descent from all initial node labels.
		best_delta = 0.0
		for node_label in initial_node_labels:
			# Construct local configuration.
			config = {}
			for graph_id, _ in inserted_nodes.items():
				config[graph_id] = tuple((np.inf, tuple(item for item in self._ged_env.get_node_label(1).items())))
				
			# Run block gradient descent.
			converged = False
			itr = 0
			while not self._insertion_termination_criterion_met(converged, itr):
				converged = not self._update_config(node_label, inserted_nodes, config, node_labels)
				node_label_dict = dict(node_label)
				converged = converged and (not self._update_node_label([dict(item) for item in node_labels], node_label_dict)) # @todo: the dict is tupled again in the function, can be better.
				node_label = tuple(item for item in node_label_dict.items()) # @todo: watch out: initial_node_labels[i] is not modified here.

				itr += 1
				
			# Compute insertion delta of converged solution.
			delta = 0.0
			for _, node in config.items():
				if node[0] == np.inf:
					delta += self._node_del_cost
				else:
					delta += self._ged_env.get_node_rel_cost(dict(node_label), dict(node[1])) - self._node_ins_cost
					
			# Update best delta and global configuration if improvement has been found.
			if delta < best_delta - self._epsilon:
				best_delta = delta
				best_label.clear()
				for key, val in node_label:
					best_label[key] = val
				best_config.clear()
				for graph_id, val in config.items():
					best_config[graph_id] = val[0]
					
		# Return the best delta.
		return best_delta


	def _compute_initial_node_labels(self, node_labels, median_labels):
		median_labels.clear()
		if self._use_real_randomness: # @todo: may not work if parallelized.
			rng = np.random.randint(0, high=2**32 - 1, size=1)
			urng = np.random.RandomState(seed=rng[0])
		else:
			urng = np.random.RandomState(seed=self._seed)
			
		# Generate the initial node label medians.
		if self._init_type_increase_order == 'K-MEANS++':
			# Use k-means++ heuristic to generate the initial node label medians.
			already_selected = [False] * len(node_labels)
			selected_label_id = urng.randint(low=0, high=len(node_labels), size=1)[0] # c++ test: 23
			median_labels.append(node_labels[selected_label_id])
			already_selected[selected_label_id] = True
# 			xxx = [41, 0, 18, 9, 6, 14, 21, 25, 33] for c++ test
# 			iii = 0 for c++ test
			while len(median_labels) < self._num_inits_increase_order:
				weights = [np.inf] * len(node_labels)
				for label_id in range(0, len(node_labels)):
					if already_selected[label_id]:
						weights[label_id] = 0
						continue
					for label in median_labels:
						weights[label_id] = min(weights[label_id], self._ged_env.get_node_rel_cost(dict(label), dict(node_labels[label_id])))
				
				# get non-zero weights.
				weights_p, idx_p = [], []
				for i, w in enumerate(weights):
					if w != 0:
						weights_p.append(w)
						idx_p.append(i)
				if len(weights_p) > 0:
					p = np.array(weights_p) / np.sum(weights_p)
					selected_label_id = urng.choice(range(0, len(weights_p)), size=1, p=p)[0] # for c++ test: xxx[iii] 
					selected_label_id = idx_p[selected_label_id]
# 				iii += 1 for c++ test
					median_labels.append(node_labels[selected_label_id])
					already_selected[selected_label_id] = True
				else: # skip the loop when all node_labels are selected. This happens when len(node_labels) <= self._num_inits_increase_order.
					break
		else:
			# Compute the initial node medians as the medians of randomly generated clusters of (roughly) equal size.
			# @todo: go through and test.
			shuffled_node_labels = [np.inf] * len(node_labels) #@todo: random?
			# @todo: std::shuffle(shuffled_node_labels.begin(), shuffled_node_labels.end(), urng);?
			cluster_size = len(node_labels) / self._num_inits_increase_order
			pos = 0.0
			cluster = []
			while len(median_labels) < self._num_inits_increase_order - 1:
				while pos < (len(median_labels) + 1) * cluster_size:
					cluster.append(shuffled_node_labels[pos])
					pos += 1
				median_labels.append(self._get_median_node_label(cluster))
				cluster.clear()
			while pos < len(shuffled_node_labels):
				pos += 1
				cluster.append(shuffled_node_labels[pos])
			median_labels.append(self._get_median_node_label(cluster))
			cluster.clear()
				
		# Run Lloyd's Algorithm.
		converged = False
		closest_median_ids = [np.inf] * len(node_labels)
		clusters = [[] for _ in range(len(median_labels))]
		itr = 1
		while not self._insertion_termination_criterion_met(converged, itr):
			converged = not self._update_clusters(node_labels, median_labels, closest_median_ids)
			if not converged:
				for cluster in clusters:
					cluster.clear()
				for label_id in range(0, len(node_labels)):
					clusters[closest_median_ids[label_id]].append(node_labels[label_id])
				for cluster_id in range(0, len(clusters)):
					node_label = dict(median_labels[cluster_id])
					self._update_node_label([dict(item) for item in clusters[cluster_id]], node_label) # @todo: the dict is tupled again in the function, can be better.
					median_labels[cluster_id] = tuple(item for item in node_label.items())
			itr += 1
			
			
	def _insertion_termination_criterion_met(self, converged, itr):
		return converged or (itr >= self._max_itrs_increase_order if self._max_itrs_increase_order > 0 else False)
	
	
	def _update_config(self, node_label, inserted_nodes, config, node_labels):
		# Determine the best configuration.
		config_modified = False
		for graph_id, node_set in inserted_nodes.items():
			best_assignment = config[graph_id]
			best_cost = 0.0
			if best_assignment[0] == np.inf:
				best_cost = self._node_del_cost
			else:
				best_cost = self._ged_env.get_node_rel_cost(dict(node_label), dict(best_assignment[1])) - self._node_ins_cost
			for node in node_set:
				cost = self._ged_env.get_node_rel_cost(dict(node_label), dict(node[1])) - self._node_ins_cost
				if cost < best_cost - self._epsilon:
					best_cost = cost
					best_assignment = node
					config_modified = True
			if self._node_del_cost < best_cost - self._epsilon:
				best_cost = self._node_del_cost
				best_assignment = tuple((np.inf, best_assignment[1]))
				config_modified = True
			config[graph_id] = best_assignment
			
		# Collect the node labels contained in the best configuration.
		node_labels.clear()
		for key, val in config.items():
			if val[0] != np.inf:
				node_labels.append(val[1])
		
		# Return true if the configuration was modified.
		return config_modified
				 
	
	def _update_node_label(self, node_labels, node_label):
		if len(node_labels) == 0: # @todo: check if this is the correct solution. Especially after calling _update_config().
			return False
		new_node_label = self._get_median_node_label(node_labels)
		if self._ged_env.get_node_rel_cost(new_node_label, node_label) > self._epsilon:
			node_label.clear()
			for key, val in new_node_label.items():
				node_label[key] = val
			return True
		return False
	
	
	def _update_clusters(self, node_labels, median_labels, closest_median_ids):
		# Determine the closest median for each node label.
		clusters_modified = False
		for label_id in range(0, len(node_labels)):
			closest_median_id = np.inf
			dist_to_closest_median = np.inf
			for median_id in range(0, len(median_labels)):
				dist_to_median = self._ged_env.get_node_rel_cost(dict(median_labels[median_id]), dict(node_labels[label_id]))
				if dist_to_median < dist_to_closest_median - self._epsilon:
					dist_to_closest_median = dist_to_median
					closest_median_id = median_id
			if closest_median_id != closest_median_ids[label_id]:
				closest_median_ids[label_id] = closest_median_id
				clusters_modified = True
				
		# Return true if the clusters were modified.
		return clusters_modified
	
	
	def _add_node_to_median(self, best_config, best_label, median):
		# Update the median.
		nb_nodes_median = nx.number_of_nodes(median)
		median.add_node(nb_nodes_median, **best_label)
		
		# Update the node maps.
		for graph_id, node_map in self._node_maps_from_median.items():
			node_map_as_rel = []
			node_map.as_relation(node_map_as_rel)
			new_node_map = NodeMap(nx.number_of_nodes(median), node_map.num_target_nodes())
			for assignment in node_map_as_rel:
				new_node_map.add_assignment(assignment[0], assignment[1])
			new_node_map.add_assignment(nx.number_of_nodes(median) - 1, best_config[graph_id])
			self._node_maps_from_median[graph_id] = new_node_map
			
		# Increase overall number of increases.
		self._num_increase_order += 1
				
	
	def _are_graphs_equal(self, g1, g2):
		"""
		Check if the two graphs are equal.

		Parameters
		----------
		g1 : NetworkX graph object
			Graph 1 to be compared.
		
		g2 : NetworkX graph object
			Graph 2 to be compared.

		Returns
		-------
		bool
			True if the two graph are equal.
			
		Notes
		-----
		This is not an identical check. Here the two graphs are equal if and only if their original_node_ids, nodes, all node labels, edges and all edge labels are equal. This function is specifically designed for class `MedianGraphEstimator` and should not be used elsewhere.
		"""
		# check original node ids.
		if not g1.graph['original_node_ids'] == g2.graph['original_node_ids']:
			return False
		# check nodes.
		nlist1 = [n for n in g1.nodes(data=True)]
		nlist2 = [n for n in g2.nodes(data=True)]
		if not nlist1 == nlist2:
			return False
		# check edges.
		elist1 = [n for n in g1.edges(data=True)]
		elist2 = [n for n in g2.edges(data=True)]
		if not elist1 == elist2:
			return False

		return True
	
	
	def compute_my_cost(g, h, node_map):
		cost = 0.0
		for node in g.nodes:
			cost += 0
			
			
	def set_label_names(self, node_labels=[], edge_labels=[], node_attrs=[], edge_attrs=[]):
		self._label_names = {'node_labels': node_labels, 'edge_labels': edge_labels,
						'node_attrs': node_attrs, 'edge_attrs': edge_attrs}
			
	
	def _get_median_node_label(self, node_labels):
		if len(self._label_names['node_labels']) > 0:
			return self._get_median_label_symbolic(node_labels)
		elif len(self._label_names['node_attrs']) > 0:
			return self._get_median_label_nonsymbolic(node_labels)
		else:
			raise Exception('Node label names are not given.')
		
			
	def _get_median_edge_label(self, edge_labels):
		if len(self._label_names['edge_labels']) > 0:
			return self._get_median_label_symbolic(edge_labels)
		elif len(self._label_names['edge_attrs']) > 0:
			return self._get_median_label_nonsymbolic(edge_labels)
		else:
			raise Exception('Edge label names are not given.')
			
			
	def _get_median_label_symbolic(self, labels):
		# Construct histogram.
		hist = {}
		for label in labels:
			label = tuple([kv for kv in label.items()]) # @todo: this may be slow.
			if label not in hist:
				hist[label] = 1
			else:
				hist[label] += 1
		
		# Return the label that appears most frequently.
		best_count = 0
		median_label = {}
		for label, count in hist.items():
			if count > best_count:
				best_count = count
				median_label = {kv[0]: kv[1] for kv in label}
				
		return median_label
		
		
	def _get_median_label_nonsymbolic(self, labels):
		if len(labels) == 0:
			return {} # @todo
		else:
			# Transform the labels into coordinates and compute mean label as initial solution.
			labels_as_coords = []
			sums = {}
			for key, val in labels[0].items():
				sums[key] = 0
			for label in labels:
				coords = {}
				for key, val in label.items():
					label_f = float(val)
					sums[key] += label_f
					coords[key] = label_f
				labels_as_coords.append(coords)
			median = {}
			for key, val in sums.items():
				median[key] = val / len(labels)
				
			# Run main loop of Weiszfeld's Algorithm.
			epsilon = 0.0001
			delta = 1.0
			num_itrs = 0
			all_equal = False
			while ((delta > epsilon) and (num_itrs < 100) and (not all_equal)):
				numerator = {}
				for key, val in sums.items():
					numerator[key] = 0
				denominator = 0
				for label_as_coord in labels_as_coords:
					norm = 0
					for key, val in label_as_coord.items():
						norm += (val - median[key]) ** 2
					norm = np.sqrt(norm)
					if norm > 0:
						for key, val in label_as_coord.items():
							numerator[key] += val / norm
						denominator += 1.0 / norm
				if denominator == 0:
					all_equal = True
				else:
					new_median = {}
					delta = 0.0
					for key, val in numerator.items():
						this_median = val / denominator
						new_median[key] = this_median
						delta += np.abs(median[key] - this_median)
					median = new_median
				
				num_itrs += 1
				
			# Transform the solution to strings and return it.
			median_label = {}
			for key, val in median.items():
				median_label[key] = str(val)
			return median_label

		
#	def _get_median_edge_label_symbolic(self, edge_labels):
#		pass
	
	
#	def _get_median_edge_label_nonsymbolic(self, edge_labels):
#		if len(edge_labels) == 0:
#			return {}
#		else:
#			# Transform the labels into coordinates and compute mean label as initial solution.
#			edge_labels_as_coords = []
#			sums = {}
#			for key, val in edge_labels[0].items():
#				sums[key] = 0
#			for edge_label in edge_labels:
#				coords = {}
#				for key, val in edge_label.items():
#					label = float(val)
#					sums[key] += label
#					coords[key] = label
#				edge_labels_as_coords.append(coords)
#			median = {}
#			for key, val in sums.items():
#				median[key] = val / len(edge_labels)
#				
#			# Run main loop of Weiszfeld's Algorithm.
#			epsilon = 0.0001
#			delta = 1.0
#			num_itrs = 0
#			all_equal = False
#			while ((delta > epsilon) and (num_itrs < 100) and (not all_equal)):
#				numerator = {}
#				for key, val in sums.items():
#					numerator[key] = 0
#				denominator = 0
#				for edge_label_as_coord in edge_labels_as_coords:
#					norm = 0
#					for key, val in edge_label_as_coord.items():
#						norm += (val - median[key]) ** 2
#					norm += np.sqrt(norm)
#					if norm > 0:
#						for key, val in edge_label_as_coord.items():
#							numerator[key] += val / norm
#						denominator += 1.0 / norm
#				if denominator == 0:
#					all_equal = True
#				else:
#					new_median = {}
#					delta = 0.0
#					for key, val in numerator.items():
#						this_median = val / denominator
#						new_median[key] = this_median
#						delta += np.abs(median[key] - this_median)
#					median = new_median
#					
#				num_itrs += 1
#				
#			# Transform the solution to ged::GXLLabel and return it.
#			median_label = {}
#			for key, val in median.items():
#				median_label[key] = str(val)
#			return median_label


def _compute_medoid_parallel(graph_ids, sort, itr):
	g_id = itr[0]
	i = itr[1]
	# @todo: timer not considered here.
# 			if timer.expired():
# 				self._state = AlgorithmState.CALLED
# 				break
	nb_nodes_g = G_ged_env.get_graph_num_nodes(g_id)
	sum_of_distances = 0
	for h_id in graph_ids:
		nb_nodes_h = G_ged_env.get_graph_num_nodes(h_id)
		if nb_nodes_g <= nb_nodes_h or not sort:
			G_ged_env.run_method(g_id, h_id)
			sum_of_distances += G_ged_env.get_upper_bound(g_id, h_id)
		else:
			G_ged_env.run_method(h_id, g_id)
			sum_of_distances += G_ged_env.get_upper_bound(h_id, g_id)
	return i, sum_of_distances
				

def _compute_init_node_maps_parallel(gen_median_id, sort, nb_nodes_median, itr):
	graph_id = itr
	nb_nodes_g = G_ged_env.get_graph_num_nodes(graph_id)
	if nb_nodes_median <= nb_nodes_g or not sort:
		G_ged_env.run_method(gen_median_id, graph_id)
		node_map = G_ged_env.get_node_map(gen_median_id, graph_id)
# 				print(self._node_maps_from_median[graph_id])
	else:
		G_ged_env.run_method(graph_id, gen_median_id)
		node_map = G_ged_env.get_node_map(graph_id, gen_median_id)
		node_map.forward_map, node_map.backward_map = node_map.backward_map, node_map.forward_map
	sum_of_distance = node_map.induced_cost()
# 				print(self._sum_of_distances)
	return graph_id, sum_of_distance, node_map
					

def _update_node_maps_parallel(median_id, epsilon, sort, nb_nodes_median, itr):
	graph_id = itr[0]
	node_map = itr[1]

	node_maps_were_modified = False
	nb_nodes_g = G_ged_env.get_graph_num_nodes(graph_id)
	if nb_nodes_median <= nb_nodes_g or not sort:
		G_ged_env.run_method(median_id, graph_id)
		if G_ged_env.get_upper_bound(median_id, graph_id) < node_map.induced_cost() - epsilon:
			node_map = G_ged_env.get_node_map(median_id, graph_id)
			node_maps_were_modified = True			
	else:
		G_ged_env.run_method(graph_id, median_id)
		if G_ged_env.get_upper_bound(graph_id, median_id) < node_map.induced_cost() - epsilon:
			node_map = G_ged_env.get_node_map(graph_id, median_id)
			node_map.forward_map, node_map.backward_map = node_map.backward_map, node_map.forward_map
			node_maps_were_modified = True	
			
	return graph_id, node_map, node_maps_were_modified