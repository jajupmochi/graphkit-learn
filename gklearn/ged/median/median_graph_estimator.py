#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:04:55 2020

@author: ljia
"""
import numpy as np
from gklearn.ged.env import AlgorithmState
from gklearn.ged.util import misc
from gklearn.utils import Timer
import time
from tqdm import tqdm
import sys
import networkx as nx


class MedianGraphEstimator(object):
	
	def __init__(self, ged_env, constant_node_costs):
		"""Constructor.
		
		Parameters
		----------
		ged_env : gklearn.gedlib.gedlibpy.GEDEnv
			Initialized GED environment. The edit costs must be set by the user.
			
		constant_node_costs : Boolean
			Set to True if the node relabeling costs are constant.
		"""
		self.__ged_env = ged_env
		self.__init_method = 'BRANCH_FAST'
		self.__init_options = ''
		self.__descent_method = 'BRANCH_FAST'
		self.__descent_options = ''
		self.__refine_method = 'IPFP'
		self.__refine_options = ''
		self.__constant_node_costs = constant_node_costs
		self.__labeled_nodes = (ged_env.get_num_node_labels() > 1)
		self.__node_del_cost = ged_env.get_node_del_cost(ged_env.get_node_label(1))
		self.__node_ins_cost = ged_env.get_node_ins_cost(ged_env.get_node_label(1))
		self.__labeled_edges = (ged_env.get_num_edge_labels() > 1)
		self.__edge_del_cost = ged_env.get_edge_del_cost(ged_env.get_edge_label(1))
		self.__edge_ins_cost = ged_env.get_edge_ins_cost(ged_env.get_edge_label(1))
		self.__init_type = 'RANDOM'
		self.__num_random_inits = 10
		self.__desired_num_random_inits = 10
		self.__use_real_randomness = True
		self.__seed = 0
		self.__refine = True
		self.__time_limit_in_sec = 0
		self.__epsilon = 0.0001
		self.__max_itrs = 100
		self.__max_itrs_without_update = 3
		self.__num_inits_increase_order = 10
		self.__init_type_increase_order = 'K-MEANS++'
		self.__max_itrs_increase_order = 10
		self.__print_to_stdout = 2
		self.__median_id = np.inf # @todo: check
		self.__median_node_id_prefix = '' # @todo: check
		self.__node_maps_from_median = {}
		self.__sum_of_distances = 0
		self.__best_init_sum_of_distances = np.inf
		self.__converged_sum_of_distances = np.inf
		self.__runtime = None
		self.__runtime_initialized = None
		self.__runtime_converged = None
		self.__itrs = [] # @todo: check: {} ?
		self.__num_decrease_order = 0
		self.__num_increase_order = 0
		self.__num_converged_descents = 0
		self.__state = AlgorithmState.TERMINATED
		
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
		self.__set_default_options()
		options_map = misc.options_string_to_options_map(options)
		for opt_name, opt_val in options_map.items():
			if opt_name == 'init-type':
				self.__init_type = opt_val
				if opt_val != 'MEDOID' and opt_val != 'RANDOM' and opt_val != 'MIN' and opt_val != 'MAX' and opt_val != 'MEAN':
					raise Exception('Invalid argument ' + opt_val + ' for option init-type. Usage: options = "[--init-type RANDOM|MEDOID|EMPTY|MIN|MAX|MEAN] [...]"')
			elif opt_name == 'random-inits':
				try:
					self.__num_random_inits = int(opt_val)
					self.__desired_num_random_inits = self.__num_random_inits
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option random-inits. Usage: options = "[--random-inits <convertible to int greater 0>]"')

				if self.__num_random_inits <= 0:
					raise Exception('Invalid argument "' + opt_val + '" for option random-inits. Usage: options = "[--random-inits <convertible to int greater 0>]"')
	
			elif opt_name == 'randomness':
				if opt_val == 'PSEUDO':
					self.__use_real_randomness = False
	
				elif opt_val == 'REAL':
					self.__use_real_randomness = True
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option randomness. Usage: options = "[--randomness REAL|PSEUDO] [...]"')
	
			elif opt_name == 'stdout':
				if opt_val == '0':
					self.__print_to_stdout = 0
	
				elif opt_val == '1':
					self.__print_to_stdout = 1
	
				elif opt_val == '2':
					self.__print_to_stdout = 2
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option stdout. Usage: options = "[--stdout 0|1|2] [...]"')
	
			elif opt_name == 'refine':
				if opt_val == 'TRUE':
					self.__refine = True
	
				elif opt_val == 'FALSE':
					self.__refine = False
	
				else:
					raise Exception('Invalid argument "' + opt_val  + '" for option refine. Usage: options = "[--refine TRUE|FALSE] [...]"')
	
			elif opt_name == 'time-limit':
				try:
					self.__time_limit_in_sec = float(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option time-limit.  Usage: options = "[--time-limit <convertible to double>] [...]')
	
			elif opt_name == 'max-itrs':
				try:
					self.__max_itrs = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option max-itrs. Usage: options = "[--max-itrs <convertible to int>] [...]')
	
			elif opt_name == 'max-itrs-without-update':
				try:
					self.__max_itrs_without_update = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option max-itrs-without-update. Usage: options = "[--max-itrs-without-update <convertible to int>] [...]')
	
			elif opt_name == 'seed':
				try:
					self.__seed = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option seed. Usage: options = "[--seed <convertible to int greater equal 0>] [...]')
	
			elif opt_name == 'epsilon':
				try:
					self.__epsilon = float(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option epsilon. Usage: options = "[--epsilon <convertible to double greater 0>] [...]')
	
				if self.__epsilon <= 0:
					raise Exception('Invalid argument "' + opt_val + '" for option epsilon. Usage: options = "[--epsilon <convertible to double greater 0>] [...]')
	
			elif opt_name == 'inits-increase-order':
				try:
					self.__num_inits_increase_order = int(opt_val)
	
				except:
					raise Exception('Invalid argument "' + opt_val + '" for option inits-increase-order. Usage: options = "[--inits-increase-order <convertible to int greater 0>]"')
	
				if self.__num_inits_increase_order <= 0:
					raise Exception('Invalid argument "' + opt_val + '" for option inits-increase-order. Usage: options = "[--inits-increase-order <convertible to int greater 0>]"')

			elif opt_name == 'init-type-increase-order':
				self.__init_type_increase_order = opt_val
				if opt_val != 'CLUSTERS' and opt_val != 'K-MEANS++':
					raise Exception('Invalid argument ' + opt_val + ' for option init-type-increase-order. Usage: options = "[--init-type-increase-order CLUSTERS|K-MEANS++] [...]"')
	
			elif opt_name == 'max-itrs-increase-order':
				try:
					self.__max_itrs_increase_order = int(opt_val)
	
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
		self.__init_method = init_method;
		self.__init_options = init_options;
	
	
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
		self.__descent_method = descent_method;
		self.__descent_options = descent_options;

	
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
		self.__refine_method = refine_method
		self.__refine_options = refine_options

	
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
			if self.__ged_env.get_graph_num_nodes(graph_id) > 0:
				self.__median_node_id_prefix = self.__ged_env.get_original_node_ids(graph_id)[0]
				all_graphs_empty = False
				break
		if all_graphs_empty:
			raise Exception('All graphs in the collection are empty.')
			
		# Start timer and record start time.
		start = time.time()
		timer = Timer(self.__time_limit_in_sec)
		self.__median_id = gen_median_id
		self.__state = AlgorithmState.TERMINATED
		
		# Get ExchangeGraph representations of the input graphs.
		graphs = {}
		for graph_id in graph_ids:
			# @todo: get_nx_graph() function may need to be modified according to the coming code.
			graphs[graph_id] = self.__ged_env.get_nx_graph(graph_id, True, True, False)
# 		print(self.__ged_env.get_graph_internal_id(0))
# 		print(graphs[0].graph)
# 		print(graphs[0].nodes(data=True))
# 		print(graphs[0].edges(data=True))
# 		print(nx.adjacency_matrix(graphs[0]))

			
		# Construct initial medians.
		medians = []
		self.__construct_initial_medians(graph_ids, timer, medians)
		end_init = time.time()
		self.__runtime_initialized = end_init - start
# 		print(medians[0].graph)
# 		print(medians[0].nodes(data=True))
# 		print(medians[0].edges(data=True))
# 		print(nx.adjacency_matrix(medians[0]))
		
		# Reset information about iterations and number of times the median decreases and increases.
		self.__itrs = [0] * len(medians)
		self.__num_decrease_order = 0
		self.__num_increase_order = 0
		self.__num_converged_descents = 0
		
		# Initialize the best median.
		best_sum_of_distances = np.inf
		self.__best_init_sum_of_distances = np.inf
		node_maps_from_best_median = {}
		
		# Run block gradient descent from all initial medians.
		self.__ged_env.set_method(self.__descent_method, self.__descent_options)
		for median_pos in range(0, len(medians)):
			
			# Terminate if the timer has expired and at least one SOD has been computed.
			if timer.expired() and median_pos > 0:
				break
			
			# Print information about current iteration.
			if self.__print_to_stdout == 2:
				print('\n===========================================================')
				print('Block gradient descent for initial median', str(median_pos + 1), 'of', str(len(medians)), '.')
				print('-----------------------------------------------------------')
				
			# Get reference to the median.
			median = medians[median_pos]
			
			# Load initial median into the environment.
			self.__ged_env.load_nx_graph(median, gen_median_id)
			self.__ged_env.init(self.__ged_env.get_init_type())
			
			# Print information about current iteration.
			if self.__print_to_stdout == 2:
				progress = tqdm(desc='Computing initial node maps', total=len(graph_ids), file=sys.stdout)
				
			# Compute node maps and sum of distances for initial median.
			self.__sum_of_distances = 0
			self.__node_maps_from_median.clear() # @todo
			for graph_id in graph_ids:
				self.__ged_env.run_method(gen_median_id, graph_id)
				self.__node_maps_from_median[graph_id] = self.__ged_env.get_node_map(gen_median_id, graph_id)
# 				print(self.__node_maps_from_median[graph_id])
				self.__sum_of_distances += self.__ged_env.get_induced_cost(gen_median_id, graph_id) # @todo: the C++ implementation for this function in GedLibBind.ipp re-call get_node_map() once more, this is not neccessary.
# 				print(self.__sum_of_distances)
				# Print information about current iteration.
				if self.__print_to_stdout == 2:
					progress.update(1)
					
			self.__best_init_sum_of_distances = min(self.__best_init_sum_of_distances, self.__sum_of_distances)
			self.__ged_env.load_nx_graph(median, set_median_id)
# 			print(self.__best_init_sum_of_distances)
			
			# Print information about current iteration.
			if self.__print_to_stdout == 2:
				print('\n')
				
			# Run block gradient descent from initial median.
			converged = False
			itrs_without_update = 0
			while not self.__termination_criterion_met(converged, timer, self.__itrs[median_pos], itrs_without_update):
				
				# Print information about current iteration.
				if self.__print_to_stdout == 2:
					print('\n===========================================================')
					print('Iteration', str(self.__itrs[median_pos] + 1), 'for initial median', str(median_pos + 1), 'of', str(len(medians)), '.')
					print('-----------------------------------------------------------')
					
				# Initialize flags that tell us what happened in the iteration.
				median_modified = False
				node_maps_modified = False
				decreased_order = False
				increased_order = False
				
				# Update the median. # @todo!!!!!!!!!!!!!!!!!!!!!!
				median_modified = self.__update_median(graphs, median)
				if not median_modified or self.__itrs[median_pos] == 0:
					decreased_order = False
					if not decreased_order or self.__itrs[median_pos] == 0:
						increased_order = False
						
				# Update the number of iterations without update of the median.
				if median_modified or decreased_order or increased_order:
					itrs_without_update = 0
				else:
					itrs_without_update += 1
					
				# Print information about current iteration.
				if self.__print_to_stdout == 2:
					print('Loading median to environment: ... ', end='')
					
				# Load the median into the environment.
				# @todo: should this function use the original node label?
				self.__ged_env.load_nx_graph(median, gen_median_id)
				self.__ged_env.init(self.__ged_env.get_init_type())
					
				# Print information about current iteration.
				if self.__print_to_stdout == 2:
					print('done.')					
					
				# Print information about current iteration.
				if self.__print_to_stdout == 2:
					print('Updating induced costs: ... ', end='')

				# Compute induced costs of the old node maps w.r.t. the updated median.
				for graph_id in graph_ids:
# 					print(self.__ged_env.get_induced_cost(gen_median_id, graph_id))
					# @todo: watch out if compute_induced_cost is correct, this may influence: increase/decrease order, induced_cost() in the following code.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					self.__ged_env.compute_induced_cost(gen_median_id, graph_id)
# 					print('---------------------------------------')
# 					print(self.__ged_env.get_induced_cost(gen_median_id, graph_id))
					
				# Print information about current iteration.
				if self.__print_to_stdout == 2:
					print('done.')					
					
				# Update the node maps.
				node_maps_modified = self.__update_node_maps() # @todo

				# Update the order of the median if no improvement can be found with the current order.
				
				# Update the sum of distances.
				old_sum_of_distances = self.__sum_of_distances
				self.__sum_of_distances = 0
				for graph_id in self.__node_maps_from_median:
					self.__sum_of_distances += self.__ged_env.get_induced_cost(gen_median_id, graph_id) # @todo: see above.
					
				# Print information about current iteration.
				if self.__print_to_stdout == 2:
					print('Old local SOD: ', old_sum_of_distances)
					print('New local SOD: ', self.__sum_of_distances)
					print('Best converged SOD: ', best_sum_of_distances)
					print('Modified median: ', median_modified)
					print('Modified node maps: ', node_maps_modified)
					print('Decreased order: ', decreased_order)
					print('Increased order: ', increased_order)
					print('===========================================================\n')
					
				converged = not (median_modified or node_maps_modified or decreased_order or increased_order)
				
				self.__itrs[median_pos] += 1
				
			# Update the best median.
			if self.__sum_of_distances < best_sum_of_distances:
				best_sum_of_distances = self.__sum_of_distances
				node_maps_from_best_median = self.__node_maps_from_median
				best_median = median
				
			# Update the number of converged descents.
			if converged:
				self.__num_converged_descents += 1
				
		# Store the best encountered median.
		self.__sum_of_distances = best_sum_of_distances
		self.__node_maps_from_median = node_maps_from_best_median
		self.__ged_env.load_nx_graph(best_median, gen_median_id)
		self.__ged_env.init(self.__ged_env.get_init_type())
		end_descent = time.time()
		self.__runtime_converged = end_descent - start
		
		# Refine the sum of distances and the node maps for the converged median.
		self.__converged_sum_of_distances = self.__sum_of_distances
		if self.__refine:
			self.__improve_sum_of_distances(timer) # @todo
		
		# Record end time, set runtime and reset the number of initial medians.
		end = time.time()
		self.__runtime = end - start
		self.__num_random_inits = self.__desired_num_random_inits
		
		# Print global information.
		if self.__print_to_stdout != 0:
			print('\n===========================================================')
			print('Finished computation of generalized median graph.')
			print('-----------------------------------------------------------')
			print('Best SOD after initialization: ', self.__best_init_sum_of_distances)
			print('Converged SOD: ', self.__converged_sum_of_distances)
			if self.__refine:
				print('Refined SOD: ', self.__sum_of_distances)
			print('Overall runtime: ', self.__runtime)
			print('Runtime of initialization: ', self.__runtime_initialized)
			print('Runtime of block gradient descent: ', self.__runtime_converged - self.__runtime_initialized)
			if self.__refine:
				print('Runtime of refinement: ', self.__runtime - self.__runtime_converged)
			print('Number of initial medians: ', len(medians))
			total_itr = 0
			num_started_descents = 0
			for itr in self.__itrs:
				total_itr += itr
				if itr > 0:
					num_started_descents += 1
			print('Size of graph collection: ', len(graph_ids))
			print('Number of started descents: ', num_started_descents)
			print('Number of converged descents: ', self.__num_converged_descents)
			print('Overall number of iterations: ', total_itr)
			print('Overall number of times the order decreased: ', self.__num_decrease_order)
			print('Overall number of times the order increased: ', self.__num_increase_order)
			print('===========================================================\n')
	
	
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
		if not self.__median_available():
			raise Exception('No median has been computed. Call run() before calling get_sum_of_distances().')
		if state == 'initialized':
			return self.__best_init_sum_of_distances
		if state == 'converged':
			return self.__converged_sum_of_distances
		return self.__sum_of_distances
	
	
	def __set_default_options(self):
		self.__init_type = 'RANDOM'
		self.__num_random_inits = 10
		self.__desired_num_random_inits = 10
		self.__use_real_randomness = True
		self.__seed = 0
		self.__refine = True
		self.__time_limit_in_sec = 0
		self.__epsilon = 0.0001
		self.__max_itrs = 100
		self.__max_itrs_without_update = 3
		self.__num_inits_increase_order = 10
		self.__init_type_increase_order = 'K-MEANS++'
		self.__max_itrs_increase_order = 10
		self.__print_to_stdout = 2
		
		
	def __construct_initial_medians(self, graph_ids, timer, initial_medians):
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('\n===========================================================')
			print('Constructing initial median(s).')
			print('-----------------------------------------------------------')
			
		# Compute or sample the initial median(s).
		initial_medians.clear()
		if self.__init_type == 'MEDOID':
			self.__compute_medoid(graph_ids, timer, initial_medians)
		elif self.__init_type == 'MAX':
			pass # @todo
# 			compute_max_order_graph_(graph_ids, initial_medians)
		elif self.__init_type == 'MIN':
			pass # @todo
# 			compute_min_order_graph_(graph_ids, initial_medians)
		elif self.__init_type == 'MEAN':
			pass # @todo
# 			compute_mean_order_graph_(graph_ids, initial_medians)
		else:
			pass # @todo
# 			sample_initial_medians_(graph_ids, initial_medians)

		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('===========================================================')
			
			
	def __compute_medoid(self, graph_ids, timer, initial_medians):
		# Use method selected for initialization phase.
		self.__ged_env.set_method(self.__init_method, self.__init_options)
		
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			progress = tqdm(desc='Computing medoid', total=len(graph_ids), file=sys.stdout)
			
		# Compute the medoid.
		medoid_id = graph_ids[0]
		best_sum_of_distances = np.inf
		for g_id in graph_ids:
			if timer.expired():
				self.__state = AlgorithmState.CALLED
				break
			sum_of_distances = 0
			for h_id in graph_ids:
				self.__ged_env.run_method(g_id, h_id)
				sum_of_distances += self.__ged_env.get_upper_bound(g_id, h_id)
			if sum_of_distances < best_sum_of_distances:
				best_sum_of_distances = sum_of_distances
				medoid_id = g_id
				
			# Print information about current iteration.
			if self.__print_to_stdout == 2:
				progress.update(1)
		initial_medians.append(self.__ged_env.get_nx_graph(medoid_id, True, True, False)) # @todo
		
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('\n')
			
		
	def __termination_criterion_met(self, converged, timer, itr, itrs_without_update):
		if timer.expired() or (itr >= self.__max_itrs if self.__max_itrs >= 0 else False):
			if self.__state == AlgorithmState.TERMINATED:
				self.__state = AlgorithmState.INITIALIZED
			return True
		return converged or (itrs_without_update > self.__max_itrs_without_update if self.__max_itrs_without_update >= 0 else False)
	
	
	def __update_median(self, graphs, median):
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('Updating median: ', end='')
			
		# Store copy of the old median.
		old_median = median.copy() # @todo: this is just a shallow copy.
		
		# Update the node labels.
		if self.__labeled_nodes:
			self.__update_node_labels(graphs, median)
			
		# Update the edges and their labels.
		self.__update_edges(graphs, median)
		
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('done.')
			
		return not self.__are_graphs_equal(median, old_median)
		
		
	def __update_node_labels(self, graphs, median):
		
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('nodes ... ', end='')
			
		# Iterate through all nodes of the median.
		for i in range(0, nx.number_of_nodes(median)):
# 			print('i: ', i)
			# Collect the labels of the substituted nodes.
			node_labels = []
			for graph_id, graph in graphs.items():
# 				print('graph_id: ', graph_id)
# 				print(self.__node_maps_from_median[graph_id])
				k = self.__get_node_image_from_map(self.__node_maps_from_median[graph_id], i)
# 				print('k: ', k)
				if k != np.inf:
					node_labels.append(graph.nodes[k])
					
			# Compute the median label and update the median.
			if len(node_labels) > 0:
				median_label = self.__ged_env.get_median_node_label(node_labels)
				if self.__ged_env.get_node_rel_cost(median.nodes[i], median_label) > self.__epsilon:
					nx.set_node_attributes(median, {i: median_label})
					
					
	def __update_edges(self, graphs, median):
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('edges ... ', end='')
			
		# Clear the adjacency lists of the median and reset number of edges to 0.
		median_edges = list(median.edges)		
		for (head, tail) in median_edges:
			median.remove_edge(head, tail)
		
		# @todo: what if edge is not labeled?
		# Iterate through all possible edges (i,j) of the median.
		for i in range(0, nx.number_of_nodes(median)):
			for j in range(i + 1, nx.number_of_nodes(median)):
				
				# Collect the labels of the edges to which (i,j) is mapped by the node maps.
				edge_labels = []
				for graph_id, graph in graphs.items():
					k = self.__get_node_image_from_map(self.__node_maps_from_median[graph_id], i)
					l = self.__get_node_image_from_map(self.__node_maps_from_median[graph_id], j)
					if k != np.inf and l != np.inf:
						if graph.has_edge(k, l):
							edge_labels.append(graph.edges[(k, l)])
							
				# Compute the median edge label and the overall edge relabeling cost.
				rel_cost = 0
				median_label = self.__ged_env.get_edge_label(1)
				if median.has_edge(i, j):
					median_label = median.edges[(i, j)]
				if self.__labeled_edges and len(edge_labels) > 0:
					new_median_label = self.__ged_env.median_edge_label(edge_labels)
					if self.__ged_env.get_edge_rel_cost(median_label, new_median_label) > self.__epsilon:
						median_label = new_median_label
					for edge_label in edge_labels:
						rel_cost += self.__ged_env.get_edge_rel_cost(median_label, edge_label)
						
				# Update the median.
				if rel_cost < (self.__edge_ins_cost + self.__edge_del_cost) * len(edge_labels) - self.__edge_del_cost * len(graphs):
					median.add_edge(i, j, **median_label)
				else:
					if median.has_edge(i, j):
						median.remove_edge(i, j)


	def __update_node_maps(self):
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			progress = tqdm(desc='Updating node maps', total=len(self.__node_maps_from_median), file=sys.stdout)
			
		# Update the node maps.
		node_maps_were_modified = False
		for graph_id in self.__node_maps_from_median:
			self.__ged_env.run_method(self.__median_id, graph_id)
			if self.__ged_env.get_upper_bound(self.__median_id, graph_id) < self.__ged_env.get_induced_cost(self.__median_id, graph_id) - self.__epsilon: # @todo: see above.
				self.__node_maps_from_median[graph_id] = self.__ged_env.get_node_map(self.__median_id, graph_id) # @todo: node_map may not assigned.
				node_maps_were_modified = True
			# Print information about current iteration.
			if self.__print_to_stdout == 2:
				progress.update(1)
			
		# Print information about current iteration.
		if self.__print_to_stdout == 2:
			print('\n')
			
		# Return true if the node maps were modified.
		return node_maps_were_modified
	
	
	def __improve_sum_of_distances(self, timer):
		pass
	
	
	def __median_available(self):
		return self.__median_id != np.inf
		
				
	def __get_node_image_from_map(self, node_map, node):
		"""
		Return ID of the node mapping of `node` in `node_map`.

		Parameters
		----------
		node_map : list[tuple(int, int)]
			List of node maps where the mapping node is found.
		
		node : int
			The mapping node of this node is returned

		Raises
		------
		Exception
			If the node with ID `node` is not contained in the source nodes of the node map.

		Returns
		-------
		int
			ID of the mapping of `node`.
			
		Notes
		-----
		This function is not implemented in the `ged::MedianGraphEstimator` class of the `GEDLIB` library. Instead it is a Python implementation of the `ged::NodeMap::image` function.
		"""
		if node < len(node_map):
			return node_map[node][1] if node_map[node][1] < len(node_map) else np.inf
		else:
 			raise Exception('The node with ID ', str(node), ' is not contained in the source nodes of the node map.')
		return np.inf
				
	
	def __are_graphs_equal(self, g1, g2):
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