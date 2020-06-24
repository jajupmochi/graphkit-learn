#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:02:36 2020

@author: ljia
"""
import numpy as np
import networkx as nx
from gklearn.ged.env import Options, OptionsStringMap
from gklearn.ged.env import GEDData


class GEDEnv(object):

	
	def __init__(self):
		self.__initialized = False
		self.__new_graph_ids = []
		self.__ged_data = GEDData()
		# Variables needed for approximating ged_instance_.
		self.__lower_bounds = {}
		self.__upper_bounds = {}
		self.__runtimes = {}
		self.__node_maps = {}
		self.__original_to_internal_node_ids = []
		self.__internal_to_original_node_ids = []
		self.__ged_method = None
	
		
	def set_edit_cost(self, edit_cost, edit_cost_constants=[]):
		"""
	/*!
	 * @brief Sets the edit costs to one of the predefined edit costs.
	 * @param[in] edit_costs Select one of the predefined edit costs.
	 * @param[in] edit_cost_constants Constants passed to the constructor of the edit cost class selected by @p edit_costs.
	 */
		"""
		self.__ged_data._set_edit_cost(edit_cost, edit_cost_constants)
		
		
	def add_graph(self, graph_name='', graph_class=''):
		"""
	/*!
	 * @brief Adds a new uninitialized graph to the environment. Call init() after calling this method.
	 * @param[in] graph_name The name of the added graph. Empty if not specified.
	 * @param[in] graph_class The class of the added graph. Empty if not specified.
	 * @return The ID of the newly added graph.
	 */
		"""
		# @todo: graphs are not uninitialized.
		self.__initialized = False
		graph_id = self.__ged_data._num_graphs_without_shuffled_copies
		self.__ged_data._num_graphs_without_shuffled_copies += 1
		self.__new_graph_ids.append(graph_id)
		self.__ged_data._graphs.append(nx.Graph())
		self.__ged_data._graph_names.append(graph_name)
		self.__ged_data._graph_classes.append(graph_class)
		self.__original_to_internal_node_ids.append({})
		self.__internal_to_original_node_ids.append({})
		self.__ged_data._strings_to_internal_node_ids.append({})
		self.__ged_data._internal_node_ids_to_strings.append({})
		return graph_id
	
	
	def add_node(self, graph_id, node_id, node_label):
		"""
	/*!
	 * @brief Adds a labeled node.
	 * @param[in] graph_id ID of graph that has been added to the environment.
	 * @param[in] node_id The user-specific ID of the vertex that has to be added.
	 * @param[in] node_label The label of the vertex that has to be added. Set to ged::NoLabel() if template parameter @p UserNodeLabel equals ged::NoLabel.
	 */
		"""
		# @todo: check ids.
		self.__initialized = False
		internal_node_id = nx.number_of_nodes(self.__ged_data._graphs[graph_id])
		self.__ged_data._graphs[graph_id].add_node(internal_node_id, label=node_label)
		self.__original_to_internal_node_ids[graph_id][node_id] = internal_node_id
		self.__internal_to_original_node_ids[graph_id][internal_node_id] = node_id
		self.__ged_data._strings_to_internal_node_ids[graph_id][str(node_id)] = internal_node_id
		self.__ged_data._internal_node_ids_to_strings[graph_id][internal_node_id] = str(node_id)
		# @todo: node_label_to_id_
		
		
	def add_edge(self, graph_id, nd_from, nd_to, edge_label, ignore_duplicates=True):
		"""
	/*!
	 * @brief Adds a labeled edge.
	 * @param[in] graph_id ID of graph that has been added to the environment.
	 * @param[in] tail The user-specific ID of the tail of the edge that has to be added.
	 * @param[in] head The user-specific ID of the head of the edge that has to be added.
	 * @param[in] edge_label The label of the vertex that has to be added. Set to ged::NoLabel() if template parameter @p UserEdgeLabel equals ged::NoLabel.
	 * @param[in] ignore_duplicates If @p true, duplicate edges are ignores. Otherwise, an exception is thrown if an existing edge is added to the graph.
	 */
		"""
		# @todo: check everything.
		self.__initialized = False
		# @todo: check ignore_duplicates.
		self.__ged_data._graphs[graph_id].add_edge(self.__original_to_internal_node_ids[graph_id][nd_from], self.__original_to_internal_node_ids[graph_id][nd_to], label=edge_label)
		# @todo: edge_id and label_id, edge_label_to_id_.
		
	
	def add_nx_graph(self, g, classe, ignore_duplicates=True) :
		"""
			Add a Graph (made by networkx) on the environment. Be careful to respect the same format as GXL graphs for labelling nodes and edges. 
	
			:param g: The graph to add (networkx graph)
			:param ignore_duplicates: If True, duplicate edges are ignored, otherwise it's raise an error if an existing edge is added. True by default
			:type g: networkx.graph
			:type ignore_duplicates: bool
			:return: The ID of the newly added graphe
			:rtype: size_t
	
			.. note:: The NX graph must respect the GXL structure. Please see how a GXL graph is construct.  
			
		"""
		graph_id = self.add_graph(g.name, classe) # check if the graph name already exists.
		for node in g.nodes: # @todo: if the keys of labels include int and str at the same time.
			self.add_node(graph_id, node, tuple(sorted(g.nodes[node].items(), key=lambda kv: kv[0])))
		for edge in g.edges:
			self.add_edge(graph_id, edge[0], edge[1], tuple(sorted(g.edges[(edge[0], edge[1])].items(), key=lambda kv: kv[0])), ignore_duplicates)
		return graph_id
	
	
	def init(self, init_type=Options.InitType.EAGER_WITHOUT_SHUFFLED_COPIES, print_to_stdout=False):
		if isinstance(init_type, str):
			init_type = OptionsStringMap.InitType[init_type]
			
		# Throw an exception if no edit costs have been selected.
		if self.__ged_data._edit_cost is None:
			raise Exception('No edit costs have been selected. Call set_edit_cost() before calling init().')
			
		# Return if the environment is initialized.
		if self.__initialized:
			return
		
		# Set initialization type.
		self.__ged_data._init_type = init_type
		
		# @todo: Construct shuffled graph copies if necessary.
		
		# Re-initialize adjacency matrices (also previously initialized graphs must be re-initialized because of possible re-allocation).
		# @todo: setup_adjacency_matrix, don't know if neccessary.
		self.__ged_data._max_num_nodes = np.max([nx.number_of_nodes(g) for g in self.__ged_data._graphs])
		self.__ged_data._max_num_edges = np.max([nx.number_of_edges(g) for g in self.__ged_data._graphs])
			
		# Initialize cost matrices if necessary.
		if self.__ged_data._eager_init():
			pass # @todo: init_cost_matrices_: 1. Update node cost matrix if new node labels have been added to the environment; 2. Update edge cost matrix if new edge labels have been added to the environment.
			
		# Mark environment as initialized.
		self.__initialized = True
		self.__new_graph_ids.clear()
		
		
	def set_method(self, method, options=''):
		"""
	/*!
	 * @brief Sets the GEDMethod to be used by run_method().
	 * @param[in] method Select the method that is to be used.
	 * @param[in] options An options string of the form @"[--@<option@> @<arg@>] [...]@" passed to the selected method.
	 */
		"""
		del self.__ged_method
		
		if isinstance(method, str):
			method = OptionsStringMap.GEDMethod[method]

		if method == Options.GEDMethod.BRANCH:
			self.__ged_method = Branch(self.__ged_data)
		elif method == Options.GEDMethod.BRANCH_FAST:
			self.__ged_method = BranchFast(self.__ged_data)
		elif method == Options.GEDMethod.BRANCH_FAST:
			self.__ged_method = BranchFast(self.__ged_data)	
		elif method == Options.GEDMethod.BRANCH_TIGHT:
			self.__ged_method = BranchTight(self.__ged_data)	
		elif method == Options.GEDMethod.BRANCH_UNIFORM:
			self.__ged_method = BranchUniform(self.__ged_data)	
		elif method == Options.GEDMethod.BRANCH_COMPACT:
			self.__ged_method = BranchCompact(self.__ged_data)	
		elif method == Options.GEDMethod.PARTITION:
			self.__ged_method = Partition(self.__ged_data)	
		elif method == Options.GEDMethod.HYBRID:
			self.__ged_method = Hybrid(self.__ged_data)	
		elif method == Options.GEDMethod.RING:
			self.__ged_method = Ring(self.__ged_data)	
		elif method == Options.GEDMethod.ANCHOR_AWARE_GED:
			self.__ged_method = AnchorAwareGED(self.__ged_data)	
		elif method == Options.GEDMethod.WALKS:
			self.__ged_method = Walks(self.__ged_data)	
		elif method == Options.GEDMethod.IPFP:
			self.__ged_method = IPFP(self.__ged_data)	
		elif method == Options.GEDMethod.BIPARTITE:
			from gklearn.ged.methods import Bipartite
			self.__ged_method = Bipartite(self.__ged_data)	
		elif method == Options.GEDMethod.SUBGRAPH:
			self.__ged_method = Subgraph(self.__ged_data)	
		elif method == Options.GEDMethod.NODE:
			self.__ged_method = Node(self.__ged_data)	
		elif method == Options.GEDMethod.RING_ML:
			self.__ged_method = RingML(self.__ged_data)	
		elif method == Options.GEDMethod.BIPARTITE_ML:
			self.__ged_method = BipartiteML(self.__ged_data)	
		elif method == Options.GEDMethod.REFINE:
			self.__ged_method = Refine(self.__ged_data)	
		elif method == Options.GEDMethod.BP_BEAM:
			self.__ged_method = BPBeam(self.__ged_data)	
		elif method == Options.GEDMethod.SIMULATED_ANNEALING:
			self.__ged_method = SimulatedAnnealing(self.__ged_data)	
		elif method == Options.GEDMethod.HED:
			self.__ged_method = HED(self.__ged_data)	
		elif method == Options.GEDMethod.STAR:
			self.__ged_method = STAR(self.__ged_data)	
		# #ifdef GUROBI
		elif method == Options.GEDMethod.F1:
			self.__ged_method = F1(self.__ged_data)	
		elif method == Options.GEDMethod.F2:
			self.__ged_method = F2(self.__ged_data)	
		elif method == Options.GEDMethod.COMPACT_MIP:
			self.__ged_method = CompactMIP(self.__ged_data)	
		elif method == Options.GEDMethod.BLP_NO_EDGE_LABELS:
			self.__ged_method = BLPNoEdgeLabels(self.__ged_data)	

		self.__ged_method.set_options(options)
		
		
	def run_method(self, g_id, h_id):
		"""
	/*!
	 * @brief Runs the GED method specified by call to set_method() between the graphs with IDs @p g_id and @p h_id.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 */
		"""
		if g_id >= self.__ged_data.num_graphs():
			raise Exception('The graph with ID', str(g_id), 'has not been added to the environment.')
		if h_id >= self.__ged_data.num_graphs():
			raise Exception('The graph with ID', str(h_id), 'has not been added to the environment.')
		if not self.__initialized:
			raise Exception('The environment is uninitialized. Call init() after adding all graphs to the environment.')
		if self.__ged_method is None:
			raise Exception('No method has been set. Call set_method() before calling run().')
		
		# Call selected GEDMethod and store results.
		if self.__ged_data.shuffled_graph_copies_available() and (g_id == h_id):
			self.__ged_method.run(g_id, self.__ged_data.id_shuffled_graph_copy(h_id)) # @todo: why shuffle?
		else:
			self.__ged_method.run(g_id, h_id)
		self.__lower_bounds[(g_id, h_id)] = self.__ged_method.get_lower_bound()
		self.__upper_bounds[(g_id, h_id)] = self.__ged_method.get_upper_bound()
		self.__runtimes[(g_id, h_id)] = self.__ged_method.get_runtime()
		self.__node_maps[(g_id, h_id)] = self.__ged_method.get_node_map()
		
		
	def init_method(self):
		"""Initializes the method specified by call to set_method().
		"""
		if not self.__initialized:
			raise Exception('The environment is uninitialized. Call init() before calling init_method().')
		if self.__ged_method is None:
			raise Exception('No method has been set. Call set_method() before calling init_method().')
		self.__ged_method.init()
		
		
	def get_upper_bound(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns upper bound for edit distance between the input graphs.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Upper bound computed by the last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self.__upper_bounds:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_upper_bound(' + str(g_id) + ',' + str(h_id) + ').')
		return self.__upper_bounds[(g_id, h_id)]
		
		
	def get_lower_bound(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns lower bound for edit distance between the input graphs.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Lower bound computed by the last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self.__lower_bounds:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_lower_bound(' + str(g_id) + ',' + str(h_id) + ').')
		return self.__lower_bounds[(g_id, h_id)]
		
		
	def get_runtime(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns runtime.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Runtime of last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self.__runtimes:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_runtime(' + str(g_id) + ',' + str(h_id) + ').')
		return self.__runtimes[(g_id, h_id)]
	

	def get_init_time(self):
		"""
	/*!
	 * @brief Returns initialization time.
	 * @return Runtime of the last call to init_method().
	 */
		"""		
		return self.__ged_method.get_init_time()


	def get_node_map(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns node map between the input graphs.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Node map computed by the last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self.__node_maps:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_node_map(' + str(g_id) + ',' + str(h_id) + ').')
		return self.__node_maps[(g_id, h_id)]
	

	def get_forward_map(self, g_id, h_id) :
		"""
			Returns the forward map (or the half of the adjacence matrix) between nodes of the two indicated graphs. 
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The forward map to the adjacence matrix between nodes of the two graphs
			:rtype: list[npy_uint32]
			
			.. seealso:: run_method(), get_upper_bound(), get_lower_bound(), get_backward_map(), get_runtime(), quasimetric_cost(), get_node_map(), get_assignment_matrix()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: I don't know how to connect the two map to reconstruct the adjacence matrix. Please come back when I know how it's work ! 
		"""
		return self.get_node_map(g_id, h_id).forward_map
	
	
	def get_backward_map(self, g_id, h_id) :
		"""
			Returns the backward map (or the half of the adjacence matrix) between nodes of the two indicated graphs. 
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The backward map to the adjacence matrix between nodes of the two graphs
			:rtype: list[npy_uint32]
			
			.. seealso:: run_method(), get_upper_bound(), get_lower_bound(),  get_forward_map(), get_runtime(), quasimetric_cost(), get_node_map(), get_assignment_matrix()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: I don't know how to connect the two map to reconstruct the adjacence matrix. Please come back when I know how it's work ! 
		"""
		return self.get_node_map(g_id, h_id).backward_map
		
		
	def get_all_graph_ids(self):
		return [i for i in range(0, self.__ged_data._num_graphs_without_shuffled_copies)]