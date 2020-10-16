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
		self._initialized = False
		self._new_graph_ids = []
		self._ged_data = GEDData()
		# Variables needed for approximating ged_instance_.
		self._lower_bounds = {}
		self._upper_bounds = {}
		self._runtimes = {}
		self._node_maps = {}
		self._original_to_internal_node_ids = []
		self._internal_to_original_node_ids = []
		self._ged_method = None
	
		
	def set_edit_cost(self, edit_cost, edit_cost_constants=[]):
		"""
	/*!
	 * @brief Sets the edit costs to one of the predefined edit costs.
	 * @param[in] edit_costs Select one of the predefined edit costs.
	 * @param[in] edit_cost_constants Constants passed to the constructor of the edit cost class selected by @p edit_costs.
	 */
		"""
		self._ged_data._set_edit_cost(edit_cost, edit_cost_constants)
		
		
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
		self._initialized = False
		graph_id = self._ged_data._num_graphs_without_shuffled_copies
		self._ged_data._num_graphs_without_shuffled_copies += 1
		self._new_graph_ids.append(graph_id)
		self._ged_data._graphs.append(nx.Graph())
		self._ged_data._graph_names.append(graph_name)
		self._ged_data._graph_classes.append(graph_class)
		self._original_to_internal_node_ids.append({})
		self._internal_to_original_node_ids.append({})
		self._ged_data._strings_to_internal_node_ids.append({})
		self._ged_data._internal_node_ids_to_strings.append({})
		return graph_id
	
	
	def clear_graph(self, graph_id):
		"""
	/*!
	 * @brief Clears and de-initializes a graph that has previously been added to the environment. Call init() after calling this method.
	 * @param[in] graph_id ID of graph that has to be cleared.
	 */
		"""
		if graph_id > self._ged_data.num_graphs_without_shuffled_copies():
			raise Exception('The graph', self.get_graph_name(graph_id), 'has not been added to the environment.')
		self._ged_data._graphs[graph_id].clear()
		self._original_to_internal_node_ids[graph_id].clear()
		self._internal_to_original_node_ids[graph_id].clear()
		self._ged_data._strings_to_internal_node_ids[graph_id].clear()
		self._ged_data._internal_node_ids_to_strings[graph_id].clear()
		self._initialized = False
	
	
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
		self._initialized = False
		internal_node_id = nx.number_of_nodes(self._ged_data._graphs[graph_id])
		self._ged_data._graphs[graph_id].add_node(internal_node_id, label=node_label)
		self._original_to_internal_node_ids[graph_id][node_id] = internal_node_id
		self._internal_to_original_node_ids[graph_id][internal_node_id] = node_id
		self._ged_data._strings_to_internal_node_ids[graph_id][str(node_id)] = internal_node_id
		self._ged_data._internal_node_ids_to_strings[graph_id][internal_node_id] = str(node_id)
		self._ged_data._node_label_to_id(node_label)
		label_id = self._ged_data._node_label_to_id(node_label)
		# @todo: ged_data_.graphs_[graph_id].set_label
		
		
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
		self._initialized = False
		# @todo: check ignore_duplicates.
		self._ged_data._graphs[graph_id].add_edge(self._original_to_internal_node_ids[graph_id][nd_from], self._original_to_internal_node_ids[graph_id][nd_to], label=edge_label)
		label_id = self._ged_data._edge_label_to_id(edge_label)
		# @todo: ged_data_.graphs_[graph_id].set_label
		
	
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
	
	
	def load_nx_graph(self, nx_graph, graph_id, graph_name='', graph_class=''):
		"""
		Loads NetworkX Graph into the GED environment.

		Parameters
		----------
		nx_graph : NetworkX Graph object
			The graph that should be loaded.
			
		graph_id : int or None
			The ID of a graph contained the environment (overwrite existing graph) or add new graph if `None`.
																							
		graph_name : string, optional
			The name of newly added graph. The default is ''. Has no effect unless `graph_id` equals `None`.
			
		graph_class : string, optional
			The class of newly added graph. The default is ''. Has no effect unless `graph_id` equals `None`.

		Returns
		-------
		int
			The ID of the newly loaded graph.
		"""
		if graph_id is None: # @todo: undefined.
			graph_id = self.add_graph(graph_name, graph_class)
		else:
			self.clear_graph(graph_id)
		for node in nx_graph.nodes:
			self.add_node(graph_id, node, tuple(sorted(nx_graph.nodes[node].items(), key=lambda kv: kv[0])))
		for edge in nx_graph.edges:
			self.add_edge(graph_id, edge[0], edge[1], tuple(sorted(nx_graph.edges[(edge[0], edge[1])].items(), key=lambda kv: kv[0])))
		return graph_id
	
	
	def init(self, init_type=Options.InitType.EAGER_WITHOUT_SHUFFLED_COPIES, print_to_stdout=False):
		if isinstance(init_type, str):
			init_type = OptionsStringMap.InitType[init_type]
			
		# Throw an exception if no edit costs have been selected.
		if self._ged_data._edit_cost is None:
			raise Exception('No edit costs have been selected. Call set_edit_cost() before calling init().')
			
		# Return if the environment is initialized.
		if self._initialized:
			return
		
		# Set initialization type.
		self._ged_data._init_type = init_type
		
		# @todo: Construct shuffled graph copies if necessary.
		
		# Re-initialize adjacency matrices (also previously initialized graphs must be re-initialized because of possible re-allocation).
		# @todo: setup_adjacency_matrix, don't know if neccessary.
		self._ged_data._max_num_nodes = np.max([nx.number_of_nodes(g) for g in self._ged_data._graphs])
		self._ged_data._max_num_edges = np.max([nx.number_of_edges(g) for g in self._ged_data._graphs])
			
		# Initialize cost matrices if necessary.
		if self._ged_data._eager_init():
			pass # @todo: init_cost_matrices_: 1. Update node cost matrix if new node labels have been added to the environment; 2. Update edge cost matrix if new edge labels have been added to the environment.
			
		# Mark environment as initialized.
		self._initialized = True
		self._new_graph_ids.clear()
		
		
	def is_initialized(self):
		"""
	/*!
	 * @brief Check if the environment is initialized.
	 * @return True if the environment is initialized.
	 */
		"""
		return self._initialized
	
	
	def get_init_type(self):
		"""
	/*!
	 * @brief Returns the initialization type of the last initialization.
	 * @return Initialization type.
	 */
		"""
		return self._ged_data._init_type
	
	
	def set_label_costs(self, node_label_costs=None, edge_label_costs=None):
		"""Set the costs between labels. 
		"""
		if node_label_costs is not None:
			self._ged_data._node_label_costs = node_label_costs
		if edge_label_costs is not None:
			self._ged_data._edge_label_costs = edge_label_costs
		
		
	def set_method(self, method, options=''):
		"""
	/*!
	 * @brief Sets the GEDMethod to be used by run_method().
	 * @param[in] method Select the method that is to be used.
	 * @param[in] options An options string of the form @"[--@<option@> @<arg@>] [...]@" passed to the selected method.
	 */
		"""
		del self._ged_method
		
		if isinstance(method, str):
			method = OptionsStringMap.GEDMethod[method]

		if method == Options.GEDMethod.BRANCH:
			self._ged_method = Branch(self._ged_data)
		elif method == Options.GEDMethod.BRANCH_FAST:
			self._ged_method = BranchFast(self._ged_data)
		elif method == Options.GEDMethod.BRANCH_FAST:
			self._ged_method = BranchFast(self._ged_data)	
		elif method == Options.GEDMethod.BRANCH_TIGHT:
			self._ged_method = BranchTight(self._ged_data)	
		elif method == Options.GEDMethod.BRANCH_UNIFORM:
			self._ged_method = BranchUniform(self._ged_data)	
		elif method == Options.GEDMethod.BRANCH_COMPACT:
			self._ged_method = BranchCompact(self._ged_data)	
		elif method == Options.GEDMethod.PARTITION:
			self._ged_method = Partition(self._ged_data)	
		elif method == Options.GEDMethod.HYBRID:
			self._ged_method = Hybrid(self._ged_data)	
		elif method == Options.GEDMethod.RING:
			self._ged_method = Ring(self._ged_data)	
		elif method == Options.GEDMethod.ANCHOR_AWARE_GED:
			self._ged_method = AnchorAwareGED(self._ged_data)	
		elif method == Options.GEDMethod.WALKS:
			self._ged_method = Walks(self._ged_data)	
		elif method == Options.GEDMethod.IPFP:
			self._ged_method = IPFP(self._ged_data)	
		elif method == Options.GEDMethod.BIPARTITE:
			from gklearn.ged.methods import Bipartite
			self._ged_method = Bipartite(self._ged_data)	
		elif method == Options.GEDMethod.SUBGRAPH:
			self._ged_method = Subgraph(self._ged_data)	
		elif method == Options.GEDMethod.NODE:
			self._ged_method = Node(self._ged_data)	
		elif method == Options.GEDMethod.RING_ML:
			self._ged_method = RingML(self._ged_data)	
		elif method == Options.GEDMethod.BIPARTITE_ML:
			self._ged_method = BipartiteML(self._ged_data)	
		elif method == Options.GEDMethod.REFINE:
			self._ged_method = Refine(self._ged_data)	
		elif method == Options.GEDMethod.BP_BEAM:
			self._ged_method = BPBeam(self._ged_data)	
		elif method == Options.GEDMethod.SIMULATED_ANNEALING:
			self._ged_method = SimulatedAnnealing(self._ged_data)	
		elif method == Options.GEDMethod.HED:
			self._ged_method = HED(self._ged_data)	
		elif method == Options.GEDMethod.STAR:
			self._ged_method = STAR(self._ged_data)	
		# #ifdef GUROBI
		elif method == Options.GEDMethod.F1:
			self._ged_method = F1(self._ged_data)	
		elif method == Options.GEDMethod.F2:
			self._ged_method = F2(self._ged_data)	
		elif method == Options.GEDMethod.COMPACT_MIP:
			self._ged_method = CompactMIP(self._ged_data)	
		elif method == Options.GEDMethod.BLP_NO_EDGE_LABELS:
			self._ged_method = BLPNoEdgeLabels(self._ged_data)	

		self._ged_method.set_options(options)
		
		
	def run_method(self, g_id, h_id):
		"""
	/*!
	 * @brief Runs the GED method specified by call to set_method() between the graphs with IDs @p g_id and @p h_id.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 */
		"""
		if g_id >= self._ged_data.num_graphs():
			raise Exception('The graph with ID', str(g_id), 'has not been added to the environment.')
		if h_id >= self._ged_data.num_graphs():
			raise Exception('The graph with ID', str(h_id), 'has not been added to the environment.')
		if not self._initialized:
			raise Exception('The environment is uninitialized. Call init() after adding all graphs to the environment.')
		if self._ged_method is None:
			raise Exception('No method has been set. Call set_method() before calling run().')
		
		# Call selected GEDMethod and store results.
		if self._ged_data.shuffled_graph_copies_available() and (g_id == h_id):
			self._ged_method.run(g_id, self._ged_data.id_shuffled_graph_copy(h_id)) # @todo: why shuffle?
		else:
			self._ged_method.run(g_id, h_id)
		self._lower_bounds[(g_id, h_id)] = self._ged_method.get_lower_bound()
		self._upper_bounds[(g_id, h_id)] = self._ged_method.get_upper_bound()
		self._runtimes[(g_id, h_id)] = self._ged_method.get_runtime()
		self._node_maps[(g_id, h_id)] = self._ged_method.get_node_map()
		
		
	def init_method(self):
		"""Initializes the method specified by call to set_method().
		"""
		if not self._initialized:
			raise Exception('The environment is uninitialized. Call init() before calling init_method().')
		if self._ged_method is None:
			raise Exception('No method has been set. Call set_method() before calling init_method().')
		self._ged_method.init()
		
		
	def get_num_node_labels(self):
		"""
	/*!
	 * @brief Returns the number of node labels.
	 * @return Number of pairwise different node labels contained in the environment.
	 * @note If @p 1 is returned, the nodes are unlabeled.
	 */
		"""
		return len(self._ged_data._node_labels)
	
	
	def get_all_node_labels(self):
		"""
	/*!
	 * @brief Returns the list of all node labels.
	 * @return List of pairwise different node labels contained in the environment.
	 * @note If @p 1 is returned, the nodes are unlabeled.
	 */
		"""
		return self._ged_data._node_labels
	
	
	def get_node_label(self, label_id, to_dict=True):
		"""
	/*!
	 * @brief Returns node label.
	 * @param[in] label_id ID of node label that should be returned. Must be between 1 and num_node_labels().
	 * @return Node label for selected label ID.
	 */
		"""
		if label_id < 1 or label_id > self.get_num_node_labels():
			raise Exception('The environment does not contain a node label with ID', str(label_id), '.')
		if to_dict:
			return dict(self._ged_data._node_labels[label_id - 1])
		return self._ged_data._node_labels[label_id - 1]
		
		
	def get_num_edge_labels(self):
		"""
	/*!
	 * @brief Returns the number of edge labels.
	 * @return Number of pairwise different edge labels contained in the environment.
	 * @note If @p 1 is returned, the edges are unlabeled.
	 */
		"""
		return len(self._ged_data._edge_labels)
	
	
	def get_all_edge_labels(self):
		"""
	/*!
	 * @brief Returns the list of all edge labels.
	 * @return List of pairwise different edge labels contained in the environment.
	 * @note If @p 1 is returned, the edges are unlabeled.
	 */
		"""
		return self._ged_data._edge_labels
	
		
	def get_edge_label(self, label_id, to_dict=True):
		"""
	/*!
	 * @brief Returns edge label.
	 * @param[in] label_id ID of edge label that should be returned. Must be between 1 and num_node_labels().
	 * @return Edge label for selected label ID.
	 */
		"""
		if label_id < 1 or label_id > self.get_num_edge_labels():
			raise Exception('The environment does not contain an edge label with ID', str(label_id), '.')
		if to_dict:
			return dict(self._ged_data._edge_labels[label_id - 1])
		return self._ged_data._edge_labels[label_id - 1]
	
		
	def get_upper_bound(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns upper bound for edit distance between the input graphs.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Upper bound computed by the last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self._upper_bounds:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_upper_bound(' + str(g_id) + ',' + str(h_id) + ').')
		return self._upper_bounds[(g_id, h_id)]
		
		
	def get_lower_bound(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns lower bound for edit distance between the input graphs.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Lower bound computed by the last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self._lower_bounds:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_lower_bound(' + str(g_id) + ',' + str(h_id) + ').')
		return self._lower_bounds[(g_id, h_id)]
		
		
	def get_runtime(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns runtime.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Runtime of last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self._runtimes:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_runtime(' + str(g_id) + ',' + str(h_id) + ').')
		return self._runtimes[(g_id, h_id)]
	

	def get_init_time(self):
		"""
	/*!
	 * @brief Returns initialization time.
	 * @return Runtime of the last call to init_method().
	 */
		"""		
		return self._ged_method.get_init_time()


	def get_node_map(self, g_id, h_id):
		"""
	/*!
	 * @brief Returns node map between the input graphs.
	 * @param[in] g_id ID of an input graph that has been added to the environment.
	 * @param[in] h_id ID of an input graph that has been added to the environment.
	 * @return Node map computed by the last call to run_method() with arguments @p g_id and @p h_id.
	 */
		"""
		if (g_id, h_id) not in self._node_maps:
			raise Exception('Call run(' + str(g_id) + ',' + str(h_id) + ') before calling get_node_map(' + str(g_id) + ',' + str(h_id) + ').')
		return self._node_maps[(g_id, h_id)]
	

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
	
	
	def compute_induced_cost(self, g_id, h_id, node_map):
		"""
	/*!
	 * @brief Computes the edit cost between two graphs induced by a node map.
	 * @param[in] g_id ID of input graph.
	 * @param[in] h_id ID of input graph.
	 * @param[in,out] node_map Node map whose induced edit cost is to be computed.
	 */
		"""
		self._ged_data.compute_induced_cost(self._ged_data._graphs[g_id], self._ged_data._graphs[h_id], node_map)
	
	
	def get_nx_graph(self, graph_id):
		"""
	 * @brief Returns NetworkX.Graph() representation.
	 * @param[in] graph_id ID of the selected graph.
		"""
		graph = nx.Graph() # @todo: add graph attributes.
		graph.graph['id'] = graph_id
		
		nb_nodes = self.get_graph_num_nodes(graph_id)
		original_node_ids = self.get_original_node_ids(graph_id)
		node_labels = self.get_graph_node_labels(graph_id, to_dict=True)
		graph.graph['original_node_ids'] = original_node_ids		
		
		for node_id in range(0, nb_nodes):
			graph.add_node(node_id, **node_labels[node_id])
			
		edges = self.get_graph_edges(graph_id, to_dict=True)
		for (head, tail), labels in edges.items():
			graph.add_edge(head, tail, **labels)

		return graph
	
	
	def get_graph_node_labels(self, graph_id, to_dict=True):
		"""
			Searchs and returns all the labels of nodes on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The list of nodes' labels on the selected graph
			:rtype: list[dict{string : string}]
			
			.. seealso:: get_graph_internal_id(), get_graph_num_nodes(), get_graph_num_edges(), get_original_node_ids(), get_graph_edges(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		graph = self._ged_data.graph(graph_id)
		node_labels = []
		for n in graph.nodes():
			node_labels.append(graph.nodes[n]['label'])
		if to_dict:
			return [dict(i) for i in node_labels]
		return node_labels
	
	
	def get_graph_edges(self, graph_id, to_dict=True):
		"""
			Searchs and returns all the edges on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The list of edges on the selected graph
			:rtype: dict{tuple(size_t, size_t) : dict{string : string}}
			
			.. seealso::get_graph_internal_id(), get_graph_num_nodes(), get_graph_num_edges(), get_original_node_ids(), get_graph_node_labels(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		graph = self._ged_data.graph(graph_id)		
		if to_dict:
			edges = {}		
			for n1, n2, attr in graph.edges(data=True):
				edges[(n1, n2)] = dict(attr['label'])
			return edges
		return {(n1, n2): attr['label'] for n1, n2, attr in graph.edges(data=True)}

	
	
	def get_graph_name(self, graph_id):
		"""
	/*!
	 * @brief Returns the graph name.
	 * @param[in] graph_id ID of an input graph that has been added to the environment.
	 * @return Name of the input graph.
	 */
		"""
		return self._ged_data._graph_names[graph_id]
	
	
	def get_graph_num_nodes(self, graph_id):
		"""
	/*!
	 * @brief Returns the number of nodes.
	 * @param[in] graph_id ID of an input graph that has been added to the environment.
	 * @return Number of nodes in the graph.
	 */
		"""
		return nx.number_of_nodes(self._ged_data.graph(graph_id))
	
	
	def get_original_node_ids(self, graph_id):
		"""
			Searchs and returns all th Ids of nodes on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The list of IDs's nodes on the selected graph
			:rtype: list[string]
			
			.. seealso::get_graph_internal_id(), get_graph_num_nodes(), get_graph_num_edges(), get_graph_node_labels(), get_graph_edges(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return [i for i in self._internal_to_original_node_ids[graph_id].values()]		
	
	
	def get_node_cost(self, node_label_1, node_label_2):
		return self._ged_data.node_cost(node_label_1, node_label_2)
	
	
	def get_node_rel_cost(self, node_label_1, node_label_2):
		"""
	/*!
	 * @brief Returns node relabeling cost.
	 * @param[in] node_label_1 First node label.
	 * @param[in] node_label_2 Second node label.
	 * @return Node relabeling cost for the given node labels.
	 */
		"""
		if isinstance(node_label_1, dict):
			node_label_1 = tuple(sorted(node_label_1.items(), key=lambda kv: kv[0]))
		if isinstance(node_label_2, dict):
			node_label_2 = tuple(sorted(node_label_2.items(), key=lambda kv: kv[0]))
		return self._ged_data._edit_cost.node_rel_cost_fun(node_label_1, node_label_2) # @todo: may need to use node_cost() instead (or change node_cost() and modify ged_method for pre-defined cost matrices.)
		
		
	def get_node_del_cost(self, node_label):
		"""
	/*!
	 * @brief Returns node deletion cost.
	 * @param[in] node_label Node label.
	 * @return Cost of deleting node with given label.
	 */
		"""
		if isinstance(node_label, dict):
			node_label = tuple(sorted(node_label.items(), key=lambda kv: kv[0]))
		return self._ged_data._edit_cost.node_del_cost_fun(node_label)
		
		
	def get_node_ins_cost(self, node_label):
		"""
	/*!
	 * @brief Returns node insertion cost.
	 * @param[in] node_label Node label.
	 * @return Cost of inserting node with given label.
	 */
		"""
		if isinstance(node_label, dict):
			node_label = tuple(sorted(node_label.items(), key=lambda kv: kv[0]))
		return self._ged_data._edit_cost.node_ins_cost_fun(node_label)
	
	
	def get_edge_cost(self, edge_label_1, edge_label_2):
		return self._ged_data.edge_cost(edge_label_1, edge_label_2)
		
		
	def get_edge_rel_cost(self, edge_label_1, edge_label_2):
		"""
	/*!
	 * @brief Returns edge relabeling cost.
	 * @param[in] edge_label_1 First edge label.
	 * @param[in] edge_label_2 Second edge label.
	 * @return Edge relabeling cost for the given edge labels.
	 */
		"""
		if isinstance(edge_label_1, dict):
			edge_label_1 = tuple(sorted(edge_label_1.items(), key=lambda kv: kv[0]))
		if isinstance(edge_label_2, dict):
			edge_label_2 = tuple(sorted(edge_label_2.items(), key=lambda kv: kv[0]))
		return self._ged_data._edit_cost.edge_rel_cost_fun(edge_label_1, edge_label_2)
		
		
	def get_edge_del_cost(self, edge_label):
		"""
	/*!
	 * @brief Returns edge deletion cost.
	 * @param[in] edge_label Edge label.
	 * @return Cost of deleting edge with given label.
	 */
		"""
		if isinstance(edge_label, dict):
			edge_label = tuple(sorted(edge_label.items(), key=lambda kv: kv[0]))
		return self._ged_data._edit_cost.edge_del_cost_fun(edge_label)
		
		
	def get_edge_ins_cost(self, edge_label):
		"""
	/*!
	 * @brief Returns edge insertion cost.
	 * @param[in] edge_label Edge label.
	 * @return Cost of inserting edge with given label.
	 */
		"""
		if isinstance(edge_label, dict):
			edge_label = tuple(sorted(edge_label.items(), key=lambda kv: kv[0]))
		return self._ged_data._edit_cost.edge_ins_cost_fun(edge_label)
		
		
	def get_all_graph_ids(self):
		return [i for i in range(0, self._ged_data._num_graphs_without_shuffled_copies)]