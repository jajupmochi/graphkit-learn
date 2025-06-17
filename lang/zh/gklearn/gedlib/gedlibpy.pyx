# distutils: language = c++

"""
	Python GedLib module
	======================
	
	This module allow to use a C++ library for edit distance between graphs (GedLib) with Python.

	
	Authors
	-------------------
 
	David Blumenthal
	Natacha Lambert
	Linlin Jia

	Copyright (C) 2019-2020 by all the authors

	Classes & Functions
	-------------------
 
"""

#################################
##DECLARATION OF C++ INTERFACES##
#################################


#Types imports for C++ compatibility
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.list cimport list

#Long unsigned int equivalent
cimport numpy as cnp
ctypedef cnp.npy_uint32 UINT32_t
from cpython cimport array

		
cdef extern from "src/GedLibBind.hpp" namespace "pyged":
	
	cdef vector[string] getEditCostStringOptions() except +
	cdef vector[string] getMethodStringOptions() except +
	cdef vector[string] getInitStringOptions() except +
	cdef size_t getDummyNode() except +
	
	cdef cppclass PyGEDEnv:
		PyGEDEnv() except +
		bool isInitialized() except +
		void restartEnv() except +
		void loadGXLGraph(string pathFolder, string pathXML, bool node_type, bool edge_type) except +
		pair[size_t,size_t] getGraphIds() except +
		vector[size_t] getAllGraphIds() except +
		string getGraphClass(size_t id) except +
		string getGraphName(size_t id) except +
		size_t addGraph(string name, string classe) except +
		void addNode(size_t graphId, string nodeId, map[string, string] nodeLabel) except +
		void addEdge(size_t graphId, string tail, string head, map[string, string] edgeLabel, bool ignoreDuplicates) except +
		void clearGraph(size_t graphId) except +
		size_t getGraphInternalId(size_t graphId) except +
		size_t getGraphNumNodes(size_t graphId) except +
		size_t getGraphNumEdges(size_t graphId) except +
		vector[string] getGraphOriginalNodeIds(size_t graphId) except +
		vector[map[string, string]] getGraphNodeLabels(size_t graphId) except +
		map[pair[size_t, size_t], map[string, string]] getGraphEdges(size_t graphId) except +
		vector[vector[size_t]] getGraphAdjacenceMatrix(size_t graphId) except +
		void setEditCost(string editCost, vector[double] editCostConstant) except +
		void setPersonalEditCost(vector[double] editCostConstant) except +
		void initEnv(string initOption, bool print_to_stdout) except +
		void setMethod(string method, string options) except +
		void initMethod() except +
		double getInitime() except +
		void runMethod(size_t g, size_t h) except +
		double getUpperBound(size_t g, size_t h) except +
		double getLowerBound(size_t g, size_t h) except +
		vector[cnp.npy_uint64] getForwardMap(size_t g, size_t h) except +
		vector[cnp.npy_uint64] getBackwardMap(size_t g, size_t h) except +
		size_t getNodeImage(size_t g, size_t h, size_t nodeId) except +
		size_t getNodePreImage(size_t g, size_t h, size_t nodeId) except +
		double getInducedCost(size_t g, size_t h) except +
		vector[pair[size_t,size_t]] getNodeMap(size_t g, size_t h) except +
		vector[vector[int]] getAssignmentMatrix(size_t g, size_t h) except +
		vector[vector[cnp.npy_uint64]] getAllMap(size_t g, size_t h) except +
		double getRuntime(size_t g, size_t h) except +
		bool quasimetricCosts() except +
		vector[vector[size_t]] hungarianLSAP(vector[vector[size_t]] matrixCost) except +
		vector[vector[double]] hungarianLSAPE(vector[vector[double]] matrixCost) except +
		# added by Linlin Jia.
		size_t getNumNodeLabels() except +
		map[string, string] getNodeLabel(size_t label_id) except +
		size_t getNumEdgeLabels() except +
		map[string, string] getEdgeLabel(size_t label_id) except +
# 		size_t getNumNodes(size_t graph_id) except +
		double getAvgNumNodes() except +
		double getNodeRelCost(map[string, string] & node_label_1, map[string, string] & node_label_2) except +
		double getNodeDelCost(map[string, string] & node_label) except +
		double getNodeInsCost(map[string, string] & node_label) except +
		map[string, string] getMedianNodeLabel(vector[map[string, string]] & node_labels) except +
		double getEdgeRelCost(map[string, string] & edge_label_1, map[string, string] & edge_label_2) except +
		double getEdgeDelCost(map[string, string] & edge_label) except +
		double getEdgeInsCost(map[string, string] & edge_label) except +
		map[string, string] getMedianEdgeLabel(vector[map[string, string]] & edge_labels) except +
		string getInitType() except +
# 		double getNodeCost(size_t label1, size_t label2) except +
		double computeInducedCost(size_t g_id, size_t h_id, vector[pair[size_t,size_t]]) except +
		
		
#############################
##CYTHON WRAPPER INTERFACES##
#############################

# import cython
import numpy as np
import networkx as nx
from gklearn.ged.env import NodeMap

# import librariesImport
from ctypes import *
import os
lib1 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/fann/libdoublefann.so')
lib2 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/libsvm.3.22/libsvm.so')
lib3 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/nomad/libnomad.so')
lib4 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/nomad/libsgtelib.so')


def get_edit_cost_options() :
	"""
		Searchs the differents edit cost functions and returns the result.
 
		:return: The list of edit cost functions
		:rtype: list[string]
 
		.. warning:: This function is useless for an external use. Please use directly list_of_edit_cost_options. 
		.. note:: Prefer the list_of_edit_cost_options attribute of this module.
	"""
	
	return [option.decode('utf-8') for option in getEditCostStringOptions()]


def get_method_options() :
	"""
		Searchs the differents method for edit distance computation between graphs and returns the result.
 
		:return: The list of method to compute the edit distance between graphs
		:rtype: list[string]
 
		.. warning:: This function is useless for an external use. Please use directly list_of_method_options.
		.. note:: Prefer the list_of_method_options attribute of this module.
	"""
	return [option.decode('utf-8') for option in getMethodStringOptions()]


def get_init_options() :
	"""
		Searchs the differents initialization parameters for the environment computation for graphs and returns the result.
 
		:return: The list of options to initialize the computation environment
		:rtype: list[string]
 
		.. warning:: This function is useless for an external use. Please use directly list_of_init_options.
		.. note:: Prefer the list_of_init_options attribute of this module.
	"""
	return [option.decode('utf-8') for option in getInitStringOptions()]


def get_dummy_node() :
	"""
		Returns the ID of a dummy node.

		:return: The ID of the dummy node (18446744073709551614 for my computer, the hugest number possible)
		:rtype: size_t
		
		.. note:: A dummy node is used when a node isn't associated to an other node.	  
	"""
	return getDummyNode()
	

# @cython.auto_pickle(True)
cdef class GEDEnv:
	"""Cython wrapper class for C++ class PyGEDEnv
	"""
# 	cdef PyGEDEnv c_env  # Hold a C++ instance which we're wrapping
	cdef PyGEDEnv* c_env  # hold a pointer to the C++ instance which we're wrapping


	def __cinit__(self):
# 		self.c_env = PyGEDEnv()
		self.c_env = new PyGEDEnv()
		
	
	def __dealloc__(self):
		del self.c_env


# 	def __reduce__(self):
# # 		return GEDEnv, (self.c_env,)
# 		return GEDEnv, tuple()


	def is_initialized(self) :
		"""
			Checks and returns if the computation environment is initialized or not.
	 
			:return: True if it's initialized, False otherwise
			:rtype: bool
			
			.. note:: This function exists for internals verifications but you can use it for your code. 
		"""
		return self.c_env.isInitialized()
	
	
	def restart_env(self) :
		"""
			Restarts the environment variable. All data related to it will be delete. 
	 
			.. warning:: This function deletes all graphs, computations and more so make sure you don't need anymore your environment. 
			.. note:: You can now delete and add somes graphs after initialization so you can avoid this function. 
		"""
		self.c_env.restartEnv()

	
	def load_GXL_graphs(self, path_folder, path_XML, node_type, edge_type) :
		"""
			Loads some GXL graphes on the environment which is in a same folder, and present in the XMLfile. 
			
			:param path_folder: The folder's path which contains GXL graphs
			:param path_XML: The XML's path which indicates which graphes you want to load
			:param node_type: Select if nodes are labeled or unlabeled
			:param edge_type: Select if edges are labeled or unlabeled
			:type path_folder: string
			:type path_XML: string
			:type node_type: bool
			:type edge_type: bool

	 
			.. note:: You can call this function multiple times if you want, but not after an init call. 
		"""
		self.c_env.loadGXLGraph(path_folder.encode('utf-8'), path_XML.encode('utf-8'), node_type, edge_type)

	
	def graph_ids(self) :
		"""
			Searchs the first and last IDs of the loaded graphs in the environment. 
	 
			:return: The pair of the first and the last graphs Ids
			:rtype: tuple(size_t, size_t)
			
			.. note:: Prefer this function if you have huges structures with lots of graphs.  
		"""
		return self.c_env.getGraphIds()

	
	def get_all_graph_ids(self) :
		"""
			Searchs all the IDs of the loaded graphs in the environment. 
	 
			:return: The list of all graphs's Ids 
			:rtype: list[size_t]
			
			.. note:: The last ID is equal to (number of graphs - 1). The order correspond to the loading order. 
		"""
		return self.c_env.getAllGraphIds()

	
	def get_graph_class(self, id) :
		"""
			Returns the class of a graph with its ID.
	
			:param id: The ID of the wanted graph
			:type id: size_t
			:return: The class of the graph which correpond to the ID
			:rtype: string
			
			.. seealso:: get_graph_class()
			.. note:: An empty string can be a class. 
		"""
		return self.c_env.getGraphClass(id)

	
	def get_graph_name(self, id) :
		"""
			Returns the name of a graph with its ID. 
	
			:param id: The ID of the wanted graph
			:type id: size_t
			:return: The name of the graph which correpond to the ID
			:rtype: string
			
			.. seealso:: get_graph_class()
			.. note:: An empty string can be a name. 
		"""
		return self.c_env.getGraphName(id).decode('utf-8')

	
	def add_graph(self, name="", classe="") :
		"""
			Adds a empty graph on the environment, with its name and its class. Nodes and edges will be add in a second time. 
	
			:param name: The name of the new graph, an empty string by default
			:param classe: The class of the new graph, an empty string by default
			:type name: string
			:type classe: string
			:return: The ID of the newly graphe
			:rtype: size_t
			
			.. seealso::add_node(), add_edge() , add_symmetrical_edge()
			.. note:: You can call this function without parameters. You can also use this function after initialization, call init() after you're finished your modifications. 
		"""
		return self.c_env.addGraph(name.encode('utf-8'), classe.encode('utf-8'))

	
	def add_node(self, graph_id, node_id, node_label):
		"""
			Adds a node on a graph selected by its ID. A ID and a label for the node is required. 
	
			:param graph_id: The ID of the wanted graph
			:param node_id: The ID of the new node
			:param node_label: The label of the new node
			:type graph_id: size_t
			:type node_id: string
			:type node_label: dict{string : string}
			
			.. seealso:: add_graph(), add_edge(), add_symmetrical_edge()
			.. note:: You can also use this function after initialization, but only on a newly added graph. Call init() after you're finished your modifications. 
		"""
		self.c_env.addNode(graph_id, node_id.encode('utf-8'), encode_your_map(node_label))

	
	def add_edge(self, graph_id, tail, head, edge_label, ignore_duplicates=True) :
		"""
			Adds an edge on a graph selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:param tail: The ID of the tail node for the new edge
			:param head: The ID of the head node for the new edge
			:param edge_label: The label of the new edge
			:param ignore_duplicates: If True, duplicate edges are ignored, otherwise it's raise an error if an existing edge is added. True by default
			:type graph_id: size_t
			:type tail: string
			:type head: string
			:type edge_label: dict{string : string}
			:type ignore_duplicates: bool
			
			.. seealso:: add_graph(), add_node(), add_symmetrical_edge()
			.. note:: You can also use this function after initialization, but only on a newly added graph. Call init() after you're finished your modifications. 
		"""
		self.c_env.addEdge(graph_id, tail.encode('utf-8'), head.encode('utf-8'), encode_your_map(edge_label), ignore_duplicates)

	
	def add_symmetrical_edge(self, graph_id, tail, head, edge_label) :
		"""
			Adds a symmetrical edge on a graph selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:param tail: The ID of the tail node for the new edge
			:param head: The ID of the head node for the new edge
			:param edge_label: The label of the new edge
			:type graph_id: size_t
			:type tail: string
			:type head: string
			:type edge_label: dict{string : string}
			
			.. seealso:: add_graph(), add_node(), add_edge()
			.. note:: You can also use this function after initialization, but only on a newly added graph. Call init() after you're finished your modifications. 
		"""
		tailB = tail.encode('utf-8')
		headB = head.encode('utf-8')
		edgeLabelB = encode_your_map(edge_label)
		self.c_env.addEdge(graph_id, tailB, headB, edgeLabelB, True)
		self.c_env.addEdge(graph_id, headB, tailB, edgeLabelB, True)

	
	def clear_graph(self, graph_id) :
		"""
			Deletes a graph, selected by its ID, to the environment.
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			
			.. note:: Call init() after you're finished your modifications. 
		"""
		self.c_env.clearGraph(graph_id)

	
	def get_graph_internal_id(self, graph_id) :
		"""
			Searchs and returns the internal Id of a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The internal ID of the selected graph
			:rtype: size_t
			
			.. seealso:: get_graph_num_nodes(), get_graph_num_edges(), get_original_node_ids(), get_graph_node_labels(), get_graph_edges(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return self.c_env.getGraphInternalId(graph_id)

	
	def get_graph_num_nodes(self, graph_id) :
		"""
			Searchs and returns the number of nodes on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The number of nodes on the selected graph
			:rtype: size_t
			
			.. seealso:: get_graph_internal_id(), get_graph_num_edges(), get_original_node_ids(), get_graph_node_labels(), get_graph_edges(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return self.c_env.getGraphNumNodes(graph_id)

	
	def get_graph_num_edges(self, graph_id) :
		"""
			Searchs and returns the number of edges on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The number of edges on the selected graph
			:rtype: size_t
			
			.. seealso:: get_graph_internal_id(), get_graph_num_nodes(), get_original_node_ids(), get_graph_node_labels(), get_graph_edges(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return self.c_env.getGraphNumEdges(graph_id)

	
	def get_original_node_ids(self, graph_id) :
		"""
			Searchs and returns all th Ids of nodes on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The list of IDs's nodes on the selected graph
			:rtype: list[string]
			
			.. seealso::get_graph_internal_id(), get_graph_num_nodes(), get_graph_num_edges(), get_graph_node_labels(), get_graph_edges(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return [gid.decode('utf-8') for gid in self.c_env.getGraphOriginalNodeIds(graph_id)]

	
	def get_graph_node_labels(self, graph_id) :
		"""
			Searchs and returns all the labels of nodes on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The list of nodes' labels on the selected graph
			:rtype: list[dict{string : string}]
			
			.. seealso:: get_graph_internal_id(), get_graph_num_nodes(), get_graph_num_edges(), get_original_node_ids(), get_graph_edges(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return [decode_your_map(node_label) for node_label in self.c_env.getGraphNodeLabels(graph_id)]

	
	def get_graph_edges(self, graph_id) :
		"""
			Searchs and returns all the edges on a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The list of edges on the selected graph
			:rtype: dict{tuple(size_t, size_t) : dict{string : string}}
			
			.. seealso::get_graph_internal_id(), get_graph_num_nodes(), get_graph_num_edges(), get_original_node_ids(), get_graph_node_labels(), get_graph_adjacence_matrix()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return decode_graph_edges(self.c_env.getGraphEdges(graph_id))

	
	def get_graph_adjacence_matrix(self, graph_id) :
		"""
			Searchs and returns the adjacence list of a graph, selected by its ID. 
	
			:param graph_id: The ID of the wanted graph
			:type graph_id: size_t
			:return: The adjacence list of the selected graph
			:rtype: list[list[size_t]]
			
			.. seealso:: get_graph_internal_id(), get_graph_num_nodes(), get_graph_num_edges(), get_original_node_ids(), get_graph_node_labels(), get_graph_edges()
			.. note:: These functions allow to collect all the graph's informations.
		"""
		return self.c_env.getGraphAdjacenceMatrix(graph_id)

	
	def set_edit_cost(self, edit_cost, edit_cost_constant = []) :
		"""
			Sets an edit cost function to the environment, if it exists. 
	
			:param edit_cost: The name of the edit cost function
			:type edit_cost: string
			:param edi_cost_constant: The parameters you will add to the editCost, empty by default
			:type edit_cost_constant: list
			
			.. seealso:: list_of_edit_cost_options
			.. note:: Try to make sure the edit cost function exists with list_of_edit_cost_options, raise an error otherwise. 
		"""
		if edit_cost in list_of_edit_cost_options:
			edit_cost_b = edit_cost.encode('utf-8')
			self.c_env.setEditCost(edit_cost_b, edit_cost_constant)
		else:
			raise EditCostError("This edit cost function doesn't exist, please see list_of_edit_cost_options for selecting a edit cost function")

	
	def set_personal_edit_cost(self, edit_cost_constant = []) :
		"""
			Sets an personal edit cost function to the environment.
	
			:param edit_cost_constant: The parameters you will add to the editCost, empty by default
			:type edit_cost_constant: list
	
			.. seealso:: list_of_edit_cost_options, set_edit_cost()
			.. note::You have to modify the C++ function to use it. Please see the documentation to add your Edit Cost function. 
		"""
		self.c_env.setPersonalEditCost(edit_cost_constant)

	
	def init(self, init_option='EAGER_WITHOUT_SHUFFLED_COPIES', print_to_stdout=False) :
		"""
			Initializes the environment with the chosen edit cost function and graphs.
	
			:param init_option: The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
			:type init_option: string
			
			.. seealso:: list_of_init_options
			.. warning:: No modification were allowed after initialization. Try to make sure your choices is correct. You can though clear or add a graph, but recall init() after that. 
			.. note:: Try to make sure the option exists with list_of_init_options or choose no options, raise an error otherwise.
		"""
		if init_option in list_of_init_options: 
			init_option_b = init_option.encode('utf-8')
			self.c_env.initEnv(init_option_b, print_to_stdout)		
		else:
			raise InitError("This init option doesn't exist, please see list_of_init_options for selecting an option. You can choose any options.")

	
	def set_method(self, method, options="") :
		"""
			Sets a computation method to the environment, if its exists. 
	
			:param method: The name of the computation method
			:param options: The options of the method (like bash options), an empty string by default
			:type method: string
			:type options: string
			
			.. seealso:: init_method(), list_of_method_options
			.. note:: Try to make sure the edit cost function exists with list_of_method_options, raise an error otherwise. Call init_method() after your set. 
		"""
		if method in list_of_method_options:
			method_b = method.encode('utf-8')
			self.c_env.setMethod(method_b, options.encode('utf-8'))
		else:
			raise MethodError("This method doesn't exist, please see list_of_method_options for selecting a method")

	
	def init_method(self) :
		"""
			Inits the environment with the set method.
	
			.. seealso:: set_method(), list_of_method_options
			.. note:: Call this function after set the method. You can't launch computation or change the method after that. 
		"""
		self.c_env.initMethod()

	
	def get_init_time(self) :
		"""
			Returns the initialization time.
	
			:return: The initialization time
			:rtype: double
		"""
		return self.c_env.getInitime()
	
	
	def run_method(self, g, h) :
		"""
			Computes the edit distance between two graphs g and h, with the edit cost function and method computation selected.  
	
			:param g: The Id of the first graph to compare
			:param h: The Id of the second graph to compare
			:type g: size_t
			:type h: size_t
			
			.. seealso:: get_upper_bound(), get_lower_bound(),  get_forward_map(), get_backward_map(), get_runtime(), quasimetric_cost()
			.. note:: This function only compute the distance between two graphs, without returning a result. Use the differents function to see the result between the two graphs.  
		"""
		self.c_env.runMethod(g, h)
	
	
	def get_upper_bound(self, g, h) :
		"""
			Returns the upper bound of the edit distance cost between two graphs g and h. 
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The upper bound of the edit distance cost
			:rtype: double
			
			.. seealso:: run_method(), get_lower_bound(),  get_forward_map(), get_backward_map(), get_runtime(), quasimetric_cost()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: The upper bound is equivalent to the result of the pessimist edit distance cost. Methods are heuristics so the library can't compute the real perfect result because it's NP-Hard problem.
		"""
		return self.c_env.getUpperBound(g, h)
	
	
	def get_lower_bound(self, g, h) :
		"""
			  Returns the lower bound of the edit distance cost between two graphs g and h. 
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The lower bound of the edit distance cost
			:rtype: double
			
			.. seealso:: run_method(), get_upper_bound(),  get_forward_map(), get_backward_map(), get_runtime(), quasimetric_cost()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: This function can be ignored, because lower bound doesn't have a crucial utility.	
		"""
		return self.c_env.getLowerBound(g, h)
	
	
	def get_forward_map(self, g, h) :
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
		return self.c_env.getForwardMap(g, h)
	
	
	def get_backward_map(self, g, h) :
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
		return self.c_env.getBackwardMap(g, h)
	
	
	def get_node_image(self, g, h, node_id) :
		"""
			Returns the node's image in the adjacence matrix, if it exists.   
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:param node_id: The ID of the node which you want to see the image
			:type g: size_t
			:type h: size_t
			:type node_id: size_t
			:return: The ID of the image node
			:rtype: size_t
			
			.. seealso:: run_method(), get_forward_map(), get_backward_map(), get_node_pre_image(), get_node_map(), get_assignment_matrix()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: Use BackwardMap's Node to find its images ! You can also use get_forward_map() and get_backward_map().	 
	
		"""
		return self.c_env.getNodeImage(g, h, node_id)
	
	
	def get_node_pre_image(self, g, h, node_id) :
		"""
			Returns the node's preimage in the adjacence matrix, if it exists.   
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:param node_id: The ID of the node which you want to see the preimage
			:type g: size_t
			:type h: size_t
			:type node_id: size_t
			:return: The ID of the preimage node
			:rtype: size_t
			
			.. seealso:: run_method(), get_forward_map(), get_backward_map(), get_node_image(), get_node_map(), get_assignment_matrix()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: Use ForwardMap's Node to find its images ! You can also use get_forward_map() and get_backward_map().	 
	
		"""
		return self.c_env.getNodePreImage(g, h, node_id)


	def get_induced_cost(self, g, h) :
		"""
			Returns the induced cost between the two indicated graphs.	

			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The induced cost between the two indicated graphs
			:rtype: double
			
			.. seealso:: run_method(), get_forward_map(), get_backward_map(), get_node_image(), get_node_map(), get_assignment_matrix()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: Use ForwardMap's Node to find its images ! You can also use get_forward_map() and get_backward_map().	 
	
		"""
		return self.c_env.getInducedCost(g, h)	
	
	
	def get_node_map(self, g, h) : 
		"""
			Returns the Node Map, like C++ NodeMap.   
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The Node Map between the two selected graph. 
			:rtype: gklearn.ged.env.NodeMap.
			
			.. seealso:: run_method(), get_forward_map(), get_backward_map(), get_node_image(), get_node_pre_image(), get_assignment_matrix()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: This function creates datas so use it if necessary, however you can understand how assignement works with this example.	 
		"""
		map_as_relation = self.c_env.getNodeMap(g, h)
		induced_cost = self.c_env.getInducedCost(g, h) # @todo: the C++ implementation for this function in GedLibBind.ipp re-call get_node_map() once more, this is not neccessary.
		source_map = [item.first if item.first < len(map_as_relation) else np.inf for item in map_as_relation] # item.first < len(map_as_relation) is not exactly correct.
# 		print(source_map)
		target_map = [item.second if item.second < len(map_as_relation) else np.inf for item in map_as_relation]
# 		print(target_map)
		num_node_source = len([item for item in source_map if item != np.inf])
# 		print(num_node_source)
		num_node_target = len([item for item in target_map if item != np.inf])
# 		print(num_node_target)
		
		node_map = NodeMap(num_node_source, num_node_target)
# 		print(node_map.get_forward_map(), node_map.get_backward_map())
		for i in range(len(source_map)):
			node_map.add_assignment(source_map[i], target_map[i])
		node_map.set_induced_cost(induced_cost)
		
		return node_map
	
	
	def get_assignment_matrix(self, g, h) :
		"""
			Returns the Assignment Matrix between two selected graphs g and h.   
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The Assignment Matrix between the two selected graph. 
			:rtype: list[list[int]]
			
			.. seealso:: run_method(), get_forward_map(), get_backward_map(), get_node_image(), get_node_pre_image(), get_node_map()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: This function creates datas so use it if necessary.	 
		"""
		return self.c_env.getAssignmentMatrix(g, h)
			
	
	def get_all_map(self, g, h) :
		"""
			Returns a vector which contains the forward and the backward maps between nodes of the two indicated graphs. 
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The forward and backward maps to the adjacence matrix between nodes of the two graphs
			:rtype: list[list[npy_uint32]]
			
			.. seealso:: run_method(), get_upper_bound(), get_lower_bound(),  get_forward_map(), get_backward_map(), get_runtime(), quasimetric_cost()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: This function duplicates data so please don't use it. I also don't know how to connect the two map to reconstruct the adjacence matrix. Please come back when I know how it's work !  
		"""
		return self.c_env.getAllMap(g, h)
	
	
	def get_runtime(self, g, h) :
		"""
			Returns the runtime to compute the edit distance cost between two graphs g and h  
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: The runtime of the computation of edit distance cost between the two selected graphs
			:rtype: double
			
			.. seealso:: run_method(), get_upper_bound(), get_lower_bound(),  get_forward_map(), get_backward_map(), quasimetric_cost()
			.. warning:: run_method() between the same two graph must be called before this function. 
			.. note:: Python is a bit longer than C++ due to the functions's encapsulate.	
		"""
		return self.c_env.getRuntime(g,h)
	
	
	def quasimetric_cost(self) :
		"""
			Checks and returns if the edit costs are quasimetric. 
	
			:param g: The Id of the first compared graph 
			:param h: The Id of the second compared graph
			:type g: size_t
			:type h: size_t
			:return: True if it's verified, False otherwise
			:rtype: bool
			
			.. seealso:: run_method(), get_upper_bound(), get_lower_bound(),  get_forward_map(), get_backward_map(), get_runtime()
			.. warning:: run_method() between the same two graph must be called before this function. 
		"""
		return self.c_env.quasimetricCosts()
	
	
	def hungarian_LSAP(self, matrix_cost) :
		"""
			Applies the hungarian algorithm (LSAP) on a matrix Cost. 
	
			:param matrix_cost: The matrix Cost  
			:type matrix_cost: vector[vector[size_t]]
			:return: The values of rho, varrho, u and v, in this order
			:rtype: vector[vector[size_t]]
			
			.. seealso:: hungarian_LSAPE() 
		"""
		return self.c_env.hungarianLSAP(matrix_cost)
	
	
	def hungarian_LSAPE(self, matrix_cost) :
		"""
			Applies the hungarian algorithm (LSAPE) on a matrix Cost. 
	
			:param matrix_cost: The matrix Cost 
			:type matrix_cost: vector[vector[double]]
			:return: The values of rho, varrho, u and v, in this order
			:rtype: vector[vector[double]]
			
			.. seealso:: hungarian_LSAP() 
		"""
		return self.c_env.hungarianLSAPE(matrix_cost)
	
	
	def add_random_graph(self, name, classe, list_of_nodes, list_of_edges, ignore_duplicates=True) :
		"""
			Add a Graph (not GXL) on the environment. Be careful to respect the same format as GXL graphs for labelling nodes and edges. 
	
			:param name: The name of the graph to add, can be an empty string
			:param classe: The classe of the graph to add, can be an empty string
			:param list_of_nodes: The list of nodes to add
			:param list_of_edges: The list of edges to add
			:param ignore_duplicates: If True, duplicate edges are ignored, otherwise it's raise an error if an existing edge is added. True by default
			:type name: string
			:type classe: string
			:type list_of_nodes: list[tuple(size_t, dict{string : string})]
			:type list_of_edges: list[tuple(tuple(size_t,size_t), dict{string : string})]
			:type ignore_duplicates: bool
			:return: The ID of the newly added graphe
			:rtype: size_t
	
			.. note:: The graph must respect the GXL structure. Please see how a GXL graph is construct.  
			
		"""
		id = self.add_graph(name, classe)
		for node in list_of_nodes:
			self.add_node(id, node[0], node[1])
		for edge in list_of_edges:
			self.add_edge(id, edge[0], edge[1], edge[2], ignore_duplicates)
		return id
	
	
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
		id = self.add_graph(g.name, classe)
		for node in g.nodes:
			self.add_node(id, str(node), g.nodes[node])
		for edge in g.edges:
			self.add_edge(id, str(edge[0]), str(edge[1]), g.get_edge_data(edge[0], edge[1]), ignore_duplicates)
		return id
	
	
	def compute_ged_on_two_graphs(self, g1, g2, edit_cost, method, options, init_option="EAGER_WITHOUT_SHUFFLED_COPIES") :
		"""
			Computes the edit distance between two NX graphs. 
			
			:param g1: The first graph to add and compute
			:param g2: The second graph to add and compute
			:param edit_cost: The name of the edit cost function
			:param method: The name of the computation method
			:param options: The options of the method (like bash options), an empty string by default
			:param init_option:  The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
			:type g1: networksx.graph
			:type g2: networksx.graph
			:type edit_cost: string
			:type method: string
			:type options: string
			:type init_option: string
			:return: The edit distance between the two graphs and the nodeMap between them. 
			:rtype: double, list[tuple(size_t, size_t)]
	
			.. seealso:: list_of_edit_cost_options, list_of_method_options, list_of_init_options 
			.. note:: Make sure each parameter exists with your architecture and these lists :  list_of_edit_cost_options, list_of_method_options, list_of_init_options. The structure of graphs must be similar as GXL. 
			
		"""
		if self.is_initialized() :
			self.restart_env()
	
		g = self.add_nx_graph(g1, "")
		h = self.add_nx_graph(g2, "")
	
		self.set_edit_cost(edit_cost)
		self.init(init_option)
		
		self.set_method(method, options)
		self.init_method()
	
		resDistance = 0
		resMapping = []
		self.run_method(g, h)
		resDistance = self.get_upper_bound(g, h)
		resMapping = self.get_node_map(g, h)
	
		return resDistance, resMapping
	
	
	def compute_edit_distance_on_nx_graphs(self, dataset, classes, edit_cost, method, options, init_option="EAGER_WITHOUT_SHUFFLED_COPIES") :
		"""
	
			Computes all the edit distance between each NX graphs on the dataset. 
			
			:param dataset: The list of graphs to add and compute
			:param classes: The classe of all the graph, can be an empty string
			:param edit_cost: The name of the edit cost function
			:param method: The name of the computation method
			:param options: The options of the method (like bash options), an empty string by default
			:param init_option:  The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
			:type dataset: list[networksx.graph]
			:type classes: string
			:type edit_cost: string
			:type method: string
			:type options: string
			:type init_option: string
			:return: Two matrix, the first with edit distances between graphs and the second the nodeMap between graphs. The result between g and h is one the [g][h] coordinates.
			:rtype: list[list[double]], list[list[list[tuple(size_t, size_t)]]]
	
			.. seealso:: list_of_edit_cost_options, list_of_method_options, list_of_init_options
			.. note:: Make sure each parameter exists with your architecture and these lists :  list_of_edit_cost_options, list_of_method_options, list_of_init_options. The structure of graphs must be similar as GXL. 
			
		"""
		if self.is_initialized() :
			self.restart_env()
	
		print("Loading graphs in progress...")
		for graph in dataset :
			self.add_nx_graph(graph, classes)
		listID = self.graph_ids()
		print("Graphs loaded ! ")
		print("Number of graphs = " + str(listID[1]))
	
		self.set_edit_cost(edit_cost)
		print("Initialization in progress...")
		self.init(init_option)
		print("Initialization terminated !")
		
		self.set_method(method, options)
		self.init_method()
	
		resDistance = [[]]
		resMapping = [[]]
		for g in range(listID[0], listID[1]) :
			print("Computation between graph " + str(g) + " with all the others including himself.")
			for h in range(listID[0], listID[1]) :
				#print("Computation between graph " + str(g) + " and graph " + str(h))
				self.run_method(g, h)
				resDistance[g][h] = self.get_upper_bound(g, h)
				resMapping[g][h] = self.get_node_map(g, h)
	
		print("Finish ! The return contains edit distances and NodeMap but you can check the result with graphs'ID until you restart the environment")
		return resDistance, resMapping
		
	
	def compute_edit_distance_on_GXl_graphs(self, path_folder, path_XML, edit_cost, method, options="", init_option="EAGER_WITHOUT_SHUFFLED_COPIES") :
		"""
			Computes all the edit distance between each GXL graphs on the folder and the XMl file. 
			
			:param path_folder: The folder's path which contains GXL graphs
			:param path_XML: The XML's path which indicates which graphes you want to load
			:param edit_cost: The name of the edit cost function
			:param method: The name of the computation method
			:param options: The options of the method (like bash options), an empty string by default
			:param init_option:  The name of the init option, "EAGER_WITHOUT_SHUFFLED_COPIES" by default
			:type path_folder: string
			:type path_XML: string
			:type edit_cost: string
			:type method: string
			:type options: string
			:type init_option: string
			:return: The list of the first and last-1 ID of graphs
			:rtype: tuple(size_t, size_t)
	
			.. seealso:: list_of_edit_cost_options, list_of_method_options, list_of_init_options
			.. note:: Make sure each parameter exists with your architecture and these lists : list_of_edit_cost_options, list_of_method_options, list_of_init_options. 
			
		"""
	
		if self.is_initialized() :
			self.restart_env()
	
		print("Loading graphs in progress...")
		self.load_GXL_graphs(path_folder, path_XML)
		listID = self.graph_ids()
		print("Graphs loaded ! ")
		print("Number of graphs = " + str(listID[1]))
	
		self.set_edit_cost(edit_cost)
		print("Initialization in progress...")
		self.init(init_option)
		print("Initialization terminated !")
		
		self.set_method(method, options)
		self.init_method()
	
		#res = []
		for g in range(listID[0], listID[1]) :
			print("Computation between graph " + str(g) + " with all the others including himself.")
			for h in range(listID[0], listID[1]) :
				#print("Computation between graph " + str(g) + " and graph " + str(h))
				self.run_method(g,h)
				#res.append((get_upper_bound(g,h), get_node_map(g,h), get_runtime(g,h)))
				
		#return res
	
		print ("Finish ! You can check the result with each ID of graphs ! There are in the return")
		print ("Please don't restart the environment or recall this function, you will lose your results !")
		return listID
	
	
	def get_num_node_labels(self):
		"""
			Returns the number of node labels.
			
			:return: Number of pairwise different node labels contained in the environment.
			:rtype: size_t
			
			.. note:: If 1 is returned, the nodes are unlabeled.
		"""
		return self.c_env.getNumNodeLabels()
	

	def get_node_label(self, label_id):
		"""
			Returns node label.
			
			:param label_id: ID of node label that should be returned. Must be between 1 and get_num_node_labels().
			:type label_id: size_t
			:return: Node label for selected label ID.
			:rtype: dict{string : string}
	 	"""
		return decode_your_map(self.c_env.getNodeLabel(label_id))


	def get_num_edge_labels(self):
		"""
			Returns the number of edge labels.
			
			:return: Number of pairwise different edge labels contained in the environment.
			:rtype: size_t
			
			.. note:: If 1 is returned, the edges are unlabeled.
	 	"""
		return self.c_env.getNumEdgeLabels()


	def get_edge_label(self, label_id):
		"""
			Returns edge label.
			
			:param label_id: ID of edge label that should be returned. Must be between 1 and get_num_edge_labels().
			:type label_id: size_t
			:return: Edge label for selected label ID.
			:rtype: dict{string : string}
	 	"""
		return decode_your_map(self.c_env.getEdgeLabel(label_id))
	
	
# 	def get_num_nodes(self, graph_id):
# 		"""
# 			Returns the number of nodes.
# 		
# 			:param graph_id: ID of an input graph that has been added to the environment.
# 			:type graph_id: size_t
# 			:return: Number of nodes in the graph.
# 			:rtype: size_t
# 		 """
# 		return self.c_env.getNumNodes(graph_id)

	def get_avg_num_nodes(self):
		"""
			Returns average number of nodes.
			 
			:return: Average number of nodes of the graphs contained in the environment.
			:rtype: double
		""" 
		return self.c_env.getAvgNumNodes()

	def get_node_rel_cost(self, node_label_1, node_label_2):
		"""
			Returns node relabeling cost.
			
			:param node_label_1: First node label.
			:param node_label_2: Second node label.
			:type node_label_1: dict{string : string}
			:type node_label_2: dict{string : string}
			:return: Node relabeling cost for the given node labels.
			:rtype: double
	 	"""
		return self.c_env.getNodeRelCost(encode_your_map(node_label_1), encode_your_map(node_label_2))


	def get_node_del_cost(self, node_label):
		"""
			Returns node deletion cost.
			
			:param node_label: Node label.
			:type node_label: dict{string : string}
			:return: Cost of deleting node with given label.
			:rtype: double
	 	"""
		return self.c_env.getNodeDelCost(encode_your_map(node_label))


	def get_node_ins_cost(self, node_label):
		"""
			Returns node insertion cost.
			
			:param node_label: Node label.
			:type node_label: dict{string : string}
			:return: Cost of inserting node with given label.
			:rtype: double
	 	"""
		return self.c_env.getNodeInsCost(encode_your_map(node_label))


	def get_median_node_label(self, node_labels):
		"""
			Computes median node label.
			
			:param node_labels: The node labels whose median should be computed.
			:type node_labels: list[dict{string : string}]
			:return: Median of the given node labels.
			:rtype: dict{string : string}
		"""
		node_labels_b = [encode_your_map(node_label) for node_label in node_labels]
		return decode_your_map(self.c_env.getMedianNodeLabel(node_labels_b))


	def get_edge_rel_cost(self, edge_label_1, edge_label_2):
		"""
			Returns edge relabeling cost.
			
			:param edge_label_1: First edge label.
			:param edge_label_2: Second edge label.
			:type edge_label_1: dict{string : string}
			:type edge_label_2: dict{string : string}
			:return: Edge relabeling cost for the given edge labels.
			:rtype: double
	 	"""
		return self.c_env.getEdgeRelCost(encode_your_map(edge_label_1), encode_your_map(edge_label_2))


	def get_edge_del_cost(self, edge_label):
		"""
			Returns edge deletion cost.
			
			:param edge_label: Edge label.
			:type edge_label: dict{string : string}
			:return: Cost of deleting edge with given label.
			:rtype: double
	 	"""
		return self.c_env.getEdgeDelCost(encode_your_map(edge_label))


	def get_edge_ins_cost(self, edge_label):
		"""
			Returns edge insertion cost.
			
			:param edge_label: Edge label.
			:type edge_label: dict{string : string}
			:return: Cost of inserting edge with given label.
			:rtype: double
	 	"""
		return self.c_env.getEdgeInsCost(encode_your_map(edge_label))


	def get_median_edge_label(self, edge_labels):
		"""
			Computes median edge label.
			
			:param edge_labels: The edge labels whose median should be computed.
			:type edge_labels: list[dict{string : string}]
			:return: Median of the given edge labels.
			:rtype: dict{string : string}
		"""
		edge_labels_b = [encode_your_map(edge_label) for edge_label in edge_labels] 
		return decode_your_map(self.c_env.getMedianEdgeLabel(edge_label_b))
	
	
	def get_nx_graph(self, graph_id, adj_matrix=True, adj_lists=False, edge_list=False): # @todo
		"""
		Get graph with id `graph_id` in the form of the NetworkX Graph.

		Parameters
		----------
		graph_id : int
			ID of the selected graph.
			
		adj_matrix : bool
			Set to `True` to construct an adjacency matrix `adj_matrix` and a hash-map `edge_labels`, which has a key for each pair `(i,j)` such that `adj_matrix[i][j]` equals 1. No effect for now.
			
		adj_lists : bool
			No effect for now.
			
		edge_list : bool
			No effect for now.

		Returns
		-------
		NetworkX Graph object
			The obtained graph.
		"""
		graph = nx.Graph()
		graph.graph['id'] = graph_id

		nb_nodes = self.get_graph_num_nodes(graph_id)
		original_node_ids = self.get_original_node_ids(graph_id)
		node_labels = self.get_graph_node_labels(graph_id)
# 		print(original_node_ids)
# 		print(node_labels)
		graph.graph['original_node_ids'] = original_node_ids
		
		for node_id in range(0, nb_nodes):
			graph.add_node(node_id, **node_labels[node_id])
# 			graph.nodes[node_id]['original_node_id'] = original_node_ids[node_id]
			
		edges = self.get_graph_edges(graph_id)
		for (head, tail), labels in edges.items():
			graph.add_edge(head, tail, **labels)
# 		print(edges)
			
		return graph
	
	
	def get_init_type(self):
		"""
		Returns the initialization type of the last initialization in string.

		Returns
		-------
		string
			Initialization type in string.
		"""
		return self.c_env.getInitType().decode('utf-8')
	
	
# 	def get_node_cost(self, label1, label2):
# 		"""
# 		Returns node relabeling, insertion, or deletion cost.

# 		Parameters
# 		----------
# 		label1 : int
# 			First node label.
# 		
# 		label2 : int
# 			Second node label. 
# 			
# 		Returns
# 		-------
# 		Node relabeling cost if `label1` and `label2` are both different from `ged::dummy_label()`, node insertion cost if `label1` equals `ged::dummy_label` and `label2` does not, node deletion cost if `label1` does not equal `ged::dummy_label` and `label2` does, and 0 otherwise.
# 		"""
# 		return self.c_env.getNodeCost(label1, label2)
	
	
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
		if graph_id is None:
			graph_id = self.add_graph(graph_name, graph_class)
		else:
			self.clear_graph(graph_id)
		for node in nx_graph.nodes:
			self.add_node(graph_id, str(node), nx_graph.nodes[node])
		for edge in nx_graph.edges:
			self.add_edge(graph_id, str(edge[0]), str(edge[1]), nx_graph.get_edge_data(edge[0], edge[1]))
		return graph_id
	
	
	def compute_induced_cost(self, g_id, h_id, node_map):
		"""
		Computes the edit cost between two graphs induced by a node map.

		Parameters
		----------
		g_id : int
			ID of input graph.
		h_id : int
			ID of input graph.
		node_map: gklearn.ged.env.NodeMap.
			The NodeMap instance whose reduced cost will be computed and re-assigned.

		Returns
		-------
		None.		
		"""
		relation = []
		node_map.as_relation(relation)
# 		print(relation)
		dummy_node = get_dummy_node()
# 		print(dummy_node)
		for i, val in enumerate(relation):
			val1 = dummy_node if val[0] == np.inf else val[0]
			val2 = dummy_node if val[1] == np.inf else val[1]
			relation[i] = tuple((val1, val2))
# 		print(relation)
		induced_cost = self.c_env.computeInducedCost(g_id, h_id, relation)
		node_map.set_induced_cost(induced_cost)

	
#####################################################################
##LISTS OF EDIT COST FUNCTIONS, METHOD COMPUTATION AND INIT OPTIONS##
#####################################################################

list_of_edit_cost_options = get_edit_cost_options()
list_of_method_options = get_method_options()
list_of_init_options = get_init_options()


#####################
##ERRORS MANAGEMENT##
#####################

class Error(Exception):
	"""
		Class for error's management. This one is general. 
	"""
	pass


class EditCostError(Error) :
	"""
		Class for Edit Cost Error. Raise an error if an edit cost function doesn't exist in the library (not in list_of_edit_cost_options).

		:attribute message: The message to print when an error is detected.
		:type message: string
	"""
	def __init__(self, message):
		"""
			Inits the error with its message. 

			:param message: The message to print when the error is detected
			:type message: string
		"""
		self.message = message
	
	
class MethodError(Error) :
	"""
		Class for Method Error. Raise an error if a computation method doesn't exist in the library (not in list_of_method_options).

		:attribute message: The message to print when an error is detected.
		:type message: string
	"""
	def __init__(self, message):
		"""
			Inits the error with its message. 

			:param message: The message to print when the error is detected
			:type message: string
		"""
		self.message = message


class InitError(Error) :
	"""
		Class for Init Error. Raise an error if an init option doesn't exist in the library (not in list_of_init_options).

		:attribute message: The message to print when an error is detected.
		:type message: string
	"""
	def __init__(self, message):
		"""
			Inits the error with its message. 

			:param message: The message to print when the error is detected
			:type message: string
		"""
		self.message = message


#########################################
##PYTHON FUNCTIONS FOR SOME COMPUTATION##
#########################################

def encode_your_map(map_u):
	"""
		Encodes Python unicode strings in dictionnary `map` to utf-8 byte strings for C++ functions.

		:param map_b: The map to encode
		:type map_b: dict{string : string}
		:return: The encoded map
		:rtype: dict{'b'string : 'b'string}

		.. note:: This function is used for type connection.  
		
	"""
	res = {}
	for key, value in map_u.items():
		res[key.encode('utf-8')] = value.encode('utf-8')
	return res


def decode_your_map(map_b):
	"""
		Decodes utf-8 byte strings in `map` from C++ functions to Python unicode strings. 

		:param map_b: The map to decode
		:type map_b: dict{'b'string : 'b'string}
		:return: The decoded map
		:rtype: dict{string : string}

		.. note:: This function is used for type connection.  
		
	"""
	res = {}
	for key, value in map_b.items():
		res[key.decode('utf-8')] = value.decode('utf-8')
	return res


def decode_graph_edges(map_edge_b):	
	"""
	Decode utf-8 byte strings in graph edges `map` from C++ functions to Python unicode strings. 

	Parameters
	----------
	map_edge_b : dict{tuple(size_t, size_t) : dict{'b'string : 'b'string}}
		The map to decode.

	Returns
	-------
	dict{tuple(size_t, size_t) : dict{string : string}}
		The decoded map.
	
	Notes
	-----
	This is a helper function for function `GEDEnv.get_graph_edges()`.
	"""
	map_edges = {}
	for key, value in map_edge_b.items():
		map_edges[key] = decode_your_map(value)			
	return map_edges







# cdef extern from "src/GedLibBind.h" namespace "shapes":
#	 cdef cppclass Rectangle:
#		 Rectangle() except +
#		 Rectangle(int, int, int, int) except +
#		 int x0, y0, x1, y1
#		 int getArea()
#		 void getSize(int* width, int* height)
#		 void move(int, int)
		

# # Create a Cython extension type which holds a C++ instance
# # as an attribute and create a bunch of forwarding methods
# # Python extension type.
# cdef class PyRectangle:
#	 cdef Rectangle c_rect  # Hold a C++ instance which we're wrapping

#	 def __cinit__(self, int x0, int y0, int x1, int y1):
#		 self.c_rect = Rectangle(x0, y0, x1, y1)

#	 def get_area(self):
#		 return self.c_rect.getArea()

#	 def get_size(self):
#		 cdef int width, height
#		 self.c_rect.getSize(&width, &height)
#		 return width, height

#	 def move(self, dx, dy):
#		 self.c_rect.move(dx, dy)

#	 # Attribute access
#	 @property
#	 def x0(self):
#		 return self.c_rect.x0
#	 @x0.setter
#	 def x0(self, x0):
#		 self.c_rect.x0 = x0

#	 # Attribute access
#	 @property
#	 def x1(self):
#		 return self.c_rect.x1
#	 @x1.setter
#	 def x1(self, x1):
#		 self.c_rect.x1 = x1

#	 # Attribute access
#	 @property
#	 def y0(self):
#		 return self.c_rect.y0
#	 @y0.setter
#	 def y0(self, y0):
#		 self.c_rect.y0 = y0

#	 # Attribute access
#	 @property
#	 def y1(self):
#		 return self.c_rect.y1
#	 @y1.setter
#	 def y1(self, y1):
#		 self.c_rect.y1 = y1