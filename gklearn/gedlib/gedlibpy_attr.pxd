# distutils: language = c++

"""
	Python GedLib module for the AttrLabel type
	======================

	This module allows using a C++ library for edit distance between graphs (GedLib) with Python.


	Authors
	-------------------

	Linlin Jia
	David Blumenthal
	Natacha Lambert

	Copyright (C) 2019-2025 by all the authors

	Classes & Functions
	-------------------

"""

#################################
##DECLARATION OF C++ INTERFACES##
#################################


#Types imports for C++ compatibility
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from libcpp.pair cimport pair

#Long unsigned int equivalent
cimport numpy as cnp

ctypedef cnp.npy_uint32 UINT32_t


cdef extern from "src/gedlib_bind.hpp" namespace "pyged":

	cdef cppclass PyGEDEnvAttr:
		# PyGEDEnvAttr() except +
		PyGEDEnvAttr() except +
		bool isInitialized() except +
		void restartEnv() except +
		void loadGXLGraph(
				string pathFolder, string pathXML, bool node_type, bool edge_type
		) except +
		pair[size_t, size_t] getGraphIds() except +
		vector[size_t] getAllGraphIds() except +
		string getGraphClass(size_t id) except +
		string getGraphName(size_t id) except +
		size_t addGraph(string name, string classe) except +
		# void addNode(  # todo: need conversion in ipp file
		# 		size_t graphId, string nodeId, map[string, string] nodeLabel
		# ) except +
		void addNode(
				size_t graphId,
				string nodeId,
				unordered_map[string, string] str_map,
				unordered_map[string, int] int_map,
				unordered_map[string, double] float_map,
				unordered_map[string, vector[string]] list_str_map,
				unordered_map[string, vector[int]] list_int_map,
				unordered_map[string, vector[double]] list_float_map
		) except +
		# void addEdge(
		# 		size_t graphId, string tail, string head, map[string, string] edgeLabel,
		# 		bool ignoreDuplicates
		# ) except +
		void addEdge(
				size_t graphId,
				string tail,
				string head,
				unordered_map[string, string] str_map,
				unordered_map[string, int] int_map,
				unordered_map[string, double] float_map,
				unordered_map[string, vector[string]] list_str_map,
				unordered_map[string, vector[int]] list_int_map,
				unordered_map[string, vector[double]] list_float_map,
				bool ignoreDuplicates
		) except +
		void clearGraph(size_t graphId) except +
		size_t getGraphInternalId(size_t graphId) except +
		size_t getGraphNumNodes(size_t graphId) except +
		size_t getGraphNumEdges(size_t graphId) except +
		vector[string] getGraphOriginalNodeIds(size_t graphId) except +
		# vector[map[string, string]] getGraphNodeLabels(size_t graphId) except +
		# map[pair[size_t, size_t], map[string, string]] getGraphEdges(
		# 		size_t graphId
		# ) except +
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
		vector[pair[size_t, size_t]] getNodeMap(size_t g, size_t h) except +
		vector[vector[int]] getAssignmentMatrix(size_t g, size_t h) except +
		vector[vector[cnp.npy_uint64]] getAllMap(size_t g, size_t h) except +
		double getRuntime(size_t g, size_t h) except +
		bool quasimetricCosts() except +
		vector[vector[size_t]] hungarianLSAP(vector[vector[size_t]] matrixCost) except +
		vector[vector[double]] hungarianLSAPE(
				vector[vector[double]] matrixCost
		) except +
		# added by Linlin Jia.
		size_t getNumGraphs() except +
		size_t getNumNodeLabels() except +
		# map[string, string] getNodeLabel(size_t label_id) except +
		size_t getNumEdgeLabels() except +
		# map[string, string] getEdgeLabel(size_t label_id) except +
		# 		size_t getNumNodes(size_t graph_id) except +
		double getAvgNumNodes() except +
		# double getNodeRelCost(
		# 		map[string, string] & node_label_1, map[string, string] & node_label_2
		# ) except +
		# double getNodeDelCost(map[string, string] & node_label) except +
		# double getNodeInsCost(map[string, string] & node_label) except +
		# map[string, string] getMedianNodeLabel(
		# 		vector[map[string, string]] & node_labels
		# ) except +
		# double getEdgeRelCost(
		# 		map[string, string] & edge_label_1, map[string, string] & edge_label_2
		# ) except +
		# double getEdgeDelCost(map[string, string] & edge_label) except +
		# double getEdgeInsCost(map[string, string] & edge_label) except +
		# map[string, string] getMedianEdgeLabel(
		# 		vector[map[string, string]] & edge_labels
		# ) except +
		string getInitType() except +
		# 		double getNodeCost(size_t label1, size_t label2) except +
		double computeInducedCost(
				size_t g_id, size_t h_id, vector[pair[size_t, size_t]]
		) except +


#############################
##External Libs Import     ##
#############################

from libraries_import import lib1, lib2, lib3, lib4

#############################
##CYTHON WRAPPER INTERFACES##
#############################

# @cython.auto_pickle(True)
cdef class GEDEnvAttr:
	"""Cython wrapper class for C++ class PyGEDEnvAttr
	"""
	# 	cdef PyGEDEnvAttr c_env  # Hold a C++ instance which we're wrapping
	cdef PyGEDEnvAttr * c_env  # hold a pointer to the C++ instance which we're wrapping
