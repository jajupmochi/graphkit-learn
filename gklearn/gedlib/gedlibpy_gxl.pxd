# distutils: language = c++

"""
	Python GedLib module for the GXLLabel version (string labels)
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
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.pair cimport pair

#Long unsigned int equivalent
cimport numpy as cnp

ctypedef cnp.npy_uint32 UINT32_t

from common_bind import *


cdef extern from "src/gedlib_bind.hpp" namespace "pyged":


	cdef cppclass PyGEDEnvGXL:
		PyGEDEnvGXL() except +
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
		void addNode(
				size_t graphId, string nodeId, map[string, string] nodeLabel
		) except +
		void addEdge(
				size_t graphId, string tail, string head, map[string, string] edgeLabel,
				bool ignoreDuplicates
		) except +
		void clearGraph(size_t graphId) except +
		size_t getGraphInternalId(size_t graphId) except +
		size_t getGraphNumNodes(size_t graphId) except +
		size_t getGraphNumEdges(size_t graphId) except +
		vector[string] getGraphOriginalNodeIds(size_t graphId) except +
		vector[map[string, string]] getGraphNodeLabels(size_t graphId) except +
		map[pair[size_t, size_t], map[string, string]] getGraphEdges(
				size_t graphId
		) except +
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
		size_t getNumNodeLabels() except +
		map[string, string] getNodeLabel(size_t label_id) except +
		size_t getNumEdgeLabels() except +
		map[string, string] getEdgeLabel(size_t label_id) except +
		# 		size_t getNumNodes(size_t graph_id) except +
		double getAvgNumNodes() except +
		double getNodeRelCost(
				map[string, string] & node_label_1, map[string, string] & node_label_2
		) except +
		double getNodeDelCost(map[string, string] & node_label) except +
		double getNodeInsCost(map[string, string] & node_label) except +
		map[string, string] getMedianNodeLabel(
				vector[map[string, string]] & node_labels
		) except +
		double getEdgeRelCost(
				map[string, string] & edge_label_1, map[string, string] & edge_label_2
		) except +
		double getEdgeDelCost(map[string, string] & edge_label) except +
		double getEdgeInsCost(map[string, string] & edge_label) except +
		map[string, string] getMedianEdgeLabel(
				vector[map[string, string]] & edge_labels
		) except +
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
cdef class GEDEnvGXL:
	"""Cython wrapper class for C++ class PyGEDEnvGXL
	"""
	# 	cdef PyGEDEnv c_env  # Hold a C++ instance which we're wrapping
	cdef PyGEDEnvGXL * c_env  # hold a pointer to the C++ instance which we're wrapping


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