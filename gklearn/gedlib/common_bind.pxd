# distutils: language = c++

"""
	Python GedLib module for the common definition bindings
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

from libcpp.vector cimport vector
from libcpp.string cimport string


#################################
##DECLARATION OF C++ INTERFACES##
#################################


cdef extern from "src/gedlib_bind.hpp" namespace "pyged":
	cdef vector[string] getEditCostStringOptions() except +
	cdef vector[string] getMethodStringOptions() except +
	cdef vector[string] getInitStringOptions() except +
	cdef size_t getDummyNode() except +
