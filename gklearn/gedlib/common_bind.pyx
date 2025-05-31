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

#############################
##CYTHON WRAPPER INTERFACES##
#############################


def get_edit_cost_options():
	"""
		Searchs the differents edit cost functions and returns the result.

		:return: The list of edit cost functions
		:rtype: list[string]

		.. warning:: This function is useless for an external use. Please use directly list_of_edit_cost_options.
		.. note:: Prefer the list_of_edit_cost_options attribute of this module.
	"""

	return [option.decode('utf-8') for option in getEditCostStringOptions()]


def get_method_options():
	"""
		Searchs the differents method for edit distance computation between graphs and returns the result.

		:return: The list of method to compute the edit distance between graphs
		:rtype: list[string]

		.. warning:: This function is useless for an external use. Please use directly list_of_method_options.
		.. note:: Prefer the list_of_method_options attribute of this module.
	"""
	return [option.decode('utf-8') for option in getMethodStringOptions()]


def get_init_options():
	"""
		Searchs the differents initialization parameters for the environment computation for graphs and returns the result.

		:return: The list of options to initialize the computation environment
		:rtype: list[string]

		.. warning:: This function is useless for an external use. Please use directly list_of_init_options.
		.. note:: Prefer the list_of_init_options attribute of this module.
	"""
	return [option.decode('utf-8') for option in getInitStringOptions()]


def get_dummy_node():
	"""
		Returns the ID of a dummy node.

		:return: The ID of the dummy node (18446744073709551614 for my computer, the hugest number possible)
		:rtype: size_t

		.. note:: A dummy node is used when a node isn't associated to an other node.
	"""
	return getDummyNode()


#####################################################################
##LISTS OF EDIT COST FUNCTIONS, METHOD COMPUTATION AND INIT OPTIONS##
#####################################################################

list_of_edit_cost_options = get_edit_cost_options()
list_of_method_options = get_method_options()
list_of_init_options = get_init_options()
