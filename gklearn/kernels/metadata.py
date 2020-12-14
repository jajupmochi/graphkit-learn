#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:11:08 2020

@author: ljia
"""

# The metadata of all graph kernels.
GRAPH_KERNELS = {
	### based on walks.
	'common walk': '',
	'marginalized': '',
	'sylvester equation': '',
	'fixed point': '',
	'conjugate gradient': '',
	'spectral decomposition': '',
	### based on paths.
	'shortest path': '',
	'structural shortest path': '',
	'path up to length h': '',
	### based on non-linear patterns.
	'weisfeiler-lehman subtree': '',
	'treelet': '',
	}


def list_of_graph_kernels():
	"""List names of all graph kernels.

	Returns
	-------
	list
		The list of all graph kernels.
	"""
	return [i for i in GRAPH_KERNELS]