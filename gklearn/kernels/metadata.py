#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:11:08 2020

@author: ljia
"""
from gklearn.kernels.common_walk import CommonWalk
from gklearn.kernels.marginalized import Marginalized
from gklearn.kernels.sylvester_equation import SylvesterEquation
from gklearn.kernels.conjugate_gradient import ConjugateGradient
from gklearn.kernels.fixed_point import FixedPoint
from gklearn.kernels.spectral_decomposition import SpectralDecomposition
from gklearn.kernels.shortest_path import ShortestPath
from gklearn.kernels.structural_sp import StructuralSP
from gklearn.kernels.path_up_to_h import PathUpToH
from gklearn.kernels.treelet import Treelet
from gklearn.kernels.weisfeiler_lehman import WLSubtree


# The metadata of all graph kernels.
GRAPH_KERNELS = {
	### based on walks.
	'common walk': CommonWalk,
	'marginalized': Marginalized,
	'sylvester equation': SylvesterEquation,
	'fixed point': FixedPoint,
	'conjugate gradient': ConjugateGradient,
	'spectral decomposition': SpectralDecomposition,
	### based on paths.
	'shortest path': ShortestPath,
	'structural shortest path': StructuralSP,
	'path up to length h': PathUpToH,
	### based on non-linear patterns.
	'weisfeiler-lehman subtree': WLSubtree,
	'treelet': Treelet,
	}


def list_of_graph_kernels():
	"""List names of all graph kernels.

	Returns
	-------
	list
		The list of all graph kernels.
	"""
	return [i for i in GRAPH_KERNELS]