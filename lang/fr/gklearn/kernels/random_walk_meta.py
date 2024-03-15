#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:55:17 2020

@author: ljia

@references: 

	[1] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11(Apr):1201–1242, 2010.
"""

import networkx as nx
from gklearn.utils import SpecialLabel
from gklearn.kernels import GraphKernel


class RandomWalkMeta(GraphKernel):
	
	
	def __init__(self, **kwargs):
		GraphKernel.__init__(self)
		self._weight = kwargs.get('weight', 1)
		self._p = kwargs.get('p', None)
		self._q = kwargs.get('q', None)
		self._edge_weight = kwargs.get('edge_weight', None)
		self._ds_infos = kwargs.get('ds_infos', {})
		
		
	def _compute_gm_series(self):
		pass


	def _compute_gm_imap_unordered(self):
		pass
	
		
	def _compute_kernel_list_series(self, g1, g_list):
		pass

	
	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		pass
	
	
	def _compute_single_kernel_series(self, g1, g2):
		pass
	
	
	def _check_graphs(self, Gn):
		# remove graphs with no edges, as no walk can be found in their structures, 
		# so the weight matrix between such a graph and itself might be zero.
		for g in Gn:
			if nx.number_of_edges(g) == 0:
				raise Exception('Graphs must contain edges to construct weight matrices.')
				
	
	def _check_edge_weight(self, G0, verbose):
		eweight = None
		if self._edge_weight is None:
			if verbose >= 2:
				print('\n None edge weight is specified. Set all weight to 1.\n')
		else:
			try:
				some_weight = list(nx.get_edge_attributes(G0, self._edge_weight).values())[0]
				if isinstance(some_weight, float) or isinstance(some_weight, int):
					eweight = self._edge_weight
				else:
					if verbose >= 2:
						print('\n Edge weight with name %s is not float or integer. Set all weight to 1.\n' % self._edge_weight)
			except:
				if verbose >= 2:
					print('\n Edge weight with name "%s" is not found in the edge attributes. Set all weight to 1.\n' % self._edge_weight)
		
		self._edge_weight = eweight
				
		
	def _add_dummy_labels(self, Gn):
		if len(self._node_labels) == 0 or (len(self._node_labels) == 1 and self._node_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self._node_labels = [SpecialLabel.DUMMY]
		if len(self._edge_labels) == 0 or (len(self._edge_labels) == 1 and self._edge_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_edge_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self._edge_labels = [SpecialLabel.DUMMY]