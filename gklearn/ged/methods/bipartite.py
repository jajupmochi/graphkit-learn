#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:09:29 2020

@author: ljia
"""
import numpy as np
import networkx as nx
from gklearn.ged.methods import LSAPEBasedMethod
from gklearn.ged.util import LSAPESolver
from gklearn.utils import SpecialLabel


class Bipartite(LSAPEBasedMethod):
	
	
	def __init__(self, ged_data):
		super().__init__(ged_data)
		self._compute_lower_bound = False
		
		
	###########################################################################
	# Inherited member functions from LSAPEBasedMethod.
	###########################################################################
	
	
	def _lsape_populate_instance(self, g, h, master_problem):
		# #ifdef _OPENMP
		for row_in_master in range(0, nx.number_of_nodes(g)):
			for col_in_master in range(0, nx.number_of_nodes(h)):
				master_problem[row_in_master, col_in_master] = self._compute_substitution_cost(g, h, row_in_master, col_in_master)
		for row_in_master in range(0, nx.number_of_nodes(g)):
			master_problem[row_in_master, nx.number_of_nodes(h) + row_in_master] = self._compute_deletion_cost(g, row_in_master)
		for col_in_master in range(0, nx.number_of_nodes(h)):
			master_problem[nx.number_of_nodes(g) + col_in_master, col_in_master] = self._compute_insertion_cost(h, col_in_master)

# 		for row_in_master in range(0, master_problem.shape[0]):
# 			for col_in_master in range(0, master_problem.shape[1]):
# 				if row_in_master < nx.number_of_nodes(g) and col_in_master < nx.number_of_nodes(h):
# 					master_problem[row_in_master, col_in_master] = self._compute_substitution_cost(g, h, row_in_master, col_in_master)
# 				elif row_in_master < nx.number_of_nodes(g):
# 					master_problem[row_in_master, nx.number_of_nodes(h)] = self._compute_deletion_cost(g, row_in_master)
# 				elif col_in_master < nx.number_of_nodes(h):
# 					master_problem[nx.number_of_nodes(g), col_in_master] = self._compute_insertion_cost(h, col_in_master)


	###########################################################################
	# Helper member functions.
	###########################################################################


	def _compute_substitution_cost(self, g, h, u, v):
		# Collect node substitution costs.
		cost = self._ged_data.node_cost(g.nodes[u]['label'], h.nodes[v]['label'])
		
		# Initialize subproblem.
		d1, d2 = g.degree[u], h.degree[v]
		subproblem = np.ones((d1 + d2, d1 + d2)) * np.inf
		subproblem[d1:, d2:] = 0
# 		subproblem = np.empty((g.degree[u] + 1, h.degree[v] + 1))
		
		# Collect edge deletion costs.
		i = 0 # @todo: should directed graphs be considered?
		for label in g[u].values(): # all u's neighbor
			subproblem[i, d2 + i] = self._ged_data.edge_cost(label['label'], SpecialLabel.DUMMY)
# 			subproblem[i, h.degree[v]] = self._ged_data.edge_cost(label['label'], SpecialLabel.DUMMY)
			i += 1
			
		# Collect edge insertion costs.
		i = 0 # @todo: should directed graphs be considered?
		for label in h[v].values(): # all u's neighbor
			subproblem[d1 + i, i] = self._ged_data.edge_cost(SpecialLabel.DUMMY, label['label'])
# 			subproblem[g.degree[u], i] = self._ged_data.edge_cost(SpecialLabel.DUMMY, label['label'])
			i += 1
			
		# Collect edge relabelling costs.
		i = 0
		for label1 in g[u].values():
			j = 0
			for label2 in h[v].values():
				subproblem[i, j] = self._ged_data.edge_cost(label1['label'], label2['label'])
				j += 1
			i += 1
				
		# Solve subproblem.
		subproblem_solver = LSAPESolver(subproblem)
		subproblem_solver.set_model(self._lsape_model)
		subproblem_solver.solve()
		
		# Update and return overall substitution cost.
		cost += subproblem_solver.minimal_cost()
		return cost
	
	
	def _compute_deletion_cost(self, g, v):
		# Collect node deletion cost.
		cost = self._ged_data.node_cost(g.nodes[v]['label'], SpecialLabel.DUMMY)
		
		# Collect edge deletion costs.
		for label in g[v].values():
			cost += self._ged_data.edge_cost(label['label'], SpecialLabel.DUMMY)
			
		# Return overall deletion cost.
		return cost
	
	
	def _compute_insertion_cost(self, g, v):
		# Collect node insertion cost.
		cost = self._ged_data.node_cost(SpecialLabel.DUMMY, g.nodes[v]['label'])
		
		# Collect edge insertion costs.
		for label in g[v].values():
			cost += self._ged_data.edge_cost(SpecialLabel.DUMMY, label['label'])
			
		# Return overall insertion cost.
		return cost