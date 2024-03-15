#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:52:23 2020

@author: ljia
"""
from gklearn.ged.edit_costs import EditCost


class Constant(EditCost):
	"""Implements constant edit cost functions.
	"""

	
	def __init__(self, node_ins_cost=1, node_del_cost=1, node_rel_cost=1, edge_ins_cost=1, edge_del_cost=1, edge_rel_cost=1):
		self._node_ins_cost = node_ins_cost
		self._node_del_cost = node_del_cost
		self._node_rel_cost = node_rel_cost
		self._edge_ins_cost = edge_ins_cost
		self._edge_del_cost = edge_del_cost
		self._edge_rel_cost = edge_rel_cost
		
		
	def node_ins_cost_fun(self, node_label):
		return self._node_ins_cost
	
	
	def node_del_cost_fun(self, node_label):
		return self._node_del_cost
	
	
	def node_rel_cost_fun(self, node_label_1, node_label_2):
		if node_label_1 != node_label_2:
			return self._node_rel_cost
		return 0
	
	
	def edge_ins_cost_fun(self, edge_label):
		return self._edge_ins_cost
	
	
	def edge_del_cost_fun(self, edge_label):
		return self._edge_del_cost
	
	
	def edge_rel_cost_fun(self, edge_label_1, edge_label_2):
		if edge_label_1 != edge_label_2:
			return self._edge_rel_cost
		return 0