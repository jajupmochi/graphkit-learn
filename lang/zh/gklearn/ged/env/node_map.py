#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:31:26 2020

@author: ljia
"""
import numpy as np
from gklearn.utils import dummy_node, undefined_node


class NodeMap(object):
	
	def __init__(self, num_nodes_g, num_nodes_h):
		self._forward_map = [undefined_node()] * num_nodes_g
		self._backward_map = [undefined_node()] * num_nodes_h
		self._induced_cost = np.inf
		
		
	def clear(self):
		"""
	/*!
	 * @brief Clears the node map.
	 */
		"""
		self._forward_map = [undefined_node() for i in range(len(self._forward_map))]
		self._backward_map = [undefined_node() for i in range(len(self._backward_map))]
		
		
	def num_source_nodes(self):
		return len(self._forward_map)
	
	
	def num_target_nodes(self):
		return len(self._backward_map)
	
	
	def image(self, node):
		if node < len(self._forward_map):
			return self._forward_map[node]
		else:
			raise Exception('The node with ID ', str(node), ' is not contained in the source nodes of the node map.')
		return undefined_node()
	
	
	def pre_image(self, node):
		if node < len(self._backward_map):
			return self._backward_map[node]
		else:
			raise Exception('The node with ID ', str(node), ' is not contained in the target nodes of the node map.')
		return undefined_node()
	
	
	def as_relation(self, relation):
		relation.clear()
		for i in range(0, len(self._forward_map)):
			k = self._forward_map[i]
			if k != undefined_node():
				relation.append(tuple((i, k)))
		for k in range(0, len(self._backward_map)):
			i = self._backward_map[k]
			if i == dummy_node():
				relation.append(tuple((i, k)))
	
	
	def add_assignment(self, i, k):
		if i != dummy_node():
			if i < len(self._forward_map):
				self._forward_map[i] = k
			else:
				raise Exception('The node with ID ', str(i), ' is not contained in the source nodes of the node map.')
		if k != dummy_node():
			if k < len(self._backward_map):
				self._backward_map[k] = i
			else:
				raise Exception('The node with ID ', str(k), ' is not contained in the target nodes of the node map.')
	
	
	def set_induced_cost(self, induced_cost):
		self._induced_cost = induced_cost
		
		
	def induced_cost(self):
		return self._induced_cost
	
	
	@property
	def forward_map(self):
		return self._forward_map

	@forward_map.setter
	def forward_map(self, value):
		self._forward_map = value	
		
		
	@property
	def backward_map(self):
		return self._backward_map

	@backward_map.setter
	def backward_map(self, value):
		self._backward_map = value	