#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:10:06 2020

@author: ljia
"""
import numpy as np
import networkx as nx
import random


class GraphSynthesizer(object):
	
	
	def __init__(self, g_type=None, *args, **kwargs):
		if g_type == 'unified':
			self._graphs = self.unified_graphs(*args, *kwargs)
		else:
			self._graphs = None
	
	
	def random_graph(self, num_nodes, num_edges, num_node_labels=0, num_edge_labels=0, seed=None, directed=False, max_num_edges=None, all_edges=None):
		g = nx.Graph()
		if num_node_labels > 0:
			node_labels = np.random.randint(0, high=num_node_labels, size=num_nodes)
			for i in range(0, num_nodes):
				g.add_node(str(i), atom=node_labels[i]) # @todo: update "atom".
		else:
			for i in range(0, num_nodes):
				g.add_node(str(i))

		if num_edge_labels > 0:
			edge_labels = np.random.randint(0, high=num_edge_labels, size=num_edges)				
			for idx, i in enumerate(random.sample(range(0, max_num_edges), num_edges)):
				node1, node2 = all_edges[i]
				g.add_edge(str(node1), str(node2), bond_type=edge_labels[idx])  # @todo: update "bond_type".
		else:
			for i in random.sample(range(0, max_num_edges), num_edges):
				node1, node2 = all_edges[i]
				g.add_edge(str(node1), str(node2))
		
		return g
	
	
	def unified_graphs(self, num_graphs=1000, num_nodes=20, num_edges=40, num_node_labels=0, num_edge_labels=0, seed=None, directed=False):
		max_num_edges = int((num_nodes - 1) * num_nodes / 2)
		if num_edges > max_num_edges:
			raise Exception('Too many edges.')
		all_edges = [(i, j) for i in range(0, num_nodes) for j in range(i + 1, num_nodes)] # @todo: optimize. No directed graphs.
			
		graphs = []
		for idx in range(0, num_graphs):		
			graphs.append(self.random_graph(num_nodes, num_edges, num_node_labels=num_node_labels, num_edge_labels=num_edge_labels, seed=seed, directed=directed, max_num_edges=max_num_edges, all_edges=all_edges))
			
		return graphs
	
	
	@property
	def graphs(self):
		return self._graphs