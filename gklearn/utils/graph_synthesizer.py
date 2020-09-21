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
	
	
	def __init__(self):
		pass
	
	
	def unified_graphs(self, num_graphs=1000, num_nodes=100, num_edges=196, num_node_labels=0, num_edge_labels=0, seed=None, directed=False):
		max_num_edges = int((num_nodes - 1) * num_nodes / 2)
		if num_edges > max_num_edges:
			raise Exception('Too many edges.')
		all_edges = [(i, j) for i in range(0, num_nodes) for j in range(i + 1, num_nodes)] # @todo: optimize. No directed graphs.
			
		graphs = []
		for idx in range(0, num_graphs):
			g = nx.Graph()
			if num_node_labels > 0:
				for i in range(0, num_nodes):
					node_labels = np.random.randint(0, high=num_node_labels, size=num_nodes)
					g.add_node(str(i), node_label=node_labels[i])
			else:
				for i in range(0, num_nodes):
					g.add_node(str(i))

			if num_edge_labels > 0:
				edge_labels = np.random.randint(0, high=num_edge_labels, size=num_edges)				
				for i in random.sample(range(0, max_num_edges), num_edges):
					node1, node2 = all_edges[i]
					g.add_edge(node1, node2, edge_label=edge_labels[i])
			else:
				for i in random.sample(range(0, max_num_edges), num_edges):
					node1, node2 = all_edges[i]
					g.add_edge(node1, node2)
		
			graphs.append(g)
			
		return graphs