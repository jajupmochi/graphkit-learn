#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:08:24 2020

@author: ljia
"""
import random
import numpy as np

def test_get_nb_edit_operations_symbolic_cml():
	"""Test get_nb_edit_operations_symbolic_cml().
	"""
	"""**1.   Get dataset.**"""

	from gklearn.utils import Dataset
	
	# Predefined dataset name, use dataset "MUTAG".
	ds_name = 'MUTAG'
	
	# Initialize a Dataset.
	dataset = Dataset()
	# Load predefined dataset "MUTAG".
	dataset.load_predefined_dataset(ds_name)
	graph1 = dataset.graphs[0]
	graph2 = dataset.graphs[1]
	
	"""**2.  Compute graph edit distance.**"""
	
# 	try:
	# Initialize label costs randomly.
	node_label_costs, edge_label_costs = _initialize_label_costs(dataset)
	
	# Compute GEDs.
	pi_forward, pi_backward, dis, node_labels, edge_labels = _compute_ged(dataset, node_label_costs, edge_label_costs)
	
	
	# Compute numbers of edit operations.
	
	from gklearn.ged.util.util import get_nb_edit_operations_symbolic_cml
	
	n_edit_operations = get_nb_edit_operations_symbolic_cml(graph1, graph2, pi_forward, pi_backward, node_labels, edge_labels)
	
	assert np.abs((np.dot(np.concatenate((node_label_costs, edge_label_costs)), n_edit_operations) - dis) / dis) < 10e-6
	
# 	except Exception as exception:
# 		assert False, exception
		
		
def _initialize_label_costs(dataset):
	node_label_costs = _initialize_node_label_costs(dataset)
	edge_label_costs = _initialize_edge_label_costs(dataset)
	return node_label_costs, edge_label_costs
	
	
def _initialize_node_label_costs(dataset):
	# Get list of node labels.
	nls = dataset.get_all_node_labels()
	# Generate random costs.
	nb_nl = int((len(nls) * (len(nls) - 1)) / 2 + 2 * len(nls))
	rand_costs = random.sample(range(1, 10 * nb_nl + 1), nb_nl)
	rand_costs /= np.max(rand_costs)
			
	return rand_costs


def _initialize_edge_label_costs(dataset):
	# Get list of edge labels.
	els = dataset.get_all_edge_labels()
	# Generate random costs.
	nb_el = int((len(els) * (len(els) - 1)) / 2 + 2 * len(els))
	rand_costs = random.sample(range(1, 10 * nb_el + 1), nb_el)
	rand_costs /= np.max(rand_costs)
			
	return rand_costs


def _compute_ged(dataset, node_label_costs, edge_label_costs):
	from gklearn.ged.env import GEDEnv
	from gklearn.ged.util.util import label_costs_to_matrix
	import networkx as nx
			
	ged_env = GEDEnv() # initailize GED environment.
	ged_env.set_edit_cost('CONSTANT', # GED cost type.
	                      edit_cost_constants=[3, 3, 1, 3, 3, 1] # edit costs.
						  )
	for g in dataset.graphs:
		ged_env.add_nx_graph(g, '') # add graphs

	node_labels = ged_env.get_all_node_labels()
	edge_labels = ged_env.get_all_edge_labels()
	listID = ged_env.get_all_graph_ids() # get list IDs of graphs
	ged_env.set_label_costs(label_costs_to_matrix(node_label_costs, len(node_labels)), 
					  label_costs_to_matrix(edge_label_costs, len(edge_labels)))
	ged_env.init(init_type='LAZY_WITHOUT_SHUFFLED_COPIES') # initialize GED environment.
	options = {'initialization_method': 'RANDOM', # or 'NODE', etc.
	           'threads': 1 # parallel threads.
			   }
	ged_env.set_method('BIPARTITE', # GED method.
	                   options # options for GED method.
					   )
	ged_env.init_method() # initialize GED method.
	
	ged_env.run_method(listID[0], listID[1]) # run.
	
	pi_forward = ged_env.get_forward_map(listID[0], listID[1]) # forward map.
	pi_backward = ged_env.get_backward_map(listID[0], listID[1]) # backward map.
	dis = ged_env.get_upper_bound(listID[0], listID[1])	# GED bewteen two graphs.
	
	# make the map label correct (label remove map as np.inf)
	nodes1 = [n for n in dataset.graphs[0].nodes()]
	nodes2 = [n for n in dataset.graphs[1].nodes()]
	nb1 = nx.number_of_nodes(dataset.graphs[0])
	nb2 = nx.number_of_nodes(dataset.graphs[1])
	pi_forward = [nodes2[pi] if pi < nb2 else np.inf for pi in pi_forward]
	pi_backward = [nodes1[pi] if pi < nb1 else np.inf for pi in pi_backward]
	
	return pi_forward, pi_backward, dis, node_labels, edge_labels
		

if __name__ == "__main__":
	test_get_nb_edit_operations_symbolic_cml()