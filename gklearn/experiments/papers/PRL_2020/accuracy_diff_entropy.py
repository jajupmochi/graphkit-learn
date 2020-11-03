#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:08:33 2020

@author: ljia

This script compute classification accuracy of each geaph kernel on datasets 
with different entropy of degree distribution.
"""
from utils import Graph_Kernel_List, cross_validate
import numpy as np
import logging

num_nodes = 40
half_num_graphs = 100


def generate_graphs():
# 	from gklearn.utils.graph_synthesizer import GraphSynthesizer
# 	gsyzer = GraphSynthesizer()
# 	graphs = gsyzer.unified_graphs(num_graphs=1000, num_nodes=20, num_edges=40, num_node_labels=0, num_edge_labels=0, seed=None, directed=False)
# 	return graphs
	import networkx as nx
	
	degrees11 = [5] * num_nodes
# 	degrees12 = [2] * num_nodes
	degrees12 = [5] * num_nodes
	degrees21 = list(range(1, 11)) * 6
# 	degrees22 = [5 * i for i in list(range(1, 11)) * 6]
	degrees22 = list(range(1, 11)) * 6
	
	# method 1
	graphs11 = [nx.configuration_model(degrees11, create_using=nx.Graph) for i in range(half_num_graphs)]
	graphs12 = [nx.configuration_model(degrees12, create_using=nx.Graph) for i in range(half_num_graphs)]
	
	for g in graphs11:
		g.remove_edges_from(nx.selfloop_edges(g))
	for g in graphs12:
		g.remove_edges_from(nx.selfloop_edges(g))
	
	# method 2: can easily generate isomorphic graphs.
# 	graphs11 = [nx.random_regular_graph(2, num_nodes, seed=None) for i in range(half_num_graphs)]
# 	graphs12 = [nx.random_regular_graph(10, num_nodes, seed=None) for i in range(half_num_graphs)]
	
	# Add node labels.
	for g in graphs11:
		for n in g.nodes():
			g.nodes[n]['atom'] = 0
	for g in graphs12:
		for n in g.nodes():
			g.nodes[n]['atom'] = 1
		
	graphs1 = graphs11 + graphs12

	# method 1: the entorpy of the two classes is not the same.
	graphs21 = [nx.configuration_model(degrees21, create_using=nx.Graph) for i in range(half_num_graphs)]
	graphs22 = [nx.configuration_model(degrees22, create_using=nx.Graph) for i in range(half_num_graphs)]	

	for g in graphs21:
		g.remove_edges_from(nx.selfloop_edges(g))
	for g in graphs22:
		g.remove_edges_from(nx.selfloop_edges(g))
	
# 	# method 2: tooo slow, and may fail.
# 	graphs21 = [nx.random_degree_sequence_graph(degrees21, seed=None, tries=100) for i in range(half_num_graphs)]
# 	graphs22 = [nx.random_degree_sequence_graph(degrees22, seed=None, tries=100) for i in range(half_num_graphs)]	

# 	# method 3: no randomness.
# 	graphs21 = [nx.havel_hakimi_graph(degrees21, create_using=None) for i in range(half_num_graphs)]
# 	graphs22 = [nx.havel_hakimi_graph(degrees22, create_using=None) for i in range(half_num_graphs)]

# 	# method 4:
# 	graphs21 = [nx.configuration_model(degrees21, create_using=nx.Graph) for i in range(half_num_graphs)]
# 	graphs22 = [nx.degree_sequence_tree(degrees21, create_using=nx.Graph) for i in range(half_num_graphs)]	
	
# 	# method 5: the entorpy of the two classes is not the same.
# 	graphs21 = [nx.expected_degree_graph(degrees21, seed=None, selfloops=False) for i in range(half_num_graphs)]
# 	graphs22 = [nx.expected_degree_graph(degrees22, seed=None, selfloops=False) for i in range(half_num_graphs)]	
	
# 	# method 6: seems there is no randomness0
# 	graphs21 = [nx.random_powerlaw_tree(num_nodes, gamma=3, seed=None, tries=10000) for i in range(half_num_graphs)]
# 	graphs22 = [nx.random_powerlaw_tree(num_nodes, gamma=3, seed=None, tries=10000) for i in range(half_num_graphs)]	

	# Add node labels.
	for g in graphs21:
		for n in g.nodes():
			g.nodes[n]['atom'] = 0
	for g in graphs22:
		for n in g.nodes():
			g.nodes[n]['atom'] = 1

	graphs2 = graphs21 + graphs22
	
# 	# check for isomorphism.
# 	iso_mat1 = np.zeros((len(graphs1), len(graphs1)))
# 	num1 = 0
# 	num2 = 0
# 	for i in range(len(graphs1)):
# 		for j in range(i + 1, len(graphs1)):
# 			 if nx.is_isomorphic(graphs1[i], graphs1[j]):
# 				 iso_mat1[i, j] = 1
# 				 iso_mat1[j, i] = 1
# 				 num1 += 1
# 				 print('iso:', num1, ':', i, ',', j)
# 			 else:
# 				 num2 += 1
# 				 print('not iso:', num2, ':', i, ',', j)
# 				 
# 	iso_mat2 = np.zeros((len(graphs2), len(graphs2)))
# 	num1 = 0
# 	num2 = 0
# 	for i in range(len(graphs2)):
# 		for j in range(i + 1, len(graphs2)):
# 			 if nx.is_isomorphic(graphs2[i], graphs2[j]):
# 				 iso_mat2[i, j] = 1
# 				 iso_mat2[j, i] = 1
# 				 num1 += 1
# 				 print('iso:', num1, ':', i, ',', j)
# 			 else:
# 				 num2 += 1
# 				 print('not iso:', num2, ':', i, ',', j)
		
	return graphs1, graphs2


def get_infos(graph):
	from gklearn.utils import Dataset
	ds = Dataset()
	ds.load_graphs(graph)
	infos = ds.get_dataset_infos(keys=['all_degree_entropy', 'ave_node_degree'])
	infos['ave_degree_entropy'] = np.mean(infos['all_degree_entropy'])
	print(infos['ave_degree_entropy'], ',', infos['ave_node_degree'])
	return infos


def xp_accuracy_diff_entropy():
	
	# Generate graphs.
	graphs1, graphs2 = generate_graphs()

	
	# Compute entropy of degree distribution of the generated graphs.
	info11 = get_infos(graphs1[0:half_num_graphs])
	info12 = get_infos(graphs1[half_num_graphs:])
	info21 = get_infos(graphs2[0:half_num_graphs])
	info22 = get_infos(graphs2[half_num_graphs:])

	# Run and save.
	import pickle
	import os
	save_dir = 'outputs/accuracy_diff_entropy/'
	os.makedirs(save_dir, exist_ok=True)

	accuracies = {}
	confidences = {}
	
	for kernel_name in Graph_Kernel_List:
		print()
		print('Kernel:', kernel_name)
		
		accuracies[kernel_name] = []
		confidences[kernel_name] = []
		for set_i, graphs in enumerate([graphs1, graphs2]):
			print()
			print('Graph set', set_i)
			
			tmp_graphs = [g.copy() for g in graphs]
			targets = [0] * half_num_graphs + [1] * half_num_graphs
			
			accuracy = 'error'
			confidence = 'error'
			try:
				accuracy, confidence = cross_validate(tmp_graphs, targets, kernel_name, ds_name=str(set_i), output_dir=save_dir) #, n_jobs=1)
			except Exception as exp:
				print('An exception occured when running this experiment:')
				LOG_FILENAME = save_dir + 'error.txt'
				logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
				logging.exception('\n' + kernel_name + ', ' + str(set_i) + ':')
				print(repr(exp))
			accuracies[kernel_name].append(accuracy)
			confidences[kernel_name].append(confidence)
			
			pickle.dump(accuracy, open(save_dir + 'accuracy.' + kernel_name + '.' + str(set_i) + '.pkl', 'wb'))
			pickle.dump(confidence, open(save_dir + 'confidence.' + kernel_name + '.' + str(set_i) + '.pkl', 'wb'))
		
	# Save all.	
	pickle.dump(accuracies, open(save_dir + 'accuracies.pkl', 'wb'))	
	pickle.dump(confidences, open(save_dir + 'confidences.pkl', 'wb'))	
	
	return


if __name__ == '__main__':
	xp_accuracy_diff_entropy()