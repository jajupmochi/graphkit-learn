#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:34:26 2020

@author: ljia
"""
from utils import Graph_Kernel_List, compute_graph_kernel
import logging


def generate_graphs(num_nodes):
	from gklearn.utils.graph_synthesizer import GraphSynthesizer
	gsyzer = GraphSynthesizer()
	graphs = gsyzer.unified_graphs(num_graphs=100, num_nodes=num_nodes, num_edges=int(num_nodes*2), num_node_labels=0, num_edge_labels=0, seed=None, directed=False)
	return graphs


def xp_synthesized_graphs_num_nodes():
		
	# Run and save.
	import pickle
	import os
	save_dir = 'outputs/synthesized_graphs_num_nodes/'
	os.makedirs(save_dir, exist_ok=True)

	run_times = {}
	
	for kernel_name in Graph_Kernel_List:
		print()
		print('Kernel:', kernel_name)
		
		run_times[kernel_name] = []
		for num_nodes in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			print()
			print('Number of nodes:', num_nodes)
			
			# Generate graphs.
			graphs = generate_graphs(num_nodes)

			# Compute Gram matrix.
			run_time = 'error'
			try:
				gram_matrix, run_time = compute_graph_kernel(graphs, kernel_name)
			except Exception as exp:
				run_times[kernel_name].append('error')
				print('An exception occured when running this experiment:')
				LOG_FILENAME = save_dir + 'error.txt'
				logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
				logging.exception('')
				print(repr(exp))
			run_times[kernel_name].append(run_time)
			
			pickle.dump(run_time, open(save_dir + 'run_time.' + kernel_name + '.' + str(num_nodes) + '.pkl', 'wb'))
		
	# Save all.	
	pickle.dump(run_times, open(save_dir + 'run_times.pkl', 'wb'))	
	
	return


if __name__ == '__main__':
	xp_synthesized_graphs_num_nodes()
