#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:34:26 2020

@author: ljia
"""
from utils import Graph_Kernel_List_VSym, compute_graph_kernel
import logging


def generate_graphs(num_nl_alp):
	from gklearn.utils.graph_synthesizer import GraphSynthesizer
	gsyzer = GraphSynthesizer()
	graphs = gsyzer.unified_graphs(num_graphs=100, num_nodes=20, num_edges=40, num_node_labels=num_nl_alp, num_edge_labels=0, seed=None, directed=False)
	return graphs


def xp_synthesized_graphs_num_node_label_alphabet():
		
	# Run and save.
	import pickle
	import os
	save_dir = 'outputs/synthesized_graphs_num_node_label_alphabet/'
	os.makedirs(save_dir, exist_ok=True)

	run_times = {}
	
	for kernel_name in Graph_Kernel_List_VSym:
		print()
		print('Kernel:', kernel_name)
		
		run_times[kernel_name] = []
		for num_nl_alp in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
			print()
			print('Number of node label alphabet:', num_nl_alp)
			
			# Generate graphs.
			graphs = generate_graphs(num_nl_alp)

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
			
			pickle.dump(run_time, open(save_dir + 'run_time.' + kernel_name + '.' + str(num_nl_alp) + '.pkl', 'wb'))
		
	# Save all.	
	pickle.dump(run_times, open(save_dir + 'run_times.pkl', 'wb'))	
	
	return


if __name__ == '__main__':
	xp_synthesized_graphs_num_node_label_alphabet()
