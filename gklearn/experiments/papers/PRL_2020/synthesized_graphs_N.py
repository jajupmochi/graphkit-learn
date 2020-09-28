#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:34:26 2020

@author: ljia
"""
from utils import Graph_Kernel_List, compute_graph_kernel
import logging


def generate_graphs():
	from gklearn.utils.graph_synthesizer import GraphSynthesizer
	gsyzer = GraphSynthesizer()
	graphs = gsyzer.unified_graphs(num_graphs=1000, num_nodes=20, num_edges=40, num_node_labels=0, num_edge_labels=0, seed=None, directed=False)
	return graphs


def xp_synthesied_graphs_dataset_size():
	
	# Generate graphs.
	graphs = generate_graphs()
	
	# Run and save.
	import pickle
	import os
	save_dir = 'outputs/synthesized_graphs_N/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	run_times = {}
	
	for kernel_name in Graph_Kernel_List:
		print()
		print('Kernel:', kernel_name)
		
		run_times[kernel_name] = []
		for num_graphs in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
			print()
			print('Number of graphs:', num_graphs)
			
			sub_graphs = [g.copy() for g in graphs[0:num_graphs]]
			
			try:
				gram_matrix, run_time = compute_graph_kernel(sub_graphs, kernel_name, n_jobs=1)
				run_times[kernel_name].append(run_time)
			except Exception as exp:
				run_times[kernel_name].append('error')
				print('An exception occured when running this experiment:')
				LOG_FILENAME = save_dir + 'error.txt'
				logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
				logging.exception('')
				print(repr(exp))
			
			pickle.dump(run_time, open(save_dir + 'run_time.' + kernel_name + '.' + str(num_graphs) + '.pkl', 'wb'))
		
	# Save all.	
	pickle.dump(run_times, open(save_dir + 'run_times.pkl', 'wb'))	
	
	return


if __name__ == '__main__':
	xp_synthesied_graphs_dataset_size()