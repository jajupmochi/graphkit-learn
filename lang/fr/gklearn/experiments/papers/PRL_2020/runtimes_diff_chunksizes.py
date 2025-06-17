#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:34:26 2020

@author: ljia
"""
from utils import Graph_Kernel_List, Dataset_List, compute_graph_kernel
from gklearn.utils.graphdataset import load_predefined_dataset
import logging


def xp_runtimes_diff_chunksizes():
		
	# Run and save.
	import pickle
	import os
	save_dir = 'outputs/runtimes_diff_chunksizes/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	run_times = {}

	for ds_name in Dataset_List:
		print()
		print('Dataset:', ds_name)
			
		run_times[ds_name] = []
		for kernel_name in Graph_Kernel_List:
			print()
			print('Kernel:', kernel_name)
			
			run_times[ds_name].append([])
			for chunksize in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
				print()
				print('Chunksize:', chunksize)
			
				# get graphs.
				graphs, _ = load_predefined_dataset(ds_name)

				# Compute Gram matrix.
				run_time = 'error'
				try:
					gram_matrix, run_time = compute_graph_kernel(graphs, kernel_name, chunksize=chunksize)
				except Exception as exp:
					print('An exception occured when running this experiment:')
					LOG_FILENAME = save_dir + 'error.txt'
					logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
					logging.exception('')
					print(repr(exp))			
				run_times[ds_name][-1].append(run_time)	
				
				pickle.dump(run_time, open(save_dir + 'run_time.' + kernel_name + '.' + ds_name + '.' + str(chunksize) + '.pkl', 'wb'))
		
	# Save all.	
	pickle.dump(run_times, open(save_dir + 'run_times.pkl', 'wb'))	
	
	return


if __name__ == '__main__':
	xp_runtimes_diff_chunksizes()
