#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:41:54 2020

@author: ljia

This script compares the results with and without FCSP.
"""
from gklearn.dataset import Dataset
from shortest_path import SPSpace
from structural_sp import SSPSpace
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from gklearn.experiments import DATASET_ROOT
import functools
import os
import pickle
import sys
import logging


def run_task(kernel_name, ds_name, fcsp):
	save_file_suffix = '.' + kernel_name + '.' + ds_name + '.' + str(fcsp)
	file_name = os.path.join(save_dir, 'space' + save_file_suffix + '.pkl')

	# Return if the task is already completed.
	if os.path.isfile(file_name):
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
			if data['completed']:
				return

	print()
	print((kernel_name, ds_name, str(fcsp)))

	try:
		gram_matrix, run_time = compute(kernel_name, ds_name, fcsp, file_name)

	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = os.path.join(save_dir, 'error.space' + save_file_suffix + '.txt')
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('\n--------------' + save_file_suffix + '------------------')
		print(repr(exp))

# 	else:
# 		with open(file_name, 'wb') as f:
# 			pickle.dump(run_time, f)


def compute(kernel_name, ds_name, fcsp, file_name):
	dataset = Dataset(ds_name, root=DATASET_ROOT, verbose=True)
	if kernel_name == 'ShortestPath':
		dataset.trim_dataset(edge_required=True)
# 		dataset.cut_graphs(range(0, 10))
		kernel_class = SPSpace
	else:
# 		dataset.cut_graphs(range(0, 10))
		kernel_class = SSPSpace

	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	node_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	edge_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}

	graph_kernel = kernel_class(name=kernel_name,
							  node_labels=dataset.node_labels,
							  edge_labels=dataset.edge_labels,
							  node_attrs=dataset.node_attrs,
							  edge_attrs=dataset.edge_attrs,
							  ds_infos=dataset.get_dataset_infos(keys=['directed']),
							  fcsp=fcsp,
							  compute_method='naive',
							  node_kernels=node_kernels,
							  edge_kernels=edge_kernels,
							  file_name=file_name
							  )
	gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
											  parallel=None,
											  normalize=False,
											  verbose=2
											  )
	return gram_matrix, run_time


if __name__ == '__main__':
	if len(sys.argv) > 1:
		kernel_name = sys.argv[1]
		ds_name = sys.argv[2]
		fcsp = True if sys.argv[3] == 'True' else False
	else:
		kernel_name = 'StructuralSP'
		ds_name = 'Fingerprint'
		fcsp = True

	save_dir = 'outputs/'
	os.makedirs(save_dir, exist_ok=True)

	run_task(kernel_name, ds_name, fcsp)