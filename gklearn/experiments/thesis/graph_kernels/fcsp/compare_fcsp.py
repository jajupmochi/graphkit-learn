#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:41:54 2020

@author: ljia

This script compares the results with and without FCSP.
"""
from gklearn.dataset import Dataset
from gklearn.utils import get_graph_kernel_by_name
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from gklearn.experiments import DATASET_ROOT
import functools
import os
import pickle
import sys
import logging


# def run_all(fcsp):

# 	from sklearn.model_selection import ParameterGrid

# 	Dataset_List = ['Alkane_unlabeled', 'Alkane', 'Acyclic', 'MAO_lite', 'MAO',
# 				    'PAH_unlabeled', 'PAH', 'MUTAG', 'Monoterpens',
# 					'Letter-high', 'Letter-med', 'Letter-low',
# 					'ENZYMES', 'AIDS', 'NCI1', 'NCI109', 'DD',
# 					'BZR', 'COX2', 'DHFR', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR',
# 					'Cuneiform', 'KKI', 'OHSU', 'Peking_1', 'SYNTHETICnew',
# 					'Synthie', 'SYNTHETIC', 'Fingerprint', 'IMDB-BINARY',
# 					'IMDB-MULTI', 'COIL-DEL', 'PROTEINS', 'PROTEINS_full',
# 					'Mutagenicity', 'REDDIT-BINARY']

# 	Kernel_List = ['ShortestPath', 'StructuralSP']

# 	task_grid = ParameterGrid({'kernel': Kernel_List[:], 'dataset': Dataset_List[:]})

# 	for task in list(task_grid):

# 		save_file_suffix = '.' + task['kernel'] + '.' + task['dataset']
# 		file_name = os.path.join(save_dir, 'run_time' + save_file_suffix + '.pkl')
# 		if not os.path.isfile(file_name):
# 			print()
# 			print((task['kernel'], task['dataset']))

# 			try:
# 				gram_matrix, run_time = compute(task['kernel'], task['dataset'], fcsp)

# 			except Exception as exp:
# 				print('An exception occured when running this experiment:')
# 				LOG_FILENAME = save_dir + 'error.txt'
# 				logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
# 				logging.exception('\n--------------' + save_file_suffix + '------------------')
# 				print(repr(exp))
# 			else:
# 				save_file_suffix = '.' + task['kernel'] + task['dataset']

# 				with open(file_name, 'wb') as f:
# 					pickle.dump(run_time, f)



def run_task(kernel_name, ds_name, fcsp):
	save_file_suffix = '.' + kernel_name + '.' + ds_name + '.' + str(fcsp)
	file_name = os.path.join(save_dir, 'run_time' + save_file_suffix + '.pkl')

	if not os.path.isfile(file_name):
		print()
		print((kernel_name, ds_name, str(fcsp)))

		try:
			gram_matrix, run_time = compute(kernel_name, ds_name, fcsp)

		except Exception as exp:
			print('An exception occured when running this experiment:')
			LOG_FILENAME = os.path.join(save_dir, 'error' + save_file_suffix + '.txt')
			logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
			logging.exception('\n--------------' + save_file_suffix + '------------------')
			print(repr(exp))

		else:
			with open(file_name, 'wb') as f:
				pickle.dump(run_time, f)


def compute(kernel_name, ds_name, fcsp):
	dataset = Dataset(ds_name, root=DATASET_ROOT, verbose=True)
	if kernel_name == 'ShortestPath':
		dataset.trim_dataset(edge_required=True)


	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	node_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	edge_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}

	graph_kernel = get_graph_kernel_by_name(name=kernel_name,
							  node_labels=dataset.node_labels,
							  edge_labels=dataset.edge_labels,
							  node_attrs=dataset.node_attrs,
							  edge_attrs=dataset.edge_attrs,
							  ds_infos=dataset.get_dataset_infos(keys=['directed']),
							  fcsp=fcsp,
							  compute_method='naive',
							  node_kernels=node_kernels,
							  edge_kernels=edge_kernels,
							  )
	gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
											  parallel=None,
											  normalize=True,
											  verbose=2
											  )
	return gram_matrix, run_time


if __name__ == '__main__':
	if len(sys.argv) > 1:
		kernel_name = sys.argv[1]
		ds_name = sys.argv[2]
		fcsp = True if sys.argv[3] == 'True' else False
	else:
		kernel_name = 'ShortestPath'
		ds_name = 'Acyclic'
		fcsp = True

	save_dir = 'outputs/'
	os.makedirs(save_dir, exist_ok=True)

	run_task(kernel_name, ds_name, fcsp)