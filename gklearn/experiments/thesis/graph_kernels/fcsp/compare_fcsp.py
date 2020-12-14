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
import functools
import os
import pickle
import sys


def run_all(fcsp):
	save_dir = 'outputs/' + ('fscp' if fcsp == True else 'naive') + '/'
	os.makedirs(save_dir, exist_ok=True)

	from sklearn.model_selection import ParameterGrid

	Dataset_List = ['Alkane_unlabeled', 'Alkane', 'Acyclic', 'MAO_lite', 'MAO',
				    'PAH', 'MUTAG', 'Monoterpenoids', 'Letter-high',
					'Letter-med', 'Letter-low',
					'ENZYMES', 'AIDS_lite', 'AIDS', 'NCI1', 'NCI109', 'DD']

	Kernel_List = ['ShortestPath', 'StructuralSP']

	work_grid = ParameterGrid({'kernel': Kernel_List[0:], 'dataset': Dataset_List[2:3]})

	for work in list(work_grid):

		save_file_suffix = '.' + work['kernel'] + '.' + work['dataset']
		file_name = os.path.join(save_dir, 'run_time' + save_file_suffix + '.pkl')
		if not os.path.isfile(file_name):
			print()
			print((work['kernel'], work['dataset']))

			gram_matrix, run_time = run_work(work['kernel'], work['dataset'], fcsp)

			save_file_suffix = '.' + work['kernel'] + work['dataset']

			with open(file_name, 'wb') as f:
				pickle.dump(run_time, f)


def run_work(kernel_name, ds_name, fcsp):
	dataset = Dataset(ds_name, verbose=True)

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
		fcsp = True if sys.argv[1] == 'True' else False
	else:
		fcsp = True
	run_all(fcsp)

