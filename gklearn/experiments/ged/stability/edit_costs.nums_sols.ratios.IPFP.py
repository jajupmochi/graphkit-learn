#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  20 11:48:02 2020

@author: ljia
"""	
# This script tests the influence of the ratios between node costs and edge costs on the stability of the GED computation, where the base edit costs are [1, 1, 1, 1, 1, 1].

import os
import multiprocessing
import pickle
import logging
from gklearn.utils import Dataset
from gklearn.ged.util import compute_geds


def get_dataset(ds_name):
	# The node/edge labels that will not be used in the computation.
	if ds_name == 'MAO':
		irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']}
	elif ds_name == 'Monoterpenoides':
		irrelevant_labels = {'edge_labels': ['valence']}
	elif ds_name == 'MUTAG':
		irrelevant_labels = {'edge_labels': ['label_0']}
	elif ds_name == 'AIDS_symb':
		irrelevant_labels = {'node_attrs': ['chem', 'charge', 'x', 'y'], 'edge_labels': ['valence']}

	# Initialize a Dataset.
	dataset = Dataset()
	# Load predefined dataset.
	dataset.load_predefined_dataset(ds_name)
	# Remove irrelevant labels.
	dataset.remove_labels(**irrelevant_labels)
	print('dataset size:', len(dataset.graphs))
	return dataset


def xp_compute_ged_matrix(ds_name, num_solutions, ratio, trial):

	save_dir = 'outputs/edit_costs.num_sols.ratios.IPFP/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
	save_file_suffix = '.' + ds_name + '.num_sols_' + str(num_solutions) + '.ratio_' + "{:.2f}".format(ratio) + '.trial_' + str(trial)
	
	"""**1.   Get dataset.**"""
	dataset = get_dataset(ds_name)

	"""**2.  Set parameters.**"""

	# Parameters for GED computation.
	ged_options = {'method': 'IPFP',  # use IPFP huristic.
				   'initialization_method': 'RANDOM',  # or 'NODE', etc.
				   # when bigger than 1, then the method is considered mIPFP.
				   'initial_solutions': int(num_solutions * 4),
				   'edit_cost': 'CONSTANT',  # use CONSTANT cost.
				   # the distance between non-symbolic node/edge labels is computed by euclidean distance.
				   'attr_distance': 'euclidean',
				   'ratio_runs_from_initial_solutions': 0.25,
				   # parallel threads. Do not work if mpg_options['parallel'] = False.
				   'threads': multiprocessing.cpu_count(),
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'
				   }
	
	edit_cost_constants = [i * ratio for i in [1, 1, 1]] + [1, 1, 1]
# 	edit_cost_constants = [item * 0.01 for item in edit_cost_constants]
# 	pickle.dump(edit_cost_constants, open(save_dir + "edit_costs" + save_file_suffix + ".pkl", "wb"))

	options = ged_options.copy()
	options['edit_cost_constants'] = edit_cost_constants
	options['node_labels'] = dataset.node_labels
	options['edge_labels'] = dataset.edge_labels
	options['node_attrs'] = dataset.node_attrs
	options['edge_attrs'] = dataset.edge_attrs
	parallel = True # if num_solutions == 1 else False
	
	"""**5.   Compute GED matrix.**"""
	ged_mat = 'error'
	try:
		ged_vec_init, ged_mat, n_edit_operations = compute_geds(dataset.graphs, options=options, parallel=parallel, verbose=True)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = save_dir + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('save_file_suffix')
		print(repr(exp))
					
	"""**6. Get results.**"""
	
	pickle.dump(ged_mat, open(save_dir + 'ged_matrix' + save_file_suffix + '.pkl', 'wb'))
		

if __name__ == '__main__':
	for ds_name in ['MAO', 'Monoterpenoides', 'MUTAG', 'AIDS_symb']:
		print()
		print('Dataset:', ds_name)
		for num_solutions in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			print()
			print('# of solutions:', num_solutions)
			for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
				print()
				print('Ratio:', ratio)
				for trial in range(1, 101):
					print()
					print('Trial:', trial)
					xp_compute_ged_matrix(ds_name, num_solutions, ratio, trial)