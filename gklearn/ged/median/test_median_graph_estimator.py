#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:26:40 2020

@author: ljia
"""
	
def test_median_graph_estimator():
	from gklearn.utils import load_dataset
	from gklearn.ged.median import MedianGraphEstimator, constant_node_costs
	from gklearn.gedlib import librariesImport, gedlibpy
	from gklearn.preimage.utils import get_same_item_indices
	import multiprocessing

	# estimator parameters.
	init_type = 'MEDOID'
	num_inits = 1
	threads = multiprocessing.cpu_count()
	time_limit = 60000
	
	# algorithm parameters.
	algo = 'IPFP'
	initial_solutions = 1
	algo_options_suffix = ' --initial-solutions ' + str(initial_solutions) + ' --ratio-runs-from-initial-solutions 1 --initialization-method NODE '

	edit_cost_name = 'LETTER2'
	edit_cost_constants = [0.02987291, 0.0178211, 0.01431966, 0.001, 0.001]
	ds_name = 'Letter_high'
	
	# Load dataset.
	# dataset = '../../datasets/COIL-DEL/COIL-DEL_A.txt'
	dataset = '../../../datasets/Letter-high/Letter-high_A.txt'
	Gn, y_all, label_names = load_dataset(dataset)
	y_idx = get_same_item_indices(y_all)
	for i, (y, values) in enumerate(y_idx.items()):
		Gn_i = [Gn[val] for val in values]
		break
	
	# Set up the environment.
	ged_env = gedlibpy.GEDEnv()
	# gedlibpy.restart_env()
	ged_env.set_edit_cost(edit_cost_name, edit_cost_constant=edit_cost_constants)
	for G in Gn_i:
		ged_env.add_nx_graph(G, '')
	graph_ids = ged_env.get_all_graph_ids()
	set_median_id = ged_env.add_graph('set_median')
	gen_median_id = ged_env.add_graph('gen_median')
	ged_env.init(init_option='EAGER_WITHOUT_SHUFFLED_COPIES')
	
	# Set up the estimator.
	mge = MedianGraphEstimator(ged_env, constant_node_costs(edit_cost_name))
	mge.set_refine_method(algo, '--threads ' + str(threads) + ' --initial-solutions ' + str(initial_solutions) + ' --ratio-runs-from-initial-solutions 1')
	
	mge_options = '--time-limit ' + str(time_limit) + ' --stdout 2 --init-type ' + init_type
	mge_options += ' --random-inits ' + str(num_inits) + ' --seed ' + '1'  + ' --update-order TRUE --refine FALSE --randomness PSEUDO  --parallel TRUE '# @todo: std::to_string(rng())
	
	# Select the GED algorithm.
	algo_options = '--threads ' + str(threads) + algo_options_suffix
	mge.set_options(mge_options)
	mge.set_label_names(node_labels=label_names['node_labels'],
					  edge_labels=label_names['edge_labels'], 
					  node_attrs=label_names['node_attrs'], 
					  edge_attrs=label_names['edge_attrs'])
	mge.set_init_method(algo, algo_options)
	mge.set_descent_method(algo, algo_options)
	
	# Run the estimator.
	mge.run(graph_ids, set_median_id, gen_median_id)
	
	# Get SODs.
	sod_sm = mge.get_sum_of_distances('initialized')
	sod_gm = mge.get_sum_of_distances('converged')
	print('sod_sm, sod_gm: ', sod_sm, sod_gm)
	
	# Get median graphs.
	set_median = ged_env.get_nx_graph(set_median_id)
	gen_median = ged_env.get_nx_graph(gen_median_id)
	
	return set_median, gen_median


def test_median_graph_estimator_symb():
	from gklearn.utils import load_dataset
	from gklearn.ged.median import MedianGraphEstimator, constant_node_costs
	from gklearn.gedlib import librariesImport, gedlibpy
	from gklearn.preimage.utils import get_same_item_indices
	import multiprocessing

	# estimator parameters.
	init_type = 'MEDOID'
	num_inits = 1
	threads = multiprocessing.cpu_count()
	time_limit = 60000
	
	# algorithm parameters.
	algo = 'IPFP'
	initial_solutions = 1
	algo_options_suffix = ' --initial-solutions ' + str(initial_solutions) + ' --ratio-runs-from-initial-solutions 1 --initialization-method NODE '

	edit_cost_name = 'CONSTANT'
	edit_cost_constants = [4, 4, 2, 1, 1, 1]
	ds_name = 'MUTAG'
	
	# Load dataset.
	dataset = '../../../datasets/MUTAG/MUTAG_A.txt'
	Gn, y_all, label_names = load_dataset(dataset)
	y_idx = get_same_item_indices(y_all)
	for i, (y, values) in enumerate(y_idx.items()):
		Gn_i = [Gn[val] for val in values]
		break
	Gn_i = Gn_i[0:10]
	
	# Set up the environment.
	ged_env = gedlibpy.GEDEnv()
	# gedlibpy.restart_env()
	ged_env.set_edit_cost(edit_cost_name, edit_cost_constant=edit_cost_constants)
	for G in Gn_i:
		ged_env.add_nx_graph(G, '')
	graph_ids = ged_env.get_all_graph_ids()
	set_median_id = ged_env.add_graph('set_median')
	gen_median_id = ged_env.add_graph('gen_median')
	ged_env.init(init_option='EAGER_WITHOUT_SHUFFLED_COPIES')
	
	# Set up the estimator.
	mge = MedianGraphEstimator(ged_env, constant_node_costs(edit_cost_name))
	mge.set_refine_method(algo, '--threads ' + str(threads) + ' --initial-solutions ' + str(initial_solutions) + ' --ratio-runs-from-initial-solutions 1')
	
	mge_options = '--time-limit ' + str(time_limit) + ' --stdout 2 --init-type ' + init_type
	mge_options += ' --random-inits ' + str(num_inits) + ' --seed ' + '1'  + ' --update-order TRUE --refine FALSE --randomness PSEUDO --parallel TRUE '# @todo: std::to_string(rng())
	
	# Select the GED algorithm.
	algo_options = '--threads ' + str(threads) + algo_options_suffix
	mge.set_options(mge_options)
	mge.set_label_names(node_labels=label_names['node_labels'],
					  edge_labels=label_names['edge_labels'], 
					  node_attrs=label_names['node_attrs'], 
					  edge_attrs=label_names['edge_attrs'])
	mge.set_init_method(algo, algo_options)
	mge.set_descent_method(algo, algo_options)
	
	# Run the estimator.
	mge.run(graph_ids, set_median_id, gen_median_id)
	
	# Get SODs.
	sod_sm = mge.get_sum_of_distances('initialized')
	sod_gm = mge.get_sum_of_distances('converged')
	print('sod_sm, sod_gm: ', sod_sm, sod_gm)
	
	# Get median graphs.
	set_median = ged_env.get_nx_graph(set_median_id)
	gen_median = ged_env.get_nx_graph(gen_median_id)
	
	return set_median, gen_median


if _name_ == '_main_':
 	# set_median, gen_median = test_median_graph_estimator()
 	set_median, gen_median = test_median_graph_estimator_symb()