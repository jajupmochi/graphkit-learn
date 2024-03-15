#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:31:46 2020

@author: ljia
"""

def xp_check_results_of_GEDEnv():
	"""Compare results of GEDEnv to GEDLIB.
	"""
	"""**1.   Get dataset.**"""

	from gklearn.utils import Dataset
	
	# Predefined dataset name, use dataset "MUTAG".
	ds_name = 'MUTAG'
	
	# Initialize a Dataset.
	dataset = Dataset()
	# Load predefined dataset "MUTAG".
	dataset.load_predefined_dataset(ds_name)

	results1 = compute_geds_by_GEDEnv(dataset)
	results2 = compute_geds_by_GEDLIB(dataset)
	
	# Show results.
	import pprint
	pp = pprint.PrettyPrinter(indent=4) # pretty print
	print('Restuls using GEDEnv:')
	pp.pprint(results1)
	print()
	print('Restuls using GEDLIB:')
	pp.pprint(results2)
	
	return results1, results2
	
	
def compute_geds_by_GEDEnv(dataset):
	from gklearn.ged.env import GEDEnv		
	import numpy as np
	
	graph1 = dataset.graphs[0]
	graph2 = dataset.graphs[1]
		
	ged_env = GEDEnv() # initailize GED environment.
	ged_env.set_edit_cost('CONSTANT', # GED cost type.
	                      edit_cost_constants=[3, 3, 1, 3, 3, 1] # edit costs.
						  )
	for g in dataset.graphs[0:10]:
		ged_env.add_nx_graph(g, '')
# 	ged_env.add_nx_graph(graph1, '') # add graph1
# 	ged_env.add_nx_graph(graph2, '') # add graph2
	listID = ged_env.get_all_graph_ids() # get list IDs of graphs
	ged_env.init(init_type='LAZY_WITHOUT_SHUFFLED_COPIES') # initialize GED environment.
	options = {'threads': 1 # parallel threads.
			   }
	ged_env.set_method('BIPARTITE', # GED method.
	                   options # options for GED method.
					   )
	ged_env.init_method() # initialize GED method.
	
	ged_mat = np.empty((10, 10))
	for i in range(0, 10):
		for j in range(i, 10):
			ged_env.run_method(i, j) # run.
			ged_mat[i, j] = ged_env.get_upper_bound(i, j)
			ged_mat[j, i] = ged_mat[i, j]
	
	results = {}
	results['pi_forward'] = ged_env.get_forward_map(listID[0], listID[1]) # forward map.
	results['pi_backward'] = ged_env.get_backward_map(listID[0], listID[1]) # backward map.
	results['upper_bound'] = ged_env.get_upper_bound(listID[0], listID[1])	# GED bewteen two graphs.
	results['runtime'] = ged_env.get_runtime(listID[0], listID[1])
	results['init_time'] = ged_env.get_init_time()
	results['ged_mat'] = ged_mat
	
	return results


def compute_geds_by_GEDLIB(dataset):
	from gklearn.gedlib import librariesImport, gedlibpy	
	from gklearn.ged.util import ged_options_to_string
	import numpy as np
	
	graph1 = dataset.graphs[5]
	graph2 = dataset.graphs[6]
		
	ged_env = gedlibpy.GEDEnv() # initailize GED environment.
	ged_env.set_edit_cost('CONSTANT', # GED cost type.
	                      edit_cost_constant=[3, 3, 1, 3, 3, 1] # edit costs.
						  )  
# 	ged_env.add_nx_graph(graph1, '') # add graph1
# 	ged_env.add_nx_graph(graph2, '') # add graph2
	for g in dataset.graphs[0:10]:
		ged_env.add_nx_graph(g, '')
	listID = ged_env.get_all_graph_ids() # get list IDs of graphs
	ged_env.init(init_option='LAZY_WITHOUT_SHUFFLED_COPIES') # initialize GED environment.
	options = {'initialization-method': 'RANDOM', # or 'NODE', etc.
	           'threads': 1 # parallel threads.
			   }
	ged_env.set_method('BIPARTITE', # GED method.
	                   ged_options_to_string(options) # options for GED method.
					   )
	ged_env.init_method() # initialize GED method.
	
	ged_mat = np.empty((10, 10))
	for i in range(0, 10):
		for j in range(i, 10):
			ged_env.run_method(i, j) # run.
			ged_mat[i, j] = ged_env.get_upper_bound(i, j)
			ged_mat[j, i] = ged_mat[i, j]
	
	results = {}
	results['pi_forward'] = ged_env.get_forward_map(listID[0], listID[1]) # forward map.
	results['pi_backward'] = ged_env.get_backward_map(listID[0], listID[1]) # backward map.
	results['upper_bound'] = ged_env.get_upper_bound(listID[0], listID[1])	# GED bewteen two graphs.
	results['runtime'] = ged_env.get_runtime(listID[0], listID[1])
	results['init_time'] = ged_env.get_init_time()
	results['ged_mat'] = ged_mat
	
	return results
		
	
if __name__ == '__main__':
	results1, results2 = xp_check_results_of_GEDEnv()