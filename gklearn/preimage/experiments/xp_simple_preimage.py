#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:30:17 2020

@author: ljia

This script constructs simple preimages to test preimage methods and find bugs and shortcomings in them.
"""


def xp_simple_preimage():
	import numpy as np
	
	"""**1.   Get dataset.**"""

	from gklearn.utils import Dataset, split_dataset_by_target
	
	# Predefined dataset name, use dataset "MAO".
	ds_name = 'MAO'
	# The node/edge labels that will not be used in the computation.
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']}
	
	# Initialize a Dataset.
	dataset_all = Dataset()
	# Load predefined dataset "MAO".
	dataset_all.load_predefined_dataset(ds_name)
	# Remove irrelevant labels.
	dataset_all.remove_labels(**irrelevant_labels)
	# Split the whole dataset according to the classification targets.
	datasets = split_dataset_by_target(dataset_all)
	# Get the first class of graphs, whose median preimage will be computed.
	dataset = datasets[0]
	len(dataset.graphs)
	
	"""**2.  Set parameters.**"""
	
	import multiprocessing
	
	# Parameters for MedianPreimageGenerator (our method).
	mpg_options = {'fit_method': 'k-graphs', # how to fit edit costs. "k-graphs" means use all graphs in median set when fitting.
				   'init_ecc': [4, 4, 2, 1, 1, 1], # initial edit costs.
				   'ds_name': ds_name, # name of the dataset.
				   'parallel': True, # whether the parallel scheme is to be used.
				   'time_limit_in_sec': 0, # maximum time limit to compute the preimage. If set to 0 then no limit.
				   'max_itrs': 10, # maximum iteration limit to optimize edit costs. If set to 0 then no limit.
				   'max_itrs_without_update': 3, # If the times that edit costs is not update is more than this number, then the optimization stops.
				   'epsilon_residual': 0.01, # In optimization, the residual is only considered changed if the change is bigger than this number.
				   'epsilon_ec': 0.1, # In optimization, the edit costs are only considered changed if the changes are bigger than this number.
				   'verbose': 2 # whether to print out results.
	               }
	# Parameters for graph kernel computation.
	kernel_options = {'name': 'PathUpToH', # use path kernel up to length h.
					  'depth': 9,
					  'k_func': 'MinMax',
					  'compute_method': 'trie',
					  'parallel': 'imap_unordered', # or None
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True, # whether to use normalized Gram matrix to optimize edit costs.
					  'verbose': 2 # whether to print out results.
	                  }
	# Parameters for GED computation.
	ged_options = {'method': 'IPFP', # use IPFP huristic.
				   'initialization_method': 'RANDOM', # or 'NODE', etc.
				   'initial_solutions': 10, # when bigger than 1, then the method is considered mIPFP.
				   'edit_cost': 'CONSTANT', # use CONSTANT cost.
				   'attr_distance': 'euclidean', # the distance between non-symbolic node/edge labels is computed by euclidean distance.
				   'ratio_runs_from_initial_solutions': 1,
				   'threads': multiprocessing.cpu_count(), # parallel threads. Do not work if mpg_options['parallel'] = False.
				   'init_option': 'EAGER_WITHOUT_SHUFFLED_COPIES'
	               }
	# Parameters for MedianGraphEstimator (Boria's method).
	mge_options = {'init_type': 'MEDOID', # how to initial median (compute set-median). "MEDOID" is to use the graph with smallest SOD.
				   'random_inits': 10, # number of random initialization when 'init_type' = 'RANDOM'.
				   'time_limit': 600, # maximum time limit to compute the generalized median. If set to 0 then no limit.
				   'verbose': 2, # whether to print out results.
				   'refine': False # whether to refine the final SODs or not.
	               }
	print('done.')
	
	"""**3.   Compute the Gram matrix and distance matrix.**"""
	
	from gklearn.utils.utils import get_graph_kernel_by_name
	
	# Get a graph kernel instance.
	graph_kernel = get_graph_kernel_by_name(kernel_options['name'], 
					  node_labels=dataset.node_labels, edge_labels=dataset.edge_labels, 
					  node_attrs=dataset.node_attrs, edge_attrs=dataset.edge_attrs,
					  ds_infos=dataset.get_dataset_infos(keys=['directed']),
					  kernel_options=kernel_options)	
	# Compute Gram matrix.
	gram_matrix, run_time = graph_kernel.compute(dataset.graphs, **kernel_options)
	
	# Compute distance matrix.
	from gklearn.utils import compute_distance_matrix
	dis_mat, _, _, _ = compute_distance_matrix(gram_matrix)
	
	print('done.')

	"""**4.   Find the candidate graph.**"""
	
	from gklearn.preimage.utils import compute_k_dis
	
	# Number of the nearest neighbors.
	k_neighbors = 10
	
	# For each graph G in dataset, compute the distance between its image \Phi(G) and the mean of its neighbors' images.
	dis_min = np.inf # the minimum distance between possible \Phi(G) and the mean of its neighbors.
	for idx, G in enumerate(dataset.graphs):
		# Find the k nearest neighbors of G.
		dis_list = dis_mat[idx] # distance between \Phi(G) and image of each graphs.
		idx_sort = np.argsort(dis_list) # sort distances and get the sorted indices.
		idx_nearest = idx_sort[1:k_neighbors+1] # indices of the k-nearest neighbors.
		dis_k_nearest = [dis_list[i] for i in idx_nearest] # k-nearest distances, except the 0.
		G_k_nearest = [dataset.graphs[i] for i in idx_nearest] # k-nearest neighbors.
		
		# Compute the distance between \Phi(G) and the mean of its neighbors.
		dis_tmp = compute_k_dis(idx, # the index of G in Gram matrix.
						        idx_nearest, # the indices of the neighbors
								[1 / k_neighbors] * k_neighbors, # coefficients for neighbors. 
								gram_matrix,
								withterm3=False)
		# Check if the new distance is smallers.
		if dis_tmp < dis_min:
			dis_min = dis_tmp
			G_cand = G
			G_neighbors = G_k_nearest
			
	print('The minimum distance is', dis_min)
		
	"""**5.   Run median preimage generator.**"""
	
	from gklearn.preimage import MedianPreimageGenerator
	
	# Set the dataset as the k-nearest neighbors.
	dataset.load_graphs(G_neighbors)
	
	# Create median preimage generator instance.
	mpg = MedianPreimageGenerator()
	# Add dataset.
	mpg.dataset = dataset
	# Set parameters.
	mpg.set_options(**mpg_options.copy())
	mpg.kernel_options = kernel_options.copy()
	mpg.ged_options = ged_options.copy()
	mpg.mge_options = mge_options.copy()
	# Run.
	mpg.run()
	
	"""**4. Get results.**"""
	
	# Get results.
	import pprint
	pp = pprint.PrettyPrinter(indent=4) # pretty print
	results = mpg.get_results()
	pp.pprint(results)
	 
	draw_graph(mpg.set_median)
	draw_graph(mpg.gen_median)
	draw_graph(G_cand)


# Draw generated graphs.
def draw_graph(graph):
	import matplotlib.pyplot as plt
	import networkx as nx
	plt.figure()
	pos = nx.spring_layout(graph)
	nx.draw(graph, pos, node_size=500, labels=nx.get_node_attributes(graph, 'atom_symbol'), font_color='w', width=3, with_labels=True)
	plt.show()
	plt.clf()
	plt.close()


if __name__ == '__main__':
	xp_simple_preimage()