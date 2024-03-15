"""
@author: linlin

@references:

	[1] Shervashidze N, Schweitzer P, Leeuwen EJ, Mehlhorn K, Borgwardt KM. 
	Weisfeiler-lehman graph kernels. Journal of Machine Learning Research. 
	2011;12(Sep):2539-61.
"""

import sys
from collections import Counter
from functools import partial
import time
#from multiprocessing import Pool
from tqdm import tqdm

import networkx as nx
import numpy as np

#from gklearn.kernels.pathKernel import pathkernel
from gklearn.utils.graphdataset import get_dataset_attributes
from gklearn.utils.parallel import parallel_gm

# @todo: support edge kernel, sp kernel, user-defined kernel.
def weisfeilerlehmankernel(*args, 
						   node_label='atom',
						   edge_label='bond_type',
						   height=0,
						   base_kernel='subtree',
						   parallel=None,
						   n_jobs=None, 
						   chunksize=None,
						   verbose=True):
	"""Compute Weisfeiler-Lehman kernels between graphs.
	
	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.
	
	G1, G2 : NetworkX graphs
		Two graphs between which the kernel is computed.		

	node_label : string
		Node attribute used as label. The default node label is atom.		

	edge_label : string
		Edge attribute used as label. The default edge label is bond_type.		

	height : int
		Subtree height.

	base_kernel : string
		Base kernel used in each iteration of WL kernel. Only default 'subtree' 
		kernel can be applied for now.

	parallel : None
		Which paralleliztion method is applied to compute the kernel. No 
		parallelization can be applied for now.

	n_jobs : int
		Number of jobs for parallelization. The default is to use all 
		computational cores. This argument is only valid when one of the 
		parallelization method is applied and can be ignored for now.

	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.

	Notes
	-----
	This function now supports WL subtree kernel only.
	"""
#		The default base 
#		kernel is subtree kernel. For user-defined kernel, base_kernel is the 
#		name of the base kernel function used in each iteration of WL kernel. 
#		This function returns a Numpy matrix, each element of which is the 
#		user-defined Weisfeiler-Lehman kernel between 2 praphs.
	# pre-process
	base_kernel = base_kernel.lower()
	Gn = args[0] if len(args) == 1 else [args[0], args[1]] # arrange all graphs in a list
	Gn = [g.copy() for g in Gn]
	ds_attrs = get_dataset_attributes(Gn, attr_names=['node_labeled'], 
									  node_label=node_label)
	if not ds_attrs['node_labeled']:
		for G in Gn:
			nx.set_node_attributes(G, '0', 'atom')

	start_time = time.time()

	# for WL subtree kernel
	if base_kernel == 'subtree':		   
		Kmatrix = _wl_kernel_do(Gn, node_label, edge_label, height, parallel, n_jobs, chunksize, verbose)

	# for WL shortest path kernel
	elif base_kernel == 'sp':
		Kmatrix = _wl_spkernel_do(Gn, node_label, edge_label, height)

	# for WL edge kernel
	elif base_kernel == 'edge':
		Kmatrix = _wl_edgekernel_do(Gn, node_label, edge_label, height)

	# for user defined base kernel
	else:
		Kmatrix = _wl_userkernel_do(Gn, node_label, edge_label, height, base_kernel)

	run_time = time.time() - start_time
	if verbose:
		print("\n --- Weisfeiler-Lehman %s kernel matrix of size %d built in %s seconds ---" 
			  % (base_kernel, len(args[0]), run_time))

	return Kmatrix, run_time


def _wl_kernel_do(Gn, node_label, edge_label, height, parallel, n_jobs, chunksize, verbose):
	"""Compute Weisfeiler-Lehman kernels between graphs.

	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.	   
	node_label : string
		node attribute used as label.
	edge_label : string
		edge attribute used as label.	  
	height : int
		wl height.

	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
	"""
	height = int(height)
	Kmatrix = np.zeros((len(Gn), len(Gn)))

	# initial for height = 0
	all_num_of_each_label = [] # number of occurence of each label in each graph in this iteration

	# for each graph
	for G in Gn:
		# get the set of original labels
		labels_ori = list(nx.get_node_attributes(G, node_label).values())
		# number of occurence of each label in G
		all_num_of_each_label.append(dict(Counter(labels_ori)))

	# Compute subtree kernel with the 0th iteration and add it to the final kernel
	compute_kernel_matrix(Kmatrix, all_num_of_each_label, Gn, parallel, n_jobs, chunksize, False)

	# iterate each height
	for h in range(1, height + 1):
		all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
		num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
#		all_labels_ori = set() # all unique orignal labels in all graphs in this iteration
		all_num_of_each_label = [] # number of occurence of each label in G

#		# for each graph
#		# ---- use pool.imap_unordered to parallel and track progress. ----
#		pool = Pool(n_jobs)
#		itr = zip(Gn, range(0, len(Gn)))
#		if len(Gn) < 100 * n_jobs:
#			chunksize = int(len(Gn) / n_jobs) + 1
#		else:
#			chunksize = 100
#		all_multisets_list = [[] for _ in range(len(Gn))]
##		set_unique_list = [[] for _ in range(len(Gn))]
#		get_partial = partial(wrapper_wl_iteration, node_label)
##		if verbose:
##			iterator = tqdm(pool.imap_unordered(get_partial, itr, chunksize),
##							desc='wl iteration', file=sys.stdout)
##		else:
#		iterator = pool.imap_unordered(get_partial, itr, chunksize)
#		for i, all_multisets in iterator:
#			all_multisets_list[i] = all_multisets
##			set_unique_list[i] = set_unique
##			all_set_unique = all_set_unique | set(set_unique)
#		pool.close()
#		pool.join()
		
#		all_set_unique = set()
#		for uset in all_multisets_list:
#			all_set_unique = all_set_unique | set(uset)
#			
#		all_set_unique = list(all_set_unique)
##		# a dictionary mapping original labels to new ones. 
##		set_compressed = {}
##		for idx, uset in enumerate(all_set_unique):
##			set_compressed.update({uset: idx})
#			
#		for ig, G in enumerate(Gn):
#
##			# a dictionary mapping original labels to new ones. 
##			set_compressed = {}
##			# if a label occured before, assign its former compressed label, 
##			# else assign the number of labels occured + 1 as the compressed label. 
##			for value in set_unique_list[i]:
##				if uset in all_set_unique:
##					set_compressed.update({uset: all_set_compressed[value]})
##				else:
##					set_compressed.update({value: str(num_of_labels_occured + 1)})
##					num_of_labels_occured += 1
#					
##			all_set_compressed.update(set_compressed)
#			
#			# relabel nodes
#			for idx, node in enumerate(G.nodes()):
#				G.nodes[node][node_label] = all_set_unique.index(all_multisets_list[ig][idx])
#				
#			# get the set of compressed labels
#			labels_comp = list(nx.get_node_attributes(G, node_label).values())
##			all_labels_ori.update(labels_comp)
#			all_num_of_each_label[ig] = dict(Counter(labels_comp))
			
			

		
#		all_set_unique = list(all_set_unique)
		
		
		# @todo: parallel this part.
		for idx, G in enumerate(Gn):

			all_multisets = []
			for node, attrs in G.nodes(data=True):
				# Multiset-label determination.
				multiset = [G.nodes[neighbors][node_label] for neighbors in G[node]]
				# sorting each multiset
				multiset.sort()
				multiset = [attrs[node_label]] + multiset # add the prefix 
				all_multisets.append(tuple(multiset))

			# label compression
			set_unique = list(set(all_multisets)) # set of unique multiset labels
			# a dictionary mapping original labels to new ones. 
			set_compressed = {}
			# if a label occured before, assign its former compressed label, 
			# else assign the number of labels occured + 1 as the compressed label. 
			for value in set_unique:
				if value in all_set_compressed.keys():
					set_compressed.update({value: all_set_compressed[value]})
				else:
					set_compressed.update({value: str(num_of_labels_occured + 1)})
					num_of_labels_occured += 1

			all_set_compressed.update(set_compressed)

			# relabel nodes
			for idx, node in enumerate(G.nodes()):
				G.nodes[node][node_label] = set_compressed[all_multisets[idx]]

			# get the set of compressed labels
			labels_comp = list(nx.get_node_attributes(G, node_label).values())
#			all_labels_ori.update(labels_comp)
			all_num_of_each_label.append(dict(Counter(labels_comp)))

		# Compute subtree kernel with h iterations and add it to the final kernel
		compute_kernel_matrix(Kmatrix, all_num_of_each_label, Gn, parallel, n_jobs, chunksize, False)

	return Kmatrix


def wl_iteration(G, node_label):
	all_multisets = []
	for node, attrs in G.nodes(data=True):
		# Multiset-label determination.
		multiset = [G.nodes[neighbors][node_label] for neighbors in G[node]]
		# sorting each multiset
		multiset.sort()
		multiset = [attrs[node_label]] + multiset # add the prefix 
		all_multisets.append(tuple(multiset))
#	# label compression
#	set_unique = list(set(all_multisets)) # set of unique multiset labels
	return all_multisets
	
#	# a dictionary mapping original labels to new ones. 
#	set_compressed = {}
#	# if a label occured before, assign its former compressed label, 
#	# else assign the number of labels occured + 1 as the compressed label. 
#	for value in set_unique:
#		if value in all_set_compressed.keys():
#			set_compressed.update({value: all_set_compressed[value]})
#		else:
#			set_compressed.update({value: str(num_of_labels_occured + 1)})
#			num_of_labels_occured += 1
#
#	all_set_compressed.update(set_compressed)
#
#	# relabel nodes
#	for idx, node in enumerate(G.nodes()):
#		G.nodes[node][node_label] = set_compressed[all_multisets[idx]]
#
#	# get the set of compressed labels
#	labels_comp = list(nx.get_node_attributes(G, node_label).values())
#	all_labels_ori.update(labels_comp)
#	all_num_of_each_label.append(dict(Counter(labels_comp)))
#	return


def wrapper_wl_iteration(node_label, itr_item):
	g = itr_item[0]
	i = itr_item[1]
	all_multisets = wl_iteration(g, node_label)
	return i, all_multisets


def compute_kernel_matrix(Kmatrix, all_num_of_each_label, Gn, parallel, n_jobs, chunksize, verbose):
	"""Compute kernel matrix using the base kernel.
	"""
	if parallel == 'imap_unordered':
		# compute kernels.
		def init_worker(alllabels_toshare):
			global G_alllabels
			G_alllabels = alllabels_toshare
		do_partial = partial(wrapper_compute_subtree_kernel, Kmatrix)
		parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
					glbv=(all_num_of_each_label,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)
	elif parallel is None:
		for i in range(len(Kmatrix)):
			for j in range(i, len(Kmatrix)):
				Kmatrix[i][j] = compute_subtree_kernel(all_num_of_each_label[i],
					   all_num_of_each_label[j], Kmatrix[i][j])
				Kmatrix[j][i] = Kmatrix[i][j]


def compute_subtree_kernel(num_of_each_label1, num_of_each_label2, kernel):
	"""Compute the subtree kernel.
	"""
	labels = set(list(num_of_each_label1.keys()) + list(num_of_each_label2.keys()))
	vector1 = np.array([(num_of_each_label1[label] 
						if (label in num_of_each_label1.keys()) else 0) 
						for label in labels])
	vector2 = np.array([(num_of_each_label2[label] 
						if (label in num_of_each_label2.keys()) else 0) 
						for label in labels])
	kernel += np.dot(vector1, vector2)
	return kernel


def wrapper_compute_subtree_kernel(Kmatrix, itr):
	i = itr[0]
	j = itr[1]
	return i, j, compute_subtree_kernel(G_alllabels[i], G_alllabels[j], Kmatrix[i][j])
				

def _wl_spkernel_do(Gn, node_label, edge_label, height):
	"""Compute Weisfeiler-Lehman shortest path kernels between graphs.
	
	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.	   
	node_label : string
		node attribute used as label.	  
	edge_label : string
		edge attribute used as label.	   
	height : int
		subtree height.
		
	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
	"""
	pass
	from gklearn.utils.utils import getSPGraph
	  
	# init.
	height = int(height)
	Kmatrix = np.zeros((len(Gn), len(Gn))) # init kernel

	Gn = [ getSPGraph(G, edge_weight = edge_label) for G in Gn ] # get shortest path graphs of Gn
	
	# initial for height = 0
	for i in range(0, len(Gn)):
		for j in range(i, len(Gn)):
			for e1 in Gn[i].edges(data = True):
				for e2 in Gn[j].edges(data = True):		  
					if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
						Kmatrix[i][j] += 1
			Kmatrix[j][i] = Kmatrix[i][j]
			
	# iterate each height
	for h in range(1, height + 1):
		all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
		num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
		for G in Gn: # for each graph
			set_multisets = []
			for node in G.nodes(data = True):
				# Multiset-label determination.
				multiset = [ G.node[neighbors][node_label] for neighbors in G[node[0]] ]
				# sorting each multiset
				multiset.sort()
				multiset = node[1][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
				set_multisets.append(multiset)		  

			# label compression
			set_unique = list(set(set_multisets)) # set of unique multiset labels
			# a dictionary mapping original labels to new ones. 
			set_compressed = {}
			# if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label 
			for value in set_unique:
				if value in all_set_compressed.keys():
					set_compressed.update({ value : all_set_compressed[value] })
				else:
					set_compressed.update({ value : str(num_of_labels_occured + 1) })
					num_of_labels_occured += 1

			all_set_compressed.update(set_compressed)
			
			# relabel nodes
			for node in G.nodes(data = True):
				node[1][node_label] = set_compressed[set_multisets[node[0]]]
				
		# Compute subtree kernel with h iterations and add it to the final kernel
		for i in range(0, len(Gn)):
			for j in range(i, len(Gn)):
				for e1 in Gn[i].edges(data = True):
					for e2 in Gn[j].edges(data = True):		  
						if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
							Kmatrix[i][j] += 1
				Kmatrix[j][i] = Kmatrix[i][j]
		
	return Kmatrix



def _wl_edgekernel_do(Gn, node_label, edge_label, height):
	"""Compute Weisfeiler-Lehman edge kernels between graphs.
	
	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.	   
	node_label : string
		node attribute used as label.	  
	edge_label : string
		edge attribute used as label.	   
	height : int
		subtree height.
		
	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
	"""	  
	pass
	# init.
	height = int(height)
	Kmatrix = np.zeros((len(Gn), len(Gn))) # init kernel
  
	# initial for height = 0
	for i in range(0, len(Gn)):
		for j in range(i, len(Gn)):
			for e1 in Gn[i].edges(data = True):
				for e2 in Gn[j].edges(data = True):		  
					if e1[2][edge_label] == e2[2][edge_label] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
						Kmatrix[i][j] += 1
			Kmatrix[j][i] = Kmatrix[i][j]
			
	# iterate each height
	for h in range(1, height + 1):
		all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
		num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
		for G in Gn: # for each graph
			set_multisets = []			
			for node in G.nodes(data = True):
				# Multiset-label determination.
				multiset = [ G.node[neighbors][node_label] for neighbors in G[node[0]] ]
				# sorting each multiset
				multiset.sort()
				multiset = node[1][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
				set_multisets.append(multiset)		  

			# label compression
			set_unique = list(set(set_multisets)) # set of unique multiset labels
			# a dictionary mapping original labels to new ones. 
			set_compressed = {}
			# if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label 
			for value in set_unique:
				if value in all_set_compressed.keys():
					set_compressed.update({ value : all_set_compressed[value] })
				else:
					set_compressed.update({ value : str(num_of_labels_occured + 1) })
					num_of_labels_occured += 1

			all_set_compressed.update(set_compressed)
			
			# relabel nodes
			for node in G.nodes(data = True):
				node[1][node_label] = set_compressed[set_multisets[node[0]]]
				
		# Compute subtree kernel with h iterations and add it to the final kernel
		for i in range(0, len(Gn)):
			for j in range(i, len(Gn)):
				for e1 in Gn[i].edges(data = True):
					for e2 in Gn[j].edges(data = True):		  
						if e1[2][edge_label] == e2[2][edge_label] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
							Kmatrix[i][j] += 1
				Kmatrix[j][i] = Kmatrix[i][j]
		
	return Kmatrix


def _wl_userkernel_do(Gn, node_label, edge_label, height, base_kernel):
	"""Compute Weisfeiler-Lehman kernels based on user-defined kernel between graphs.
	
	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.	   
	node_label : string
		node attribute used as label.	  
	edge_label : string
		edge attribute used as label.	   
	height : int
		subtree height.
	base_kernel : string
		Name of the base kernel function used in each iteration of WL kernel. This function returns a Numpy matrix, each element of which is the user-defined Weisfeiler-Lehman kernel between 2 praphs.
		
	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
	"""	  
	pass
	# init.
	height = int(height)
	Kmatrix = np.zeros((len(Gn), len(Gn))) # init kernel
  
	# initial for height = 0
	Kmatrix = base_kernel(Gn, node_label, edge_label)
			
	# iterate each height
	for h in range(1, height + 1):
		all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
		num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
		for G in Gn: # for each graph
			set_multisets = []		   
			for node in G.nodes(data = True):
				# Multiset-label determination.
				multiset = [ G.node[neighbors][node_label] for neighbors in G[node[0]] ]
				# sorting each multiset
				multiset.sort()
				multiset = node[1][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
				set_multisets.append(multiset)		  

			# label compression
			set_unique = list(set(set_multisets)) # set of unique multiset labels
			# a dictionary mapping original labels to new ones. 
			set_compressed = {}
			# if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label 
			for value in set_unique:
				if value in all_set_compressed.keys():
					set_compressed.update({ value : all_set_compressed[value] })
				else:
					set_compressed.update({ value : str(num_of_labels_occured + 1) })
					num_of_labels_occured += 1

			all_set_compressed.update(set_compressed)
			
			# relabel nodes
			for node in G.nodes(data = True):
				node[1][node_label] = set_compressed[set_multisets[node[0]]]
				
		# Compute kernel with h iterations and add it to the final kernel
		Kmatrix += base_kernel(Gn, node_label, edge_label)
		
	return Kmatrix
