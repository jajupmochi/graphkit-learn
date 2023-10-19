"""
@author: linlin

@references: 

	[1] Gaüzère B, Brun L, Villemin D. Two new graphs kernels in 
	chemoinformatics. Pattern Recognition Letters. 2012 Nov 1;33(15):2038-47.
"""

import sys
import time
from collections import Counter
from itertools import chain
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import networkx as nx
import numpy as np

from gklearn.utils.graphdataset import get_dataset_attributes
from gklearn.utils.parallel import parallel_gm

def treeletkernel(*args, 
				  sub_kernel, 
				  node_label='atom', 
				  edge_label='bond_type', 
				  parallel='imap_unordered',
				  n_jobs=None, 
				  chunksize=None,
				  verbose=True):
	"""Compute treelet graph kernels between graphs.

	Parameters
	----------
	Gn : List of NetworkX graph
		List of graphs between which the kernels are computed.
	
	G1, G2 : NetworkX graphs
		Two graphs between which the kernel is computed.

	sub_kernel : function
		The sub-kernel between 2 real number vectors. Each vector counts the
		numbers of isomorphic treelets in a graph.

	node_label : string
		Node attribute used as label. The default node label is atom.   

	edge_label : string
		Edge attribute used as label. The default edge label is bond_type.

	parallel : string/None
		Which paralleliztion method is applied to compute the kernel. The 
		Following choices are available:

		'imap_unordered': use Python's multiprocessing.Pool.imap_unordered
		method.

		None: no parallelization is applied.

	n_jobs : int
		Number of jobs for parallelization. The default is to use all 
		computational cores. This argument is only valid when one of the 
		parallelization method is applied.

	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the treelet kernel between 2 praphs.
	"""
	# pre-process
	Gn = args[0] if len(args) == 1 else [args[0], args[1]]
	Gn = [g.copy() for g in Gn]
	Kmatrix = np.zeros((len(Gn), len(Gn)))
	ds_attrs = get_dataset_attributes(Gn,
		attr_names=['node_labeled', 'edge_labeled', 'is_directed'],
		node_label=node_label, edge_label=edge_label)
	labeled = False
	if ds_attrs['node_labeled'] or ds_attrs['edge_labeled']:
		labeled = True
		if not ds_attrs['node_labeled']:
			for G in Gn:
				nx.set_node_attributes(G, '0', 'atom')
		if not ds_attrs['edge_labeled']:
			for G in Gn:
				nx.set_edge_attributes(G, '0', 'bond_type')
	
	start_time = time.time()
	
	# ---- use pool.imap_unordered to parallel and track progress. ----
	if parallel == 'imap_unordered':
		# get all canonical keys of all graphs before computing kernels to save 
		# time, but this may cost a lot of memory for large dataset.
		pool = Pool(n_jobs)
		itr = zip(Gn, range(0, len(Gn)))
		if chunksize is None:
			if len(Gn) < 100 * n_jobs:
				chunksize = int(len(Gn) / n_jobs) + 1
			else:
				chunksize = 100
		canonkeys = [[] for _ in range(len(Gn))]
		get_partial = partial(wrapper_get_canonkeys, node_label, edge_label, 
								labeled, ds_attrs['is_directed'])
		if verbose:
			iterator = tqdm(pool.imap_unordered(get_partial, itr, chunksize),
							desc='getting canonkeys', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(get_partial, itr, chunksize)
		for i, ck in iterator:
			canonkeys[i] = ck
		pool.close()
		pool.join()
		
		# compute kernels.
		def init_worker(canonkeys_toshare):
			global G_canonkeys
			G_canonkeys = canonkeys_toshare
		do_partial = partial(wrapper_treeletkernel_do, sub_kernel)
		parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
					glbv=(canonkeys,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)
		
	# ---- do not use parallelization. ----
	elif parallel is None:
		# get all canonical keys of all graphs before computing kernels to save 
		# time, but this may cost a lot of memory for large dataset.
		canonkeys = []
		for g in (tqdm(Gn, desc='getting canonkeys', file=sys.stdout) if verbose else Gn):
			canonkeys.append(get_canonkeys(g, node_label, edge_label, labeled, 
										   ds_attrs['is_directed']))
		
		# compute kernels.
		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(Gn)), 2)
		for i, j in (tqdm(itr, desc='getting canonkeys', file=sys.stdout) if verbose else itr):
			Kmatrix[i][j] = _treeletkernel_do(canonkeys[i], canonkeys[j], sub_kernel)
			Kmatrix[j][i] = Kmatrix[i][j] # @todo: no directed graph considered?
			
	else:
		raise Exception('No proper parallelization method designated.')

	
	run_time = time.time() - start_time
	if verbose:
		print("\n --- treelet kernel matrix of size %d built in %s seconds ---" 
			  % (len(Gn), run_time))
		
	return Kmatrix, run_time


def _treeletkernel_do(canonkey1, canonkey2, sub_kernel):
	"""Compute treelet graph kernel between 2 graphs.
	
	Parameters
	----------
	canonkey1, canonkey2 : list
		List of canonical keys in 2 graphs, where each key is represented by a string.
		
	Return
	------
	kernel : float
		Treelet Kernel between 2 graphs.
	"""
	keys = set(canonkey1.keys()) | set(canonkey2.keys()) # find union of canonical keys in both graphs
	vector1 = np.array([canonkey1.get(key,0) for key in keys])
	vector2 = np.array([canonkey2.get(key,0) for key in keys]) 
	kernel = sub_kernel(vector1, vector2) 
	return kernel


def wrapper_treeletkernel_do(sub_kernel, itr):
	i = itr[0]
	j = itr[1]
	return i, j, _treeletkernel_do(G_canonkeys[i], G_canonkeys[j], sub_kernel)


def get_canonkeys(G, node_label, edge_label, labeled, is_directed):
	"""Generate canonical keys of all treelets in a graph.
	
	Parameters
	----------
	G : NetworkX graphs
		The graph in which keys are generated.
	node_label : string
		node attribute used as label. The default node label is atom.		
	edge_label : string
		edge attribute used as label. The default edge label is bond_type.
	labeled : boolean
		Whether the graphs are labeled. The default is True.
		
	Return
	------
	canonkey/canonkey_l : dict
		For unlabeled graphs, canonkey is a dictionary which records amount of 
		every tree pattern. For labeled graphs, canonkey_l is one which keeps 
		track of amount of every treelet.
	"""
	patterns = {} # a dictionary which consists of lists of patterns for all graphlet.
	canonkey = {} # canonical key, a dictionary which records amount of every tree pattern.

	### structural analysis ###
	### In this section, a list of patterns is generated for each graphlet, 
	### where every pattern is represented by nodes ordered by Morgan's 
	### extended labeling.
	# linear patterns
	patterns['0'] = G.nodes()
	canonkey['0'] = nx.number_of_nodes(G)
	for i in range(1, 6): # for i in range(1, 6):
		patterns[str(i)] = find_all_paths(G, i, is_directed)
		canonkey[str(i)] = len(patterns[str(i)])

	# n-star patterns
	patterns['3star'] = [[node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 3]
	patterns['4star'] = [[node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 4] # @todo: check self loop.
	patterns['5star'] = [[node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 5]		
	# n-star patterns
	canonkey['6'] = len(patterns['3star'])
	canonkey['8'] = len(patterns['4star'])
	canonkey['d'] = len(patterns['5star'])

	# pattern 7
	patterns['7'] = [] # the 1st line of Table 1 in Ref [1]
	for pattern in patterns['3star']:
		for i in range(1, len(pattern)): # for each neighbor of node 0
			if G.degree(pattern[i]) >= 2:
				pattern_t = pattern[:]
				# set the node with degree >= 2 as the 4th node
				pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
				for neighborx in G[pattern[i]]:
					if neighborx != pattern[0]:
						new_pattern = pattern_t + [neighborx]
						patterns['7'].append(new_pattern)
	canonkey['7'] = len(patterns['7'])

	# pattern 11
	patterns['11'] = [] # the 4th line of Table 1 in Ref [1]
	for pattern in patterns['4star']:
		for i in range(1, len(pattern)):
			if G.degree(pattern[i]) >= 2:
				pattern_t = pattern[:]
				pattern_t[i], pattern_t[4] = pattern_t[4], pattern_t[i]
				for neighborx in G[pattern[i]]:
					if neighborx != pattern[0]:
						new_pattern = pattern_t + [ neighborx ]
						patterns['11'].append(new_pattern)
	canonkey['b'] = len(patterns['11'])

	# pattern 12
	patterns['12'] = [] # the 5th line of Table 1 in Ref [1]
	rootlist = [] # a list of root nodes, whose extended labels are 3
	for pattern in patterns['3star']:
		if pattern[0] not in rootlist: # prevent to count the same pattern twice from each of the two root nodes
			rootlist.append(pattern[0])
			for i in range(1, len(pattern)):
				if G.degree(pattern[i]) >= 3:
					rootlist.append(pattern[i])
					pattern_t = pattern[:]
					pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
					for neighborx1 in G[pattern[i]]:
						if neighborx1 != pattern[0]:
							for neighborx2 in G[pattern[i]]:
								if neighborx1 > neighborx2 and neighborx2 != pattern[0]:
									new_pattern = pattern_t + [neighborx1] + [neighborx2]
#						 new_patterns = [ pattern + [neighborx1] + [neighborx2] for neighborx1 in G[pattern[i]] if neighborx1 != pattern[0] for neighborx2 in G[pattern[i]] if (neighborx1 > neighborx2 and neighborx2 != pattern[0]) ]
									patterns['12'].append(new_pattern)
	canonkey['c'] = int(len(patterns['12']) / 2)

	# pattern 9
	patterns['9'] = [] # the 2nd line of Table 1 in Ref [1]
	for pattern in patterns['3star']:
		for pairs in [ [neighbor1, neighbor2] for neighbor1 in G[pattern[0]] if G.degree(neighbor1) >= 2 \
			for neighbor2 in G[pattern[0]] if G.degree(neighbor2) >= 2 if neighbor1 > neighbor2 ]:
			pattern_t = pattern[:]
			# move nodes with extended labels 4 to specific position to correspond to their children
			pattern_t[pattern_t.index(pairs[0])], pattern_t[2] = pattern_t[2], pattern_t[pattern_t.index(pairs[0])]
			pattern_t[pattern_t.index(pairs[1])], pattern_t[3] = pattern_t[3], pattern_t[pattern_t.index(pairs[1])]
			for neighborx1 in G[pairs[0]]:
				if neighborx1 != pattern[0]:
					for neighborx2 in G[pairs[1]]:
						if neighborx2 != pattern[0]:
							new_pattern = pattern_t + [neighborx1] + [neighborx2]
							patterns['9'].append(new_pattern)
	canonkey['9'] = len(patterns['9'])

	# pattern 10
	patterns['10'] = [] # the 3rd line of Table 1 in Ref [1]
	for pattern in patterns['3star']:		
		for i in range(1, len(pattern)):
			if G.degree(pattern[i]) >= 2:
				for neighborx in G[pattern[i]]:
					if neighborx != pattern[0] and G.degree(neighborx) >= 2:
						pattern_t = pattern[:]
						pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
						new_patterns = [ pattern_t + [neighborx] + [neighborxx] for neighborxx in G[neighborx] if neighborxx != pattern[i] ]
						patterns['10'].extend(new_patterns)
	canonkey['a'] = len(patterns['10'])

	### labeling information ###
	### In this section, a list of canonical keys is generated for every 
	### pattern obtained in the structural analysis section above, which is a 
	### string corresponding to a unique treelet. A dictionary is built to keep
	### track of the amount of every treelet.
	if labeled == True:
		canonkey_l = {} # canonical key, a dictionary which keeps track of amount of every treelet.

		# linear patterns
		canonkey_t = Counter(list(nx.get_node_attributes(G, node_label).values()))
		for key in canonkey_t:
			canonkey_l[('0', key)] = canonkey_t[key]

		for i in range(1, 6): # for i in range(1, 6):
			treelet = []
			for pattern in patterns[str(i)]:
				canonlist = list(chain.from_iterable((G.nodes[node][node_label], \
					G[node][pattern[idx+1]][edge_label]) for idx, node in enumerate(pattern[:-1])))
				canonlist.append(G.nodes[pattern[-1]][node_label])
				canonkey_t = canonlist if canonlist < canonlist[::-1] else canonlist[::-1]
				treelet.append(tuple([str(i)] + canonkey_t))
			canonkey_l.update(Counter(treelet))

		# n-star patterns
		for i in range(3, 6):
			treelet = []
			for pattern in patterns[str(i) + 'star']:
				canonlist = [tuple((G.nodes[leaf][node_label], 
									G[leaf][pattern[0]][edge_label])) for leaf in pattern[1:]]
				canonlist.sort()
				canonlist = list(chain.from_iterable(canonlist))
				canonkey_t = tuple(['d' if i == 5 else str(i * 2)] + 
								   [G.nodes[pattern[0]][node_label]] + canonlist)
				treelet.append(canonkey_t)
			canonkey_l.update(Counter(treelet))

		# pattern 7
		treelet = []
		for pattern in patterns['7']:
			canonlist = [tuple((G.nodes[leaf][node_label], 
								G[leaf][pattern[0]][edge_label])) for leaf in pattern[1:3]]
			canonlist.sort()
			canonlist = list(chain.from_iterable(canonlist))
			canonkey_t = tuple(['7'] + [G.nodes[pattern[0]][node_label]] + canonlist 
							   + [G.nodes[pattern[3]][node_label]] 
							   + [G[pattern[3]][pattern[0]][edge_label]]
							   + [G.nodes[pattern[4]][node_label]] 
							   + [G[pattern[4]][pattern[3]][edge_label]])
			treelet.append(canonkey_t)
		canonkey_l.update(Counter(treelet))

		# pattern 11
		treelet = []
		for pattern in patterns['11']:
			canonlist = [tuple((G.nodes[leaf][node_label], 
								G[leaf][pattern[0]][edge_label])) for leaf in pattern[1:4]]
			canonlist.sort()
			canonlist = list(chain.from_iterable(canonlist))
			canonkey_t = tuple(['b'] + [G.nodes[pattern[0]][node_label]] + canonlist 
							   + [G.nodes[pattern[4]][node_label]] 
							   + [G[pattern[4]][pattern[0]][edge_label]]
							   + [G.nodes[pattern[5]][node_label]] 
							   + [G[pattern[5]][pattern[4]][edge_label]])
			treelet.append(canonkey_t)
		canonkey_l.update(Counter(treelet))

		# pattern 10
		treelet = []
		for pattern in patterns['10']:
			canonkey4 = [G.nodes[pattern[5]][node_label], G[pattern[5]][pattern[4]][edge_label]]
			canonlist = [tuple((G.nodes[leaf][node_label], 
								G[leaf][pattern[0]][edge_label])) for leaf in pattern[1:3]]
			canonlist.sort()
			canonkey0 = list(chain.from_iterable(canonlist))
			canonkey_t = tuple(['a'] + [G.nodes[pattern[3]][node_label]] 
							   + [G.nodes[pattern[4]][node_label]] 
							   + [G[pattern[4]][pattern[3]][edge_label]] 
							   + [G.nodes[pattern[0]][node_label]] 
							   + [G[pattern[0]][pattern[3]][edge_label]] 
							   + canonkey4 + canonkey0)
			treelet.append(canonkey_t)
		canonkey_l.update(Counter(treelet))

		# pattern 12
		treelet = []
		for pattern in patterns['12']:
			canonlist0 = [tuple((G.nodes[leaf][node_label], 
								 G[leaf][pattern[0]][edge_label])) for leaf in pattern[1:3]]
			canonlist0.sort()
			canonlist0 = list(chain.from_iterable(canonlist0))
			canonlist3 = [tuple((G.nodes[leaf][node_label], 
								 G[leaf][pattern[3]][edge_label])) for leaf in pattern[4:6]]
			canonlist3.sort()
			canonlist3 = list(chain.from_iterable(canonlist3))
			
			# 2 possible key can be generated from 2 nodes with extended label 3, 
			# select the one with lower lexicographic order.
			canonkey_t1 = tuple(['c'] + [G.nodes[pattern[0]][node_label]] + canonlist0 
								+ [G.nodes[pattern[3]][node_label]] 
								+ [G[pattern[3]][pattern[0]][edge_label]] 
								+ canonlist3)
			canonkey_t2 = tuple(['c'] + [G.nodes[pattern[3]][node_label]] + canonlist3 
								+ [G.nodes[pattern[0]][node_label]] 
								+ [G[pattern[0]][pattern[3]][edge_label]] 
								+ canonlist0)
			treelet.append(canonkey_t1 if canonkey_t1 < canonkey_t2 else canonkey_t2)
		canonkey_l.update(Counter(treelet))

		# pattern 9
		treelet = []
		for pattern in patterns['9']:
			canonkey2 = [G.nodes[pattern[4]][node_label], G[pattern[4]][pattern[2]][edge_label]]
			canonkey3 = [G.nodes[pattern[5]][node_label], G[pattern[5]][pattern[3]][edge_label]]
			prekey2 = [G.nodes[pattern[2]][node_label], G[pattern[2]][pattern[0]][edge_label]]
			prekey3 = [G.nodes[pattern[3]][node_label], G[pattern[3]][pattern[0]][edge_label]]
			if prekey2 + canonkey2 < prekey3 + canonkey3:
				canonkey_t = [G.nodes[pattern[1]][node_label]] \
							 + [G[pattern[1]][pattern[0]][edge_label]] \
							 + prekey2 + prekey3 + canonkey2 + canonkey3
			else:
				canonkey_t = [G.nodes[pattern[1]][node_label]] \
							 + [G[pattern[1]][pattern[0]][edge_label]] \
							 + prekey3 + prekey2 + canonkey3 + canonkey2
			treelet.append(tuple(['9'] + [G.nodes[pattern[0]][node_label]] + canonkey_t))
		canonkey_l.update(Counter(treelet))

		return canonkey_l

	return canonkey


def wrapper_get_canonkeys(node_label, edge_label, labeled, is_directed, itr_item):
	g = itr_item[0]
	i = itr_item[1]
	return i, get_canonkeys(g, node_label, edge_label, labeled, is_directed)
	

def find_paths(G, source_node, length):
	"""Find all paths with a certain length those start from a source node. 
	A recursive depth first search is applied.
	
	Parameters
	----------
	G : NetworkX graphs
		The graph in which paths are searched.
	source_node : integer
		The number of the node from where all paths start.
	length : integer
		The length of paths.
		
	Return
	------
	path : list of list
		List of paths retrieved, where each path is represented by a list of nodes.
	"""
	if length == 0:
		return [[source_node]]
	path = [[source_node] + path for neighbor in G[source_node] \
		for path in find_paths(G, neighbor, length - 1) if source_node not in path]
	return path


def find_all_paths(G, length, is_directed):
	"""Find all paths with a certain length in a graph. A recursive depth first
	search is applied.
	
	Parameters
	----------
	G : NetworkX graphs
		The graph in which paths are searched.
	length : integer
		The length of paths.
		
	Return
	------
	path : list of list
		List of paths retrieved, where each path is represented by a list of nodes.
	"""
	all_paths = []
	for node in G:
		all_paths.extend(find_paths(G, node, length))
		
	if not is_directed:
		# For each path, two presentations are retrieved from its two extremities. 
		# Remove one of them.
		all_paths_r = [path[::-1] for path in all_paths]  
		for idx, path in enumerate(all_paths[:-1]):
			for path2 in all_paths_r[idx+1::]:
				if path == path2:
					all_paths[idx] = []
					break
		all_paths = list(filter(lambda a: a != [], all_paths))
			
	return all_paths
