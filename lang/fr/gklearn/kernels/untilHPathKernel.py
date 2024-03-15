"""
@author: linlin

@references: 

	[1] Liva Ralaivola, Sanjay J Swamidass, Hiroto Saigo, and Pierre 
	Baldi. Graph kernels for chemical informatics. Neural networks, 
	18(8):1093–1110, 2005.
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
from gklearn.utils.trie import Trie


def untilhpathkernel(*args,
					 node_label='atom',
					 edge_label='bond_type',
					 depth=10,
					 k_func='MinMax',
					 compute_method='trie',
					 parallel='imap_unordered',
					 n_jobs=None,
					 chunksize=None,
					 verbose=True):
	"""Compute path graph kernels up to depth/hight h between graphs.
	
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

	depth : integer
		Depth of search. Longest length of paths.

	k_func : function
		A kernel function applied using different notions of fingerprint 
		similarity, defining the type of feature map and normalization method 
		applied for the graph kernel. The Following choices are available:

		'MinMax': use the MiniMax kernel and counting feature map.

		'tanimoto': use the Tanimoto kernel and binary feature map.

		None: no sub-kernel is used, the kernel is computed directly.

	compute_method : string
		Computation method to store paths and compute the graph kernel. The 
		Following choices are available:

		'trie': store paths as tries.

		'naive': store paths to lists.

	n_jobs : int
		Number of jobs for parallelization.

	Return
	------
	Kmatrix : Numpy matrix
		Kernel matrix, each element of which is the path kernel up to h between
		2 praphs.
	"""
	# pre-process
	depth = int(depth)
	Gn = args[0] if len(args) == 1 else [args[0], args[1]]
	Gn = [g.copy() for g in Gn]
	Kmatrix = np.zeros((len(Gn), len(Gn)))
	ds_attrs = get_dataset_attributes(
		Gn,
		attr_names=['node_labeled', 'node_attr_dim', 'edge_labeled', 
					'edge_attr_dim', 'is_directed'],
		node_label=node_label, edge_label=edge_label)
	if k_func is not None:
		if not ds_attrs['node_labeled']:
			for G in Gn:
				nx.set_node_attributes(G, '0', 'atom')
		if not ds_attrs['edge_labeled']:
			for G in Gn:
				nx.set_edge_attributes(G, '0', 'bond_type')

	start_time = time.time()		

	if parallel == 'imap_unordered':
		# ---- use pool.imap_unordered to parallel and track progress. ----
		# get all paths of all graphs before computing kernels to save time,
		# but this may cost a lot of memory for large datasets.
		pool = Pool(n_jobs)
		itr = zip(Gn, range(0, len(Gn)))
		if chunksize is None:
			if len(Gn) < 100 * n_jobs:
				chunksize = int(len(Gn) / n_jobs) + 1
			else:
				chunksize = 100
		all_paths = [[] for _ in range(len(Gn))]
		if compute_method == 'trie' and k_func is not None:
			getps_partial = partial(wrapper_find_all_path_as_trie, depth, 
									ds_attrs, node_label, edge_label)
		elif compute_method != 'trie' and k_func is not None:  
			getps_partial = partial(wrapper_find_all_paths_until_length, depth, 
									ds_attrs, node_label, edge_label, True)  
		else: 
			getps_partial = partial(wrapper_find_all_paths_until_length, depth, 
									ds_attrs, node_label, edge_label, False)
		if verbose:
			iterator = tqdm(pool.imap_unordered(getps_partial, itr, chunksize),
							desc='getting paths', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(getps_partial, itr, chunksize)
		for i, ps in iterator:
			all_paths[i] = ps
		pool.close()
		pool.join()
	
#	for g in Gn:
#		if compute_method == 'trie' and k_func is not None:
#			find_all_path_as_trie(g, depth, ds_attrs, node_label, edge_label)
#		elif compute_method != 'trie' and k_func is not None:  
#			find_all_paths_until_length(g, depth, ds_attrs, node_label, edge_label)
#		else: 
#			find_all_paths_until_length(g, depth, ds_attrs, node_label, edge_label, False)
		
##	size = sys.getsizeof(all_paths)
##	for item in all_paths:
##		size += sys.getsizeof(item)
##		for pppps in item:
##			size += sys.getsizeof(pppps)
##	print(size)
#			
##	ttt = time.time()
##	# ---- ---- use pool.map to parallel ----
##	for i, ps in tqdm(
##			pool.map(getps_partial, range(0, len(Gn))),
##			desc='getting paths', file=sys.stdout):
##		all_paths[i] = ps
##	print(time.time() - ttt)
	 
		if compute_method == 'trie' and k_func is not None:
			def init_worker(trie_toshare):
				global G_trie
				G_trie = trie_toshare
			do_partial = partial(wrapper_uhpath_do_trie, k_func)
			parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
						glbv=(all_paths,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose) 
		elif compute_method != 'trie' and k_func is not None:
			def init_worker(plist_toshare):
				global G_plist
				G_plist = plist_toshare
			do_partial = partial(wrapper_uhpath_do_naive, k_func)   
			parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
						glbv=(all_paths,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose) 
		else:
			def init_worker(plist_toshare):
				global G_plist
				G_plist = plist_toshare
			do_partial = partial(wrapper_uhpath_do_kernelless, ds_attrs, edge_kernels)   
			parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
						glbv=(all_paths,), n_jobs=n_jobs, chunksize=chunksize, verbose=verbose) 
	
	elif parallel is None:
#		from pympler import asizeof
		# ---- direct running, normally use single CPU core. ----
#		print(asizeof.asized(all_paths, detail=1).format())
	
		if compute_method == 'trie':
			all_paths = [
				find_all_path_as_trie(Gn[i],
					 depth,
					 ds_attrs,
					 node_label=node_label,
					 edge_label=edge_label) for i in tqdm(
						range(0, len(Gn)), desc='getting paths', file=sys.stdout)
			]
#			sizeof_allpaths = asizeof.asizeof(all_paths)
#			print(sizeof_allpaths)
			pbar = tqdm(
				total=((len(Gn) + 1) * len(Gn) / 2),
				desc='Computing kernels',
				file=sys.stdout)
			for i in range(0, len(Gn)):
				for j in range(i, len(Gn)):
					Kmatrix[i][j] = _untilhpathkernel_do_trie(all_paths[i], 
						   all_paths[j], k_func)
					Kmatrix[j][i] = Kmatrix[i][j]
					pbar.update(1)
		else:
			all_paths = [
				find_all_paths_until_length(
					Gn[i],
					depth,
					ds_attrs,
					node_label=node_label,
					edge_label=edge_label) for i in tqdm(
						range(0, len(Gn)), desc='getting paths', file=sys.stdout)
			]
#			sizeof_allpaths = asizeof.asizeof(all_paths)
#			print(sizeof_allpaths)
			pbar = tqdm(
				total=((len(Gn) + 1) * len(Gn) / 2),
				desc='Computing kernels',
				file=sys.stdout)
			for i in range(0, len(Gn)):
				for j in range(i, len(Gn)):
					Kmatrix[i][j] = _untilhpathkernel_do_naive(all_paths[i], all_paths[j],
														 k_func)
					Kmatrix[j][i] = Kmatrix[i][j]
					pbar.update(1)

	run_time = time.time() - start_time
	if verbose:
		print("\n --- kernel matrix of path kernel up to %d of size %d built in %s seconds ---"
			  % (depth, len(Gn), run_time))

#	print(Kmatrix[0][0:10])
	return Kmatrix, run_time


def _untilhpathkernel_do_trie(trie1, trie2, k_func):
	"""Compute path graph kernels up to depth d between 2 graphs using trie.

	Parameters
	----------
	trie1, trie2 : list
		Tries that contains all paths in 2 graphs.
	k_func : function
		A kernel function applied using different notions of fingerprint 
		similarity.

	Return
	------
	kernel : float
		Path kernel up to h between 2 graphs.
	"""
	if k_func == 'tanimoto':	  
		# traverse all paths in graph1 and search them in graph2. Deep-first 
		# search is applied.
		def traverseTrie1t(root, trie2, setlist, pcurrent=[]):
			for key, node in root['children'].items():
				pcurrent.append(key)
				if node['isEndOfWord']:					
					setlist[1] += 1
					count2 = trie2.searchWord(pcurrent)
					if count2 != 0:
						setlist[0] += 1
				if node['children'] != {}:
					traverseTrie1t(node, trie2, setlist, pcurrent)
				else:
					del pcurrent[-1]
			if pcurrent != []:
				del pcurrent[-1]
				
				
		# traverse all paths in graph2 and find out those that are not in 
		# graph1. Deep-first search is applied. 
		def traverseTrie2t(root, trie1, setlist, pcurrent=[]):
			for key, node in root['children'].items():
				pcurrent.append(key)
				if node['isEndOfWord']:
		#					print(node['count'])
					count1 = trie1.searchWord(pcurrent)
					if count1 == 0:	
						setlist[1] += 1
				if node['children'] != {}:
					traverseTrie2t(node, trie1, setlist, pcurrent)
				else:
					del pcurrent[-1]
			if pcurrent != []:
				del pcurrent[-1]
		
		setlist = [0, 0] # intersection and union of path sets of g1, g2.
#		print(trie1.root)
#		print(trie2.root)
		traverseTrie1t(trie1.root, trie2, setlist)
#		print(setlist)
		traverseTrie2t(trie2.root, trie1, setlist)
#		print(setlist)
		kernel = setlist[0] / setlist[1]
		
	else: # MinMax kernel		  
		# traverse all paths in graph1 and search them in graph2. Deep-first 
		# search is applied.
		def traverseTrie1m(root, trie2, sumlist, pcurrent=[]):
			for key, node in root['children'].items():
				pcurrent.append(key)
				if node['isEndOfWord']:
		#					print(node['count'])
					count1 = node['count']
					count2 = trie2.searchWord(pcurrent)
					sumlist[0] += min(count1, count2)
					sumlist[1] += max(count1, count2)
				if node['children'] != {}:
					traverseTrie1m(node, trie2, sumlist, pcurrent)
				else:
					del pcurrent[-1]
			if pcurrent != []:
				del pcurrent[-1]
		
		# traverse all paths in graph2 and find out those that are not in 
		# graph1. Deep-first search is applied.				
		def traverseTrie2m(root, trie1, sumlist, pcurrent=[]):
			for key, node in root['children'].items():
				pcurrent.append(key)
				if node['isEndOfWord']:				   
		#					print(node['count'])
					count1 = trie1.searchWord(pcurrent)
					if count1 == 0:	
						sumlist[1] += node['count']
				if node['children'] != {}:
					traverseTrie2m(node, trie1, sumlist, pcurrent)
				else:
					del pcurrent[-1]
			if pcurrent != []:
				del pcurrent[-1]
		
		sumlist = [0, 0] # sum of mins and sum of maxs
#		print(trie1.root)
#		print(trie2.root)
		traverseTrie1m(trie1.root, trie2, sumlist)
#		print(sumlist)
		traverseTrie2m(trie2.root, trie1, sumlist)
#		print(sumlist)
		kernel = sumlist[0] / sumlist[1]

	return kernel


def wrapper_uhpath_do_trie(k_func, itr):
	i = itr[0]
	j = itr[1]
	return i, j, _untilhpathkernel_do_trie(G_trie[i], G_trie[j], k_func)
		

def _untilhpathkernel_do_naive(paths1, paths2, k_func):
	"""Compute path graph kernels up to depth d between 2 graphs naively.

	Parameters
	----------
	paths_list : list of list
		List of list of paths in all graphs, where for unlabeled graphs, each 
		path is represented by a list of nodes; while for labeled graphs, each 
		path is represented by a string consists of labels of nodes and/or 
		edges on that path.
	k_func : function
		A kernel function applied using different notions of fingerprint 
		similarity.

	Return
	------
	kernel : float
		Path kernel up to h between 2 graphs.
	"""
	all_paths = list(set(paths1 + paths2))

	if k_func == 'tanimoto':
		length_union = len(set(paths1 + paths2))
		kernel = (len(set(paths1)) + len(set(paths2)) -
				  length_union) / length_union
#		vector1 = [(1 if path in paths1 else 0) for path in all_paths]
#		vector2 = [(1 if path in paths2 else 0) for path in all_paths]
#		kernel_uv = np.dot(vector1, vector2)
#		kernel = kernel_uv / (len(set(paths1)) + len(set(paths2)) - kernel_uv)

	else:  # MinMax kernel
		path_count1 = Counter(paths1)
		path_count2 = Counter(paths2)
		vector1 = [(path_count1[key] if (key in path_count1.keys()) else 0)
				   for key in all_paths]
		vector2 = [(path_count2[key] if (key in path_count2.keys()) else 0)
				   for key in all_paths]
		kernel = np.sum(np.minimum(vector1, vector2)) / \
			np.sum(np.maximum(vector1, vector2))

	return kernel


def wrapper_uhpath_do_naive(k_func, itr):
	i = itr[0]
	j = itr[1]
	return i, j, _untilhpathkernel_do_naive(G_plist[i], G_plist[j], k_func)


def _untilhpathkernel_do_kernelless(paths1, paths2, k_func):
	"""Compute path graph kernels up to depth d between 2 graphs naively.

	Parameters
	----------
	paths_list : list of list
		List of list of paths in all graphs, where for unlabeled graphs, each 
		path is represented by a list of nodes; while for labeled graphs, each 
		path is represented by a string consists of labels of nodes and/or 
		edges on that path.
	k_func : function
		A kernel function applied using different notions of fingerprint 
		similarity.

	Return
	------
	kernel : float
		Path kernel up to h between 2 graphs.
	"""
	all_paths = list(set(paths1 + paths2))

	if k_func == 'tanimoto':
		length_union = len(set(paths1 + paths2))
		kernel = (len(set(paths1)) + len(set(paths2)) -
				  length_union) / length_union
#		vector1 = [(1 if path in paths1 else 0) for path in all_paths]
#		vector2 = [(1 if path in paths2 else 0) for path in all_paths]
#		kernel_uv = np.dot(vector1, vector2)
#		kernel = kernel_uv / (len(set(paths1)) + len(set(paths2)) - kernel_uv)

	else:  # MinMax kernel
		path_count1 = Counter(paths1)
		path_count2 = Counter(paths2)
		vector1 = [(path_count1[key] if (key in path_count1.keys()) else 0)
				   for key in all_paths]
		vector2 = [(path_count2[key] if (key in path_count2.keys()) else 0)
				   for key in all_paths]
		kernel = np.sum(np.minimum(vector1, vector2)) / \
			np.sum(np.maximum(vector1, vector2))

	return kernel


def wrapper_uhpath_do_kernelless(k_func, itr):
	i = itr[0]
	j = itr[1]
	return i, j, _untilhpathkernel_do_kernelless(G_plist[i], G_plist[j], k_func)


# @todo: (can be removed maybe)  this method find paths repetively, it could be faster.
def find_all_paths_until_length(G,
								length,
								ds_attrs,
								node_label='atom',
								edge_label='bond_type',
								tolabelseqs=True):
	"""Find all paths no longer than a certain maximum length in a graph. A 
	recursive depth first search is applied.

	Parameters
	----------
	G : NetworkX graphs
		The graph in which paths are searched.
	length : integer
		The maximum length of paths.
	ds_attrs: dict
		Dataset attributes.
	node_label : string
		Node attribute used as label. The default node label is atom.
	edge_label : string
		Edge attribute used as label. The default edge label is bond_type.

	Return
	------
	path : list
		List of paths retrieved, where for unlabeled graphs, each path is 
		represented by a list of nodes; while for labeled graphs, each path is 
		represented by a list of strings consists of labels of nodes and/or 
		edges on that path.
	"""
	# path_l = [tuple([n]) for n in G.nodes]  # paths of length l
	# all_paths = path_l[:]
	# for l in range(1, length + 1):
	#	 path_l_new = []
	#	 for path in path_l:
	#		 for neighbor in G[path[-1]]:
	#			 if len(path) < 2 or neighbor != path[-2]:
	#				 tmp = path + (neighbor, )
	#				 if tuple(tmp[::-1]) not in path_l_new:
	#					 path_l_new.append(tuple(tmp))

	#	 all_paths += path_l_new
	#	 path_l = path_l_new[:]

	path_l = [[n] for n in G.nodes]  # paths of length l
	all_paths = [p.copy() for p in path_l]
	for l in range(1, length + 1):
		path_lplus1 = []
		for path in path_l:
			for neighbor in G[path[-1]]:
				if neighbor not in path:
					tmp = path + [neighbor]
#					if tmp[::-1] not in path_lplus1:
					path_lplus1.append(tmp)

		all_paths += path_lplus1
		path_l = [p.copy() for p in path_lplus1]

	# for i in range(0, length + 1):
	#	 new_paths = find_all_paths(G, i)
	#	 if new_paths == []:
	#		 break
	#	 all_paths.extend(new_paths)

	# consider labels
#	print(paths2labelseqs(all_paths, G, ds_attrs, node_label, edge_label))
#	print()
	return (paths2labelseqs(all_paths, G, ds_attrs, node_label, edge_label) 
			if tolabelseqs else all_paths)
		
		
def wrapper_find_all_paths_until_length(length, ds_attrs, node_label, 
									 edge_label, tolabelseqs, itr_item):
	g = itr_item[0]
	i = itr_item[1]
	return i, find_all_paths_until_length(g, length, ds_attrs,
				node_label=node_label, edge_label=edge_label, 
				tolabelseqs=tolabelseqs)


def find_all_path_as_trie(G,
						 length,
						 ds_attrs,
						 node_label='atom',
						 edge_label='bond_type'):
#	time1 = time.time()
	
#	all_path = find_all_paths_until_length(G, length, ds_attrs, 
#										   node_label=node_label,
#										   edge_label=edge_label)
#	ptrie = Trie()
#	for path in all_path:
#		ptrie.insertWord(path)
	
#	ptrie = Trie()
#	path_l = [[n] for n in G.nodes]  # paths of length l
#	path_l_str = paths2labelseqs(path_l, G, ds_attrs, node_label, edge_label)
#	for p in path_l_str:
#		ptrie.insertWord(p)
#	for l in range(1, length + 1):
#		path_lplus1 = []
#		for path in path_l:
#			for neighbor in G[path[-1]]:
#				if neighbor not in path:
#					tmp = path + [neighbor]
##					if tmp[::-1] not in path_lplus1:
#					path_lplus1.append(tmp)
#		path_l = path_lplus1[:]
#		# consider labels
#		path_l_str = paths2labelseqs(path_l, G, ds_attrs, node_label, edge_label)
#		for p in path_l_str:
#			ptrie.insertWord(p)
#	
#	print(time.time() - time1)
#	print(ptrie.root)
#	print()
			
			
	# traverse all paths up to length h in a graph and construct a trie with 
	# them. Deep-first search is applied. Notice the reverse of each path is 
	# also stored to the trie.			   
	def traverseGraph(root, ptrie, length, G, ds_attrs, node_label, edge_label,
					  pcurrent=[]):
		if len(pcurrent) < length + 1:
			for neighbor in G[root]:
				if neighbor not in pcurrent:
					pcurrent.append(neighbor)
					plstr = paths2labelseqs([pcurrent], G, ds_attrs, 
											node_label, edge_label)
					ptrie.insertWord(plstr[0])
					traverseGraph(neighbor, ptrie, length, G, ds_attrs, 
								   node_label, edge_label, pcurrent)
		del pcurrent[-1]


	ptrie = Trie()
	path_l = [[n] for n in G.nodes]  # paths of length l
	path_l_str = paths2labelseqs(path_l, G, ds_attrs, node_label, edge_label)
	for p in path_l_str:
		ptrie.insertWord(p)
	for n in G.nodes:
		traverseGraph(n, ptrie, length, G, ds_attrs, node_label, edge_label, 
					   pcurrent=[n])
		
		
#	def traverseGraph(root, all_paths, length, G, ds_attrs, node_label, edge_label,
#					  pcurrent=[]):
#		if len(pcurrent) < length + 1:
#			for neighbor in G[root]:
#				if neighbor not in pcurrent:
#					pcurrent.append(neighbor)
#					plstr = paths2labelseqs([pcurrent], G, ds_attrs, 
#											node_label, edge_label)
#					all_paths.append(pcurrent[:])
#					traverseGraph(neighbor, all_paths, length, G, ds_attrs, 
#								   node_label, edge_label, pcurrent)
#		del pcurrent[-1]
#
#
#	path_l = [[n] for n in G.nodes]  # paths of length l
#	all_paths = path_l[:]
#	path_l_str = paths2labelseqs(path_l, G, ds_attrs, node_label, edge_label)
##	for p in path_l_str:
##		ptrie.insertWord(p)
#	for n in G.nodes:
#		traverseGraph(n, all_paths, length, G, ds_attrs, node_label, edge_label, 
#					   pcurrent=[n])
	
#	print(ptrie.root)
	return ptrie


def wrapper_find_all_path_as_trie(length, ds_attrs, node_label, 
									 edge_label, itr_item):
	g = itr_item[0]
	i = itr_item[1]
	return i, find_all_path_as_trie(g, length, ds_attrs,
				node_label=node_label, edge_label=edge_label)


def paths2labelseqs(plist, G, ds_attrs, node_label, edge_label):
	if ds_attrs['node_labeled']:
		if ds_attrs['edge_labeled']:
			path_strs = [
				tuple(
					list(
						chain.from_iterable(
							(G.nodes[node][node_label],
							 G[node][path[idx + 1]][edge_label])
							for idx, node in enumerate(path[:-1]))) +
					[G.nodes[path[-1]][node_label]]) for path in plist
			]
			# path_strs = []
			# for path in all_paths:
			#	 strlist = list(
			#		 chain.from_iterable((G.node[node][node_label],
			#							  G[node][path[idx + 1]][edge_label])
			#							 for idx, node in enumerate(path[:-1])))
			#	 strlist.append(G.node[path[-1]][node_label])
			#	 path_strs.append(tuple(strlist))
		else:
			path_strs = [
				tuple([G.nodes[node][node_label] for node in path])
				for path in plist
			]
		return path_strs
	else:
		if ds_attrs['edge_labeled']:
			return [
				tuple([] if len(path) == 1 else [
					G[node][path[idx + 1]][edge_label]
					for idx, node in enumerate(path[:-1])
				]) for path in plist
			]
		else:
			return [tuple(['0' for node in path]) for path in plist]
#			return [tuple([len(path)]) for path in all_paths]   

#
#def paths2GSuffixTree(paths):
#	return Tree(paths, builder=ukkonen.Builder)


# def find_paths(G, source_node, length):
#	 """Find all paths no longer than a certain length those start from a source node. A recursive depth first search is applied.

#	 Parameters
#	 ----------
#	 G : NetworkX graphs
#		 The graph in which paths are searched.
#	 source_node : integer
#		 The number of the node from where all paths start.
#	 length : integer
#		 The length of paths.

#	 Return
#	 ------
#	 path : list of list
#		 List of paths retrieved, where each path is represented by a list of nodes.
#	 """
#	 return [[source_node]] if length == 0 else \
#		 [[source_node] + path for neighbor in G[source_node]
#		  for path in find_paths(G, neighbor, length - 1) if source_node not in path]

# def find_all_paths(G, length):
#	 """Find all paths with a certain length in a graph. A recursive depth first search is applied.

#	 Parameters
#	 ----------
#	 G : NetworkX graphs
#		 The graph in which paths are searched.
#	 length : integer
#		 The length of paths.

#	 Return
#	 ------
#	 path : list of list
#		 List of paths retrieved, where each path is represented by a list of nodes.
#	 """
#	 all_paths = []
#	 for node in G:
#		 all_paths.extend(find_paths(G, node, length))

#	 # The following process is not carried out according to the original article
#	 # all_paths_r = [ path[::-1] for path in all_paths ]

#	 # # For each path, two presentation are retrieved from its two extremities. Remove one of them.
#	 # for idx, path in enumerate(all_paths[:-1]):
#	 #	 for path2 in all_paths_r[idx+1::]:
#	 #		 if path == path2:
#	 #			 all_paths[idx] = []
#	 #			 break

#	 # return list(filter(lambda a: a != [], all_paths))
#	 return all_paths
