#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:33:13 2020

@author: ljia

@references: 

	[1] Liva Ralaivola, Sanjay J Swamidass, Hiroto Saigo, and Pierre 
	Baldi. Graph kernels for chemical informatics. Neural networks, 
	18(8):1093–1110, 2005.
"""
import sys
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import networkx as nx
from collections import Counter
from functools import partial
from gklearn.utils import SpecialLabel
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.kernels import GraphKernel
from gklearn.utils import Trie


class PathUpToH(GraphKernel): # @todo: add function for k_func == None
	
	def __init__(self, **kwargs):
		GraphKernel.__init__(self)
		self.__node_labels = kwargs.get('node_labels', [])
		self.__edge_labels = kwargs.get('edge_labels', [])
		self.__depth = int(kwargs.get('depth', 10))
		self.__k_func = kwargs.get('k_func', 'MinMax')
		self.__compute_method = kwargs.get('compute_method', 'trie')
		self.__ds_infos = kwargs.get('ds_infos', {})


	def _compute_gm_series(self):
		self.__add_dummy_labels(self._graphs)
		
		from itertools import combinations_with_replacement
		itr_kernel = combinations_with_replacement(range(0, len(self._graphs)), 2)	
		if self._verbose >= 2:
			iterator_ps = tqdm(range(0, len(self._graphs)), desc='getting paths', file=sys.stdout)
			iterator_kernel = tqdm(itr_kernel, desc='calculating kernels', file=sys.stdout)
		else:
			iterator_ps = range(0, len(self._graphs))
			iterator_kernel = itr_kernel
			
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		if self.__compute_method == 'trie':
			all_paths = [self.__find_all_path_as_trie(self._graphs[i]) for i in iterator_ps]
			for i, j in iterator_kernel:
				kernel = self.__kernel_do_trie(all_paths[i], all_paths[j])
				gram_matrix[i][j] = kernel
				gram_matrix[j][i] = kernel
		else:
			all_paths = [self.__find_all_paths_until_length(self._graphs[i]) for i in iterator_ps]
			for i, j in iterator_kernel:
				kernel = self.__kernel_do_naive(all_paths[i], all_paths[j])
				gram_matrix[i][j] = kernel
				gram_matrix[j][i] = kernel
				
		return gram_matrix
			
			
	def _compute_gm_imap_unordered(self):
		self.__add_dummy_labels(self._graphs)
		
		# get all paths of all graphs before calculating kernels to save time,
		# but this may cost a lot of memory for large datasets.
		pool = Pool(self._n_jobs)
		itr = zip(self._graphs, range(0, len(self._graphs)))
		if len(self._graphs) < 100 * self._n_jobs:
			chunksize = int(len(self._graphs) / self._n_jobs) + 1
		else:
			chunksize = 100
		all_paths = [[] for _ in range(len(self._graphs))]
		if self.__compute_method == 'trie' and self.__k_func is not None:
			get_ps_fun = self._wrapper_find_all_path_as_trie
		elif self.__compute_method != 'trie' and self.__k_func is not None:  
			get_ps_fun = partial(self._wrapper_find_all_paths_until_length, True)  
		else: 
			get_ps_fun = partial(self._wrapper_find_all_paths_until_length, False)
		if self._verbose >= 2:
			iterator = tqdm(pool.imap_unordered(get_ps_fun, itr, chunksize),
							desc='getting paths', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(get_ps_fun, itr, chunksize)
		for i, ps in iterator:
			all_paths[i] = ps
		pool.close()
		pool.join()
		
		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))
	 
		if self.__compute_method == 'trie' and self.__k_func is not None:
			def init_worker(trie_toshare):
				global G_trie
				G_trie = trie_toshare
			do_fun = self._wrapper_kernel_do_trie
		elif self.__compute_method != 'trie' and self.__k_func is not None:
			def init_worker(plist_toshare):
				global G_plist
				G_plist = plist_toshare
			do_fun = self._wrapper_kernel_do_naive   
		else:
			def init_worker(plist_toshare):
				global G_plist
				G_plist = plist_toshare
			do_fun = self.__wrapper_kernel_do_kernelless # @todo: what is this?  
		parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=init_worker, 
					glbv=(all_paths,), n_jobs=self._n_jobs, verbose=self._verbose) 	
			
		return gram_matrix
	
	
	def _compute_kernel_list_series(self, g1, g_list):
		self.__add_dummy_labels(g_list + [g1])
		
		if self._verbose >= 2:
			iterator_ps = tqdm(g_list, desc='getting paths', file=sys.stdout)
			iterator_kernel = tqdm(range(len(g_list)), desc='calculating kernels', file=sys.stdout)
		else:
			iterator_ps = g_list
			iterator_kernel = range(len(g_list))
			
		kernel_list = [None] * len(g_list)

		if self.__compute_method == 'trie':
			paths_g1 = self.__find_all_path_as_trie(g1)
			paths_g_list = [self.__find_all_path_as_trie(g) for g in iterator_ps]
			for i in iterator_kernel:
				kernel = self.__kernel_do_trie(paths_g1, paths_g_list[i])
				kernel_list[i] = kernel
		else:
			paths_g1 = self.__find_all_paths_until_length(g1)
			paths_g_list = [self.__find_all_paths_until_length(g) for g in iterator_ps]
			for i in iterator_kernel:
				kernel = self.__kernel_do_naive(paths_g1, paths_g_list[i])
				kernel_list[i] = kernel
				
		return kernel_list
	
	
	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self.__add_dummy_labels(g_list + [g1])
		
		# get all paths of all graphs before calculating kernels to save time,
		# but this may cost a lot of memory for large datasets.
		pool = Pool(self._n_jobs)
		itr = zip(g_list, range(0, len(g_list)))
		if len(g_list) < 100 * self._n_jobs:
			chunksize = int(len(g_list) / self._n_jobs) + 1
		else:
			chunksize = 100
		paths_g_list = [[] for _ in range(len(g_list))]
		if self.__compute_method == 'trie' and self.__k_func is not None:
			paths_g1 = self.__find_all_path_as_trie(g1)
			get_ps_fun = self._wrapper_find_all_path_as_trie
		elif self.__compute_method != 'trie' and self.__k_func is not None:
			paths_g1 = self.__find_all_paths_until_length(g1) 
			get_ps_fun = partial(self._wrapper_find_all_paths_until_length, True)  
		else:
			paths_g1 = self.__find_all_paths_until_length(g1)  
			get_ps_fun = partial(self._wrapper_find_all_paths_until_length, False)
		if self._verbose >= 2:
			iterator = tqdm(pool.imap_unordered(get_ps_fun, itr, chunksize),
							desc='getting paths', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(get_ps_fun, itr, chunksize)
		for i, ps in iterator:
			paths_g_list[i] = ps
		pool.close()
		pool.join()
		
		# compute kernel list.
		kernel_list = [None] * len(g_list)
		
		def init_worker(p1_toshare, plist_toshare):
			global G_p1, G_plist
			G_p1 = p1_toshare
			G_plist = plist_toshare
		do_fun = self._wrapper_kernel_list_do
		def func_assign(result, var_to_assign):	
			var_to_assign[result[0]] = result[1]
		itr = range(len(g_list))
		len_itr = len(g_list)
		parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(paths_g1, paths_g_list), method='imap_unordered', n_jobs=self._n_jobs, itr_desc='calculating kernels', verbose=self._verbose)
			 			
		return kernel_list
	
	
	def _wrapper_kernel_list_do(self, itr):
		if self.__compute_method == 'trie' and self.__k_func is not None:
			return itr, self.__kernel_do_trie(G_p1, G_plist[itr])
		elif self.__compute_method != 'trie' and self.__k_func is not None:
			return itr, self.__kernel_do_naive(G_p1, G_plist[itr])  
		else:
			return itr, self.__kernel_do_kernelless(G_p1, G_plist[itr])
	
	
	def _compute_single_kernel_series(self, g1, g2):
		self.__add_dummy_labels([g1] + [g2])
		if self.__compute_method == 'trie':
			paths_g1 = self.__find_all_path_as_trie(g1)
			paths_g2 = self.__find_all_path_as_trie(g2)
			kernel = self.__kernel_do_trie(paths_g1, paths_g2)
		else:
			paths_g1 = self.__find_all_paths_until_length(g1)
			paths_g2 = self.__find_all_paths_until_length(g2)
			kernel = self.__kernel_do_naive(paths_g1, paths_g2)
		return kernel			

	
	def __kernel_do_trie(self, trie1, trie2):
		"""Calculate path graph kernels up to depth d between 2 graphs using trie.
	
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
		if self.__k_func == 'tanimoto':	  
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
			
		elif self.__k_func == 'MinMax': # MinMax kernel		  
			# traverse all paths in graph1 and search them in graph2. Deep-first 
			# search is applied.
			def traverseTrie1m(root, trie2, sumlist, pcurrent=[]):
				for key, node in root['children'].items():
					pcurrent.append(key)
					if node['isEndOfWord']:
# 						print(node['count'])
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
# 			print(trie1.root)
# 			print(trie2.root)
			traverseTrie1m(trie1.root, trie2, sumlist)
# 			print(sumlist)
			traverseTrie2m(trie2.root, trie1, sumlist)
# 			print(sumlist)
			kernel = sumlist[0] / sumlist[1]
		else:
			raise Exception('The given "k_func" cannot be recognized. Possible choices include: "tanimoto", "MinMax".')
	
		return kernel
	
	
	def _wrapper_kernel_do_trie(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self.__kernel_do_trie(G_trie[i], G_trie[j])
	
	
	def __kernel_do_naive(self, paths1, paths2):
		"""Calculate path graph kernels up to depth d between 2 graphs naively.
	
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
	
		if self.__k_func == 'tanimoto':
			length_union = len(set(paths1 + paths2))
			kernel = (len(set(paths1)) + len(set(paths2)) -
					  length_union) / length_union
	#		vector1 = [(1 if path in paths1 else 0) for path in all_paths]
	#		vector2 = [(1 if path in paths2 else 0) for path in all_paths]
	#		kernel_uv = np.dot(vector1, vector2)
	#		kernel = kernel_uv / (len(set(paths1)) + len(set(paths2)) - kernel_uv)
	
		elif self.__k_func == 'MinMax':  # MinMax kernel
			path_count1 = Counter(paths1)
			path_count2 = Counter(paths2)
			vector1 = [(path_count1[key] if (key in path_count1.keys()) else 0)
					   for key in all_paths]
			vector2 = [(path_count2[key] if (key in path_count2.keys()) else 0)
					   for key in all_paths]
			kernel = np.sum(np.minimum(vector1, vector2)) / \
				np.sum(np.maximum(vector1, vector2))
		else:
			raise Exception('The given "k_func" cannot be recognized. Possible choices include: "tanimoto", "MinMax".')
	
		return kernel
	
	
	def _wrapper_kernel_do_naive(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self.__kernel_do_naive(G_plist[i], G_plist[j])
	
	
	def __find_all_path_as_trie(self, G):
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
		def traverseGraph(root, ptrie, G, pcurrent=[]):
			if len(pcurrent) < self.__depth + 1:
				for neighbor in G[root]:
					if neighbor not in pcurrent:
						pcurrent.append(neighbor)
						plstr = self.__paths2labelseqs([pcurrent], G)
						ptrie.insertWord(plstr[0])
						traverseGraph(neighbor, ptrie, G, pcurrent)
			del pcurrent[-1]
	
	
		ptrie = Trie()
		path_l = [[n] for n in G.nodes]  # paths of length l
		path_l_str = self.__paths2labelseqs(path_l, G)
		for p in path_l_str:
			ptrie.insertWord(p)
		for n in G.nodes:
			traverseGraph(n, ptrie, G, pcurrent=[n])
			
			
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
	
	
	def _wrapper_find_all_path_as_trie(self, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, self.__find_all_path_as_trie(g)
	
	
	# @todo: (can be removed maybe)  this method find paths repetively, it could be faster.
	def __find_all_paths_until_length(self, G, tolabelseqs=True):
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
		# for l in range(1, self.__depth + 1):
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
		for l in range(1, self.__depth + 1):
			path_lplus1 = []
			for path in path_l:
				for neighbor in G[path[-1]]:
					if neighbor not in path:
						tmp = path + [neighbor]
	#					if tmp[::-1] not in path_lplus1:
						path_lplus1.append(tmp)
	
			all_paths += path_lplus1
			path_l = [p.copy() for p in path_lplus1]
	
		# for i in range(0, self.__depth + 1):
		#	 new_paths = find_all_paths(G, i)
		#	 if new_paths == []:
		#		 break
		#	 all_paths.extend(new_paths)
	
		# consider labels
	#	print(paths2labelseqs(all_paths, G, ds_attrs, node_label, edge_label))
	#	print()
		return (self.__paths2labelseqs(all_paths, G) if tolabelseqs else all_paths)
			
			
	def _wrapper_find_all_paths_until_length(self, tolabelseqs, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, self.__find_all_paths_until_length(g, tolabelseqs=tolabelseqs)
	
	
	def __paths2labelseqs(self, plist, G):
		if len(self.__node_labels) > 0:
			if len(self.__edge_labels) > 0:
				path_strs = []
				for path in plist:
					pths_tmp = []
					for idx, node in enumerate(path[:-1]):
						pths_tmp.append(tuple(G.nodes[node][nl] for nl in self.__node_labels))
						pths_tmp.append(tuple(G[node][path[idx + 1]][el] for el in self.__edge_labels))
					pths_tmp.append(tuple(G.nodes[path[-1]][nl] for nl in self.__node_labels))
					path_strs.append(tuple(pths_tmp))
			else:
				path_strs = []
				for path in plist:
					pths_tmp = []
					for node in path:
						pths_tmp.append(tuple(G.nodes[node][nl] for nl in self.__node_labels))
					path_strs.append(tuple(pths_tmp))
			return path_strs
		else:
			if len(self.__edge_labels) > 0:
				path_strs = []
				for path in plist:
					if len(path) == 1:
						path_strs.append(tuple())
					else:
						pths_tmp = []
						for idx, node in enumerate(path[:-1]):
							pths_tmp.append(tuple(G[node][path[idx + 1]][el] for el in self.__edge_labels))
						path_strs.append(tuple(pths_tmp))
				return path_strs
			else:
				return [tuple(['0' for node in path]) for path in plist]
	#			return [tuple([len(path)]) for path in all_paths]
	
	
	def __add_dummy_labels(self, Gn):
		if self.__k_func is not None:
			if len(self.__node_labels) == 0 or (len(self.__node_labels) == 1 and self.__node_labels[0] == SpecialLabel.DUMMY):
				for i in range(len(Gn)):
					nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
				self.__node_labels = [SpecialLabel.DUMMY]
			if len(self.__edge_labels) == 0 or (len(self.__edge_labels) == 1 and self.__edge_labels[0] == SpecialLabel.DUMMY):
				for i in range(len(Gn)):
					nx.set_edge_attributes(Gn[i], '0', SpecialLabel.DUMMY)
				self.__edge_labels = [SpecialLabel.DUMMY]