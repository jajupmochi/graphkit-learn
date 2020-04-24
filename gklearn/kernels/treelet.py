#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:02:46 2020

@author: ljia

@references: 

    [1] Gaüzère B, Brun L, Villemin D. Two new graphs kernels in 
    chemoinformatics. Pattern Recognition Letters. 2012 Nov 1;33(15):2038-47.
"""

import sys
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import networkx as nx
from collections import Counter
from itertools import chain
from gklearn.utils import SpecialLabel
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.utils.utils import find_all_paths, get_mlti_dim_node_attrs
from gklearn.kernels import GraphKernel


class Treelet(GraphKernel):
	
	def __init__(self, **kwargs):
		GraphKernel.__init__(self)
		self.__node_labels = kwargs.get('node_labels', [])
		self.__edge_labels = kwargs.get('edge_labels', [])
		self.__sub_kernel = kwargs.get('sub_kernel', None)
		self.__ds_infos = kwargs.get('ds_infos', {})
		if self.__sub_kernel is None:
			raise Exception('Sub kernel not set.')


	def _compute_gm_series(self):
		self.__add_dummy_labels(self._graphs)
		
		# get all canonical keys of all graphs before calculating kernels to save 
		# time, but this may cost a lot of memory for large dataset.
		canonkeys = []
		if self._verbose >= 2:
			iterator = tqdm(self._graphs, desc='getting canonkeys', file=sys.stdout)
		else:
			iterator = self._graphs
		for g in iterator:
			canonkeys.append(self.__get_canonkeys(g))
		
		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))
		
		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(self._graphs)), 2)
		if self._verbose >= 2:
			iterator = tqdm(itr, desc='calculating kernels', file=sys.stdout)
		else:
			iterator = itr
		for i, j in iterator:
			kernel = self.__kernel_do(canonkeys[i], canonkeys[j])
			gram_matrix[i][j] = kernel
			gram_matrix[j][i] = kernel # @todo: no directed graph considered?
				
		return gram_matrix
			
			
	def _compute_gm_imap_unordered(self):
		self.__add_dummy_labels(self._graphs)
		
		# get all canonical keys of all graphs before calculating kernels to save 
		# time, but this may cost a lot of memory for large dataset.
		pool = Pool(self._n_jobs)
		itr = zip(self._graphs, range(0, len(self._graphs)))
		if len(self._graphs) < 100 * self._n_jobs:
			chunksize = int(len(self._graphs) / self._n_jobs) + 1
		else:
			chunksize = 100
		canonkeys = [[] for _ in range(len(self._graphs))]
		get_fun = self._wrapper_get_canonkeys
		if self._verbose >= 2:
			iterator = tqdm(pool.imap_unordered(get_fun, itr, chunksize),
							desc='getting canonkeys', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(get_fun, itr, chunksize)
		for i, ck in iterator:
			canonkeys[i] = ck
		pool.close()
		pool.join()
		
		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))
		
		def init_worker(canonkeys_toshare):
			global G_canonkeys
			G_canonkeys = canonkeys_toshare
		do_fun = self._wrapper_kernel_do
		parallel_gm(do_fun, gram_matrix, self._graphs, init_worker=init_worker, 
					glbv=(canonkeys,), n_jobs=self._n_jobs, verbose=self._verbose)
			
		return gram_matrix
	
	
	def _compute_kernel_list_series(self, g1, g_list):
		self.__add_dummy_labels(g_list + [g1])
		
		# get all canonical keys of all graphs before calculating kernels to save 
		# time, but this may cost a lot of memory for large dataset.
		canonkeys_1 = self.__get_canonkeys(g1)
		canonkeys_list = []
		if self._verbose >= 2:
			iterator = tqdm(g_list, desc='getting canonkeys', file=sys.stdout)
		else:
			iterator = g_list
		for g in iterator:
			canonkeys_list.append(self.__get_canonkeys(g))
				
		# compute kernel list.
		kernel_list = [None] * len(g_list)
		if self._verbose >= 2:
			iterator = tqdm(range(len(g_list)), desc='calculating kernels', file=sys.stdout)
		else:
			iterator = range(len(g_list))
		for i in iterator:
			kernel = self.__kernel_do(canonkeys_1, canonkeys_list[i])
			kernel_list[i] = kernel
				
		return kernel_list
	
	
	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self.__add_dummy_labels(g_list + [g1])
		
		# get all canonical keys of all graphs before calculating kernels to save 
		# time, but this may cost a lot of memory for large dataset.
		canonkeys_1 = self.__get_canonkeys(g1)
		canonkeys_list = [[] for _ in range(len(g_list))]
		pool = Pool(self._n_jobs)
		itr = zip(g_list, range(0, len(g_list)))
		if len(g_list) < 100 * self._n_jobs:
			chunksize = int(len(g_list) / self._n_jobs) + 1
		else:
			chunksize = 100
		get_fun = self._wrapper_get_canonkeys
		if self._verbose >= 2:
			iterator = tqdm(pool.imap_unordered(get_fun, itr, chunksize),
							desc='getting canonkeys', file=sys.stdout)
		else:
			iterator = pool.imap_unordered(get_fun, itr, chunksize)
		for i, ck in iterator:
			canonkeys_list[i] = ck
		pool.close()
		pool.join()
		
		# compute kernel list.
		kernel_list = [None] * len(g_list)

		def init_worker(ck_1_toshare, ck_list_toshare):
			global G_ck_1, G_ck_list
			G_ck_1 = ck_1_toshare
			G_ck_list = ck_list_toshare
		do_fun = self._wrapper_kernel_list_do
		def func_assign(result, var_to_assign):	
			var_to_assign[result[0]] = result[1]
		itr = range(len(g_list))
		len_itr = len(g_list)
		parallel_me(do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(canonkeys_1, canonkeys_list), method='imap_unordered', 
			n_jobs=self._n_jobs, itr_desc='calculating kernels', verbose=self._verbose)
			
		return kernel_list
	
	
	def _wrapper_kernel_list_do(self, itr):
		return itr, self.__kernel_do(G_ck_1, G_ck_list[itr])
	
	
	def _compute_single_kernel_series(self, g1, g2):
		self.__add_dummy_labels([g1] + [g2])
		canonkeys_1 = self.__get_canonkeys(g1)
		canonkeys_2 = self.__get_canonkeys(g2)
		kernel = self.__kernel_do(canonkeys_1, canonkeys_2)
		return kernel			
	
	
	def __kernel_do(self, canonkey1, canonkey2):
		"""Calculate treelet graph kernel between 2 graphs.
		
		Parameters
		----------
		canonkey1, canonkey2 : list
			List of canonical keys in 2 graphs, where each key is represented by a string.
			
		Return
		------
		kernel : float
			Treelet Kernel between 2 graphs.
		"""
		keys = set(canonkey1.keys()) & set(canonkey2.keys()) # find same canonical keys in both graphs
		vector1 = np.array([(canonkey1[key] if (key in canonkey1.keys()) else 0) for key in keys])
		vector2 = np.array([(canonkey2[key] if (key in canonkey2.keys()) else 0) for key in keys]) 
		kernel = self.__sub_kernel(vector1, vector2) 
		return kernel
	
	
	def _wrapper_kernel_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self.__kernel_do(G_canonkeys[i], G_canonkeys[j])
	
	
	def __get_canonkeys(self, G):
		"""Generate canonical keys of all treelets in a graph.
		
		Parameters
		----------
		G : NetworkX graphs
			The graph in which keys are generated.
			
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
		patterns['0'] = list(G.nodes())
		canonkey['0'] = nx.number_of_nodes(G)
		for i in range(1, 6): # for i in range(1, 6):
			patterns[str(i)] = find_all_paths(G, i, self.__ds_infos['directed'])
			canonkey[str(i)] = len(patterns[str(i)])
	
		# n-star patterns
		patterns['3star'] = [[node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 3]
		patterns['4star'] = [[node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 4]
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
							new_pattern = pattern_t + [neighborx]
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
				for neighbor2 in G[pattern[0]] if G.degree(neighbor2) >= 2 if neighbor1 > neighbor2]:
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
		if len(self.__node_labels) > 0 or len(self.__edge_labels) > 0:
			canonkey_l = {} # canonical key, a dictionary which keeps track of amount of every treelet.
	
			# linear patterns
			canonkey_t = Counter(get_mlti_dim_node_attrs(G, self.__node_labels))
			for key in canonkey_t:
				canonkey_l[('0', key)] = canonkey_t[key]
	
			for i in range(1, 6): # for i in range(1, 6):
				treelet = []
				for pattern in patterns[str(i)]:
					canonlist = []
					for idx, node in enumerate(pattern[:-1]):
						canonlist.append(tuple(G.nodes[node][nl] for nl in self.__node_labels))
						canonlist.append(tuple(G[node][pattern[idx+1]][el] for el in self.__edge_labels))
					canonlist.append(tuple(G.nodes[pattern[-1]][nl] for nl in self.__node_labels))
					canonkey_t = canonlist if canonlist < canonlist[::-1] else canonlist[::-1]
					treelet.append(tuple([str(i)] + canonkey_t))
				canonkey_l.update(Counter(treelet))
	
			# n-star patterns
			for i in range(3, 6):
				treelet = []
				for pattern in patterns[str(i) + 'star']:
					canonlist = []
					for leaf in pattern[1:]:
						nlabels = tuple(G.nodes[leaf][nl] for nl in self.__node_labels)
						elabels = tuple(G[leaf][pattern[0]][el] for el in self.__edge_labels)
						canonlist.append(tuple((nlabels, elabels)))
					canonlist.sort()
					canonlist = list(chain.from_iterable(canonlist))
					canonkey_t = tuple(['d' if i == 5 else str(i * 2)] + 
									   [tuple(G.nodes[pattern[0]][nl] for nl in self.__node_labels)] 
									   + canonlist)
					treelet.append(canonkey_t)
				canonkey_l.update(Counter(treelet))
	
			# pattern 7
			treelet = []
			for pattern in patterns['7']:
				canonlist = []
				for leaf in pattern[1:3]:
					nlabels = tuple(G.nodes[leaf][nl] for nl in self.__node_labels)
					elabels = tuple(G[leaf][pattern[0]][el] for el in self.__edge_labels)
					canonlist.append(tuple((nlabels, elabels)))
				canonlist.sort()
				canonlist = list(chain.from_iterable(canonlist))
				canonkey_t = tuple(['7'] 
					   + [tuple(G.nodes[pattern[0]][nl] for nl in self.__node_labels)] + canonlist 
					   + [tuple(G.nodes[pattern[3]][nl] for nl in self.__node_labels)] 
					   + [tuple(G[pattern[3]][pattern[0]][el] for el in self.__edge_labels)]
					   + [tuple(G.nodes[pattern[4]][nl] for nl in self.__node_labels)] 
					   + [tuple(G[pattern[4]][pattern[3]][el] for el in self.__edge_labels)])
				treelet.append(canonkey_t)
			canonkey_l.update(Counter(treelet))
	
			# pattern 11
			treelet = []
			for pattern in patterns['11']:
				canonlist = []
				for leaf in pattern[1:4]:
					nlabels = tuple(G.nodes[leaf][nl] for nl in self.__node_labels)
					elabels = tuple(G[leaf][pattern[0]][el] for el in self.__edge_labels)
					canonlist.append(tuple((nlabels, elabels)))
				canonlist.sort()
				canonlist = list(chain.from_iterable(canonlist))
				canonkey_t = tuple(['b'] 
					   + [tuple(G.nodes[pattern[0]][nl] for nl in self.__node_labels)] + canonlist 
					   + [tuple(G.nodes[pattern[4]][nl] for nl in self.__node_labels)] 
					   + [tuple(G[pattern[4]][pattern[0]][el] for el in self.__edge_labels)]
					   + [tuple(G.nodes[pattern[5]][nl] for nl in self.__node_labels)] 
					   + [tuple(G[pattern[5]][pattern[4]][el] for el in self.__edge_labels)])
				treelet.append(canonkey_t)
			canonkey_l.update(Counter(treelet))
	
			# pattern 10
			treelet = []
			for pattern in patterns['10']:
				canonkey4 = [tuple(G.nodes[pattern[5]][nl] for nl in self.__node_labels),
				 tuple(G[pattern[5]][pattern[4]][el] for el in self.__edge_labels)]
				canonlist = []
				for leaf in pattern[1:3]:
					nlabels = tuple(G.nodes[leaf][nl] for nl in self.__node_labels)
					elabels = tuple(G[leaf][pattern[0]][el] for el in self.__edge_labels)
					canonlist.append(tuple((nlabels, elabels)))
				canonlist.sort()
				canonkey0 = list(chain.from_iterable(canonlist))
				canonkey_t = tuple(['a']
					    + [tuple(G.nodes[pattern[3]][nl] for nl in self.__node_labels)] 
						+ [tuple(G.nodes[pattern[4]][nl] for nl in self.__node_labels)] 
						+ [tuple(G[pattern[4]][pattern[3]][el] for el in self.__edge_labels)] 
						+ [tuple(G.nodes[pattern[0]][nl] for nl in self.__node_labels)] 
						+ [tuple(G[pattern[0]][pattern[3]][el] for el in self.__edge_labels)] 
						+ canonkey4 + canonkey0)
				treelet.append(canonkey_t)
			canonkey_l.update(Counter(treelet))
	
			# pattern 12
			treelet = []
			for pattern in patterns['12']:
				canonlist0 = []
				for leaf in pattern[1:3]:
					nlabels = tuple(G.nodes[leaf][nl] for nl in self.__node_labels)
					elabels = tuple(G[leaf][pattern[0]][el] for el in self.__edge_labels)
					canonlist0.append(tuple((nlabels, elabels)))
				canonlist0.sort()
				canonlist0 = list(chain.from_iterable(canonlist0))
				canonlist3 = []
				for leaf in pattern[4:6]:
					nlabels = tuple(G.nodes[leaf][nl] for nl in self.__node_labels)
					elabels = tuple(G[leaf][pattern[3]][el] for el in self.__edge_labels)
					canonlist3.append(tuple((nlabels, elabels)))
				canonlist3.sort()
				canonlist3 = list(chain.from_iterable(canonlist3))
				
				# 2 possible key can be generated from 2 nodes with extended label 3, 
				# select the one with lower lexicographic order.
				canonkey_t1 = tuple(['c'] 
						+ [tuple(G.nodes[pattern[0]][nl] for nl in self.__node_labels)] + canonlist0 
						+ [tuple(G.nodes[pattern[3]][nl] for nl in self.__node_labels)] 
						+ [tuple(G[pattern[3]][pattern[0]][el] for el in self.__edge_labels)] 
						+ canonlist3)
				canonkey_t2 = tuple(['c'] 
						+ [tuple(G.nodes[pattern[3]][nl] for nl in self.__node_labels)] + canonlist3 
						+ [tuple(G.nodes[pattern[0]][nl] for nl in self.__node_labels)] 
						+ [tuple(G[pattern[0]][pattern[3]][el] for el in self.__edge_labels)] 
						+ canonlist0)
				treelet.append(canonkey_t1 if canonkey_t1 < canonkey_t2 else canonkey_t2)
			canonkey_l.update(Counter(treelet))
	
			# pattern 9
			treelet = []
			for pattern in patterns['9']:
				canonkey2 = [tuple(G.nodes[pattern[4]][nl] for nl in self.__node_labels),
				  tuple(G[pattern[4]][pattern[2]][el] for el in self.__edge_labels)]
				canonkey3 = [tuple(G.nodes[pattern[5]][nl] for nl in self.__node_labels),
				  tuple(G[pattern[5]][pattern[3]][el] for el in self.__edge_labels)]
				prekey2 = [tuple(G.nodes[pattern[2]][nl] for nl in self.__node_labels),
			      tuple(G[pattern[2]][pattern[0]][el] for el in self.__edge_labels)]
				prekey3 = [tuple(G.nodes[pattern[3]][nl] for nl in self.__node_labels), 
			      tuple(G[pattern[3]][pattern[0]][el] for el in self.__edge_labels)]
				if prekey2 + canonkey2 < prekey3 + canonkey3:
					canonkey_t = [tuple(G.nodes[pattern[1]][nl] for nl in self.__node_labels)] \
								 + [tuple(G[pattern[1]][pattern[0]][el] for el in self.__edge_labels)] \
								 + prekey2 + prekey3 + canonkey2 + canonkey3
				else:
					canonkey_t = [tuple(G.nodes[pattern[1]][nl] for nl in self.__node_labels)] \
								 + [tuple(G[pattern[1]][pattern[0]][el] for el in self.__edge_labels)] \
								 + prekey3 + prekey2 + canonkey3 + canonkey2
				treelet.append(tuple(['9']
						  + [tuple(G.nodes[pattern[0]][nl] for nl in self.__node_labels)] 
						  + canonkey_t))
			canonkey_l.update(Counter(treelet))
	
			return canonkey_l
	
		return canonkey
	
	
	def _wrapper_get_canonkeys(self, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, self.__get_canonkeys(g)
	
	
	def __add_dummy_labels(self, Gn):
		if len(self.__node_labels) == 0 or (len(self.__node_labels) == 1 and self.__node_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self.__node_labels = [SpecialLabel.DUMMY]
		if len(self.__edge_labels) == 0 or (len(self.__edge_labels) == 1 and self.__edge_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_edge_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self.__edge_labels = [SpecialLabel.DUMMY]