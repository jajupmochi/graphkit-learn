#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:16:34 2020

@author: ljia

@references:

	[1] Shervashidze N, Schweitzer P, Leeuwen EJ, Mehlhorn K, Borgwardt KM. 
	Weisfeiler-lehman graph kernels. Journal of Machine Learning Research. 
	2011;12(Sep):2539-61.
"""

import numpy as np
import networkx as nx
from collections import Counter
from functools import partial
from gklearn.utils import SpecialLabel
from gklearn.utils.parallel import parallel_gm
from gklearn.kernels import GraphKernel


class WeisfeilerLehman(GraphKernel): # @todo: total parallelization and sp, edge user kernel.
	
	def __init__(self, **kwargs):
		GraphKernel.__init__(self)
		self.__node_labels = kwargs.get('node_labels', [])
		self.__edge_labels = kwargs.get('edge_labels', [])
		self.__height = int(kwargs.get('height', 0))
		self.__base_kernel = kwargs.get('base_kernel', 'subtree')
		self.__ds_infos = kwargs.get('ds_infos', {})


	def _compute_gm_series(self):
		if self._verbose >= 2:
			import warnings
			warnings.warn('A part of the computation is parallelized.')
			
		self.__add_dummy_node_labels(self._graphs)
		
		# for WL subtree kernel
		if self.__base_kernel == 'subtree':		   
			gram_matrix = self.__subtree_kernel_do(self._graphs)
	
		# for WL shortest path kernel
		elif self.__base_kernel == 'sp':
			gram_matrix = self.__sp_kernel_do(self._graphs)
	
		# for WL edge kernel
		elif self.__base_kernel == 'edge':
			gram_matrix = self.__edge_kernel_do(self._graphs)
	
		# for user defined base kernel
		else:
			gram_matrix = self.__user_kernel_do(self._graphs)
				
		return gram_matrix
			
			
	def _compute_gm_imap_unordered(self):
		if self._verbose >= 2:
			import warnings
			warnings.warn('Only a part of the computation is parallelized due to the structure of this kernel.')
		return self._compute_gm_series()
	
	
	def _compute_kernel_list_series(self, g1, g_list): # @todo: this should be better.
		if self._verbose >= 2:
			import warnings
			warnings.warn('A part of the computation is parallelized.')
			
		self.__add_dummy_node_labels(g_list + [g1])
				
		# for WL subtree kernel
		if self.__base_kernel == 'subtree':		   
			gram_matrix = self.__subtree_kernel_do(g_list + [g1])
	
		# for WL shortest path kernel
		elif self.__base_kernel == 'sp':
			gram_matrix = self.__sp_kernel_do(g_list + [g1])
	
		# for WL edge kernel
		elif self.__base_kernel == 'edge':
			gram_matrix = self.__edge_kernel_do(g_list + [g1])
	
		# for user defined base kernel
		else:
			gram_matrix = self.__user_kernel_do(g_list + [g1])
				
		return list(gram_matrix[-1][0:-1])
	
	
	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		if self._verbose >= 2:
			import warnings
			warnings.warn('Only a part of the computation is parallelized due to the structure of this kernel.')
		return self._compute_kernel_list_series(g1, g_list)
	
	
	def _wrapper_kernel_list_do(self, itr):
		pass
	
	
	def _compute_single_kernel_series(self, g1, g2):  # @todo: this should be better.
		self.__add_dummy_node_labels([g1] + [g2])

		# for WL subtree kernel
		if self.__base_kernel == 'subtree':		   
			gram_matrix = self.__subtree_kernel_do([g1] + [g2])
	
		# for WL shortest path kernel
		elif self.__base_kernel == 'sp':
			gram_matrix = self.__sp_kernel_do([g1] + [g2])
	
		# for WL edge kernel
		elif self.__base_kernel == 'edge':
			gram_matrix = self.__edge_kernel_do([g1] + [g2])
	
		# for user defined base kernel
		else:
			gram_matrix = self.__user_kernel_do([g1] + [g2])
				
		return gram_matrix[0][1]
	
	
	def __subtree_kernel_do(self, Gn):
		"""Calculate Weisfeiler-Lehman kernels between graphs.
	
		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are calculated.	   
	
		Return
		------
		gram_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		gram_matrix = np.zeros((len(Gn), len(Gn)))
	
		# initial for height = 0
		all_num_of_each_label = [] # number of occurence of each label in each graph in this iteration
	
		# for each graph
		for G in Gn:
			# set all labels into a tuple.
			for nd, attrs in G.nodes(data=True): # @todo: there may be a better way.
				G.nodes[nd]['label_tuple'] = tuple(attrs[name] for name in self.__node_labels)
			# get the set of original labels
			labels_ori = list(nx.get_node_attributes(G, 'label_tuple').values())
			# number of occurence of each label in G
			all_num_of_each_label.append(dict(Counter(labels_ori)))
	
		# calculate subtree kernel with the 0th iteration and add it to the final kernel.
		self.__compute_gram_matrix(gram_matrix, all_num_of_each_label, Gn)
	
		# iterate each height
		for h in range(1, self.__height + 1):
			all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
			num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
	#		all_labels_ori = set() # all unique orignal labels in all graphs in this iteration
			all_num_of_each_label = [] # number of occurence of each label in G
	
			# @todo: parallel this part.
			for idx, G in enumerate(Gn):
	
				all_multisets = []
				for node, attrs in G.nodes(data=True):
					# Multiset-label determination.
					multiset = [G.nodes[neighbors]['label_tuple'] for neighbors in G[node]]
					# sorting each multiset
					multiset.sort()
					multiset = [attrs['label_tuple']] + multiset # add the prefix 
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
					G.nodes[node]['label_tuple'] = set_compressed[all_multisets[idx]]
	
				# get the set of compressed labels
				labels_comp = list(nx.get_node_attributes(G, 'label_tuple').values())
	#			all_labels_ori.update(labels_comp)
				all_num_of_each_label.append(dict(Counter(labels_comp)))
	
			# calculate subtree kernel with h iterations and add it to the final kernel
			self.__compute_gram_matrix(gram_matrix, all_num_of_each_label, Gn)
	
		return gram_matrix

	
	def __compute_gram_matrix(self, gram_matrix, all_num_of_each_label, Gn):
		"""Compute Gram matrix using the base kernel.
		"""
		if self._parallel == 'imap_unordered':
			# compute kernels.
			def init_worker(alllabels_toshare):
				global G_alllabels
				G_alllabels = alllabels_toshare
			do_partial = partial(self._wrapper_compute_subtree_kernel, gram_matrix)
			parallel_gm(do_partial, gram_matrix, Gn, init_worker=init_worker, 
						glbv=(all_num_of_each_label,), n_jobs=self._n_jobs, verbose=self._verbose)
		elif self._parallel is None:
			for i in range(len(gram_matrix)):
				for j in range(i, len(gram_matrix)):
					gram_matrix[i][j] = self.__compute_subtree_kernel(all_num_of_each_label[i],
						   all_num_of_each_label[j], gram_matrix[i][j])
					gram_matrix[j][i] = gram_matrix[i][j]
					
					
	def __compute_subtree_kernel(self, num_of_each_label1, num_of_each_label2, kernel):
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
	
	
	def _wrapper_compute_subtree_kernel(self, gram_matrix, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self.__compute_subtree_kernel(G_alllabels[i], G_alllabels[j], gram_matrix[i][j])
				
	
	def _wl_spkernel_do(Gn, node_label, edge_label, height):
		"""Calculate Weisfeiler-Lehman shortest path kernels between graphs.
		
		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are calculated.	   
		node_label : string
			node attribute used as label.	  
		edge_label : string
			edge attribute used as label.	   
		height : int
			subtree height.
			
		Return
		------
		gram_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""
		pass
		from gklearn.utils.utils import getSPGraph
		  
		# init.
		height = int(height)
		gram_matrix = np.zeros((len(Gn), len(Gn))) # init kernel
	
		Gn = [ getSPGraph(G, edge_weight = edge_label) for G in Gn ] # get shortest path graphs of Gn
		
		# initial for height = 0
		for i in range(0, len(Gn)):
			for j in range(i, len(Gn)):
				for e1 in Gn[i].edges(data = True):
					for e2 in Gn[j].edges(data = True):		  
						if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
							gram_matrix[i][j] += 1
				gram_matrix[j][i] = gram_matrix[i][j]
				
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
					
			# calculate subtree kernel with h iterations and add it to the final kernel
			for i in range(0, len(Gn)):
				for j in range(i, len(Gn)):
					for e1 in Gn[i].edges(data = True):
						for e2 in Gn[j].edges(data = True):		  
							if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
								gram_matrix[i][j] += 1
					gram_matrix[j][i] = gram_matrix[i][j]
			
		return gram_matrix
	
	
	
	def _wl_edgekernel_do(Gn, node_label, edge_label, height):
		"""Calculate Weisfeiler-Lehman edge kernels between graphs.
		
		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are calculated.	   
		node_label : string
			node attribute used as label.	  
		edge_label : string
			edge attribute used as label.	   
		height : int
			subtree height.
			
		Return
		------
		gram_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""	  
		pass
		# init.
		height = int(height)
		gram_matrix = np.zeros((len(Gn), len(Gn))) # init kernel
	  
		# initial for height = 0
		for i in range(0, len(Gn)):
			for j in range(i, len(Gn)):
				for e1 in Gn[i].edges(data = True):
					for e2 in Gn[j].edges(data = True):		  
						if e1[2][edge_label] == e2[2][edge_label] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
							gram_matrix[i][j] += 1
				gram_matrix[j][i] = gram_matrix[i][j]
				
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
					
			# calculate subtree kernel with h iterations and add it to the final kernel
			for i in range(0, len(Gn)):
				for j in range(i, len(Gn)):
					for e1 in Gn[i].edges(data = True):
						for e2 in Gn[j].edges(data = True):		  
							if e1[2][edge_label] == e2[2][edge_label] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
								gram_matrix[i][j] += 1
					gram_matrix[j][i] = gram_matrix[i][j]
			
		return gram_matrix
	
	
	def _wl_userkernel_do(Gn, node_label, edge_label, height, base_kernel):
		"""Calculate Weisfeiler-Lehman kernels based on user-defined kernel between graphs.
		
		Parameters
		----------
		Gn : List of NetworkX graph
			List of graphs between which the kernels are calculated.	   
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
		gram_matrix : Numpy matrix
			Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
		"""	  
		pass
		# init.
		height = int(height)
		gram_matrix = np.zeros((len(Gn), len(Gn))) # init kernel
	  
		# initial for height = 0
		gram_matrix = base_kernel(Gn, node_label, edge_label)
				
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
					
			# calculate kernel with h iterations and add it to the final kernel
			gram_matrix += base_kernel(Gn, node_label, edge_label)
			
		return gram_matrix
	
	
	def __add_dummy_node_labels(self, Gn):
		if len(self.__node_labels) == 0 or (len(self.__node_labels) == 1 and self.__node_labels[0] == SpecialLabel.DUMMY):
			for i in range(len(Gn)):
				nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
			self.__node_labels = [SpecialLabel.DUMMY]
			
			
class WLSubtree(WeisfeilerLehman):
	
	def __init__(self, **kwargs):
		kwargs['base_kernel'] = 'subtree'
		super().__init__(**kwargs)