#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:24:58 2020

@author: ljia

@references:

    [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData
    Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
"""

import sys
from itertools import product
# from functools import partial
from gklearn.utils import get_iters
import numpy as np
from gklearn.utils.utils import getSPGraph
from gklearn.kernels import ShortestPath
import os
import pickle
from pympler import asizeof
import time
import networkx as nx


def load_results(file_name, fcsp):
	if os.path.isfile(file_name):
		with open(file_name, 'rb') as f:
			return pickle.load(f)
	else:
		results = {'nb_comparison': [], 'i': -1, 'j': -1, 'completed': False}
		if fcsp:
			results['vk_dict_mem'] = []
		return results


def save_results(file_name, results):
	with open(file_name, 'wb') as f:
		pickle.dump(results, f)


def estimate_vk_memory(obj, nb_nodes1, nb_nodes2):
# asizeof.asized(obj, detail=1).format()
# 	return asizeof.asizeof(obj)
	key, val = next(iter(obj.items()))
# 	key = dict.iterkeys().next()
# 	key_mem = asizeof.asizeof(key)
	dict_flat = sys.getsizeof(obj)
	key_mem = 64

	if isinstance(val, float):
		val_mem = 24
		mem = (key_mem + val_mem) * len(obj) + dict_flat + 28 * (nb_nodes1 + nb_nodes2)
	else: # value is True or False
		mem = (key_mem) * len(obj) + dict_flat + 52 + 28 * (nb_nodes1 + nb_nodes2)

# 	print(mem, asizeof.asizeof(obj), '\n', asizeof.asized(obj, detail=3).format(), '\n')
	return mem


def compute_stats(file_name, results):
	del results['i']
	del results['j']
	results['nb_comparison'] = np.mean(results['nb_comparison'])
	results['completed'] = True
	if 'vk_dict_mem' in results and len(results['vk_dict_mem']) > 0:
		results['vk_dict_mem'] = np.mean(results['vk_dict_mem'])
	save_results(file_name, results)


class SPSpace(ShortestPath):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._file_name = kwargs.get('file_name')

# 	@profile
	def _compute_gm_series(self):
		self._all_graphs_have_edges(self._graphs)
		# get shortest path graph of each graph.
		iterator = get_iters(self._graphs, desc='getting sp graphs', file=sys.stdout, verbose=(self._verbose >= 2))
		self._graphs = [getSPGraph(g, edge_weight=self._edge_weight) for g in iterator]


		results = load_results(self._file_name, self._fcsp)

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(self._graphs)), 2)
		len_itr = int(len(self._graphs) * (len(self._graphs) + 1) / 2)
		iterator = get_iters(itr, desc='Computing kernels',
					length=len_itr, file=sys.stdout,verbose=(self._verbose >= 2))

		time0 = time.time()
		for i, j in iterator:
			if i > results['i'] or (i == results['i'] and j > results['j']):
				data = self._sp_do_space(self._graphs[i], self._graphs[j])
				if self._fcsp:
					results['nb_comparison'].append(data[0])
					if data[1] != {}:
						results['vk_dict_mem'].append(estimate_vk_memory(data[1],
								    nx.number_of_nodes(self._graphs[i]),
									nx.number_of_nodes(self._graphs[j])))
				else:
					results['nb_comparison'].append(data)
				results['i'] = i
				results['j'] = j

				time1 = time.time()
				if time1 - time0 > 600:
					save_results(self._file_name, results)
					time0 = time1

		compute_stats(self._file_name, results)

		return gram_matrix


	def _sp_do_space(self, g1, g2):

		if self._fcsp: # @todo: it may be put outside the _sp_do().
			return self._sp_do_fcsp(g1, g2)
		else:
			return self._sp_do_naive(g1, g2)


	def _sp_do_fcsp(self, g1, g2):

		nb_comparison = 0

		# compute shortest path matrices first, method borrowed from FCSP.
		vk_dict = {}  # shortest path matrices dict
		if len(self._node_labels) > 0: # @todo: it may be put outside the _sp_do().
			# node symb and non-synb labeled
			if len(self._node_attrs) > 0:
				kn = self._node_kernels['mix']
				for n1, n2 in product(
						g1.nodes(data=True), g2.nodes(data=True)):
					n1_labels = [n1[1][nl] for nl in self._node_labels]
					n2_labels = [n2[1][nl] for nl in self._node_labels]
					n1_attrs = [n1[1][na] for na in self._node_attrs]
					n2_attrs = [n2[1][na] for na in self._node_attrs]
					vk_dict[(n1[0], n2[0])] = kn(n1_labels, n2_labels, n1_attrs, n2_attrs)
					nb_comparison += 1
			# node symb labeled
			else:
				kn = self._node_kernels['symb']
				for n1 in g1.nodes(data=True):
					for n2 in g2.nodes(data=True):
						n1_labels = [n1[1][nl] for nl in self._node_labels]
						n2_labels = [n2[1][nl] for nl in self._node_labels]
						vk_dict[(n1[0], n2[0])] = kn(n1_labels, n2_labels)
						nb_comparison += 1
		else:
			# node non-synb labeled
			if len(self._node_attrs) > 0:
				kn = self._node_kernels['nsymb']
				for n1 in g1.nodes(data=True):
					for n2 in g2.nodes(data=True):
						n1_attrs = [n1[1][na] for na in self._node_attrs]
						n2_attrs = [n2[1][na] for na in self._node_attrs]
						vk_dict[(n1[0], n2[0])] = kn(n1_attrs, n2_attrs)
						nb_comparison += 1
			# node unlabeled
			else:
				for e1, e2 in product(
						g1.edges(data=True), g2.edges(data=True)):
					pass
# 					if e1[2]['cost'] == e2[2]['cost']:
# 						kernel += 1
# 					nb_comparison += 1

		return nb_comparison, vk_dict

# 		# compute graph kernels
# 		if self._ds_infos['directed']:
# 			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
# 				if e1[2]['cost'] == e2[2]['cost']:
# 					nk11, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(e1[1], e2[1])]
# 					kn1 = nk11 * nk22
# 					kernel += kn1
# 		else:
# 			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
# 				if e1[2]['cost'] == e2[2]['cost']:
# 					# each edge walk is counted twice, starting from both its extreme nodes.
# 					nk11, nk12, nk21, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(
# 						e1[0], e2[1])], vk_dict[(e1[1], e2[0])], vk_dict[(e1[1], e2[1])]
# 					kn1 = nk11 * nk22
# 					kn2 = nk12 * nk21
# 					kernel += kn1 + kn2


	def _sp_do_naive(self, g1, g2):

		nb_comparison = 0

		# Define the function to compute kernels between vertices in each condition.
		if len(self._node_labels) > 0:
			# node symb and non-synb labeled
			if len(self._node_attrs) > 0:
				def compute_vk(n1, n2):
 					kn = self._node_kernels['mix']
 					n1_labels = [g1.nodes[n1][nl] for nl in self._node_labels]
 					n2_labels = [g2.nodes[n2][nl] for nl in self._node_labels]
 					n1_attrs = [g1.nodes[n1][na] for na in self._node_attrs]
 					n2_attrs = [g2.nodes[n2][na] for na in self._node_attrs]
 					return kn(n1_labels, n2_labels, n1_attrs, n2_attrs)
			# node symb labeled
			else:
				def compute_vk(n1, n2):
					kn = self._node_kernels['symb']
					n1_labels = [g1.nodes[n1][nl] for nl in self._node_labels]
					n2_labels = [g2.nodes[n2][nl] for nl in self._node_labels]
					return kn(n1_labels, n2_labels)
		else:
			# node non-synb labeled
			if len(self._node_attrs) > 0:
				def compute_vk(n1, n2):
					kn = self._node_kernels['nsymb']
					n1_attrs = [g1.nodes[n1][na] for na in self._node_attrs]
					n2_attrs = [g2.nodes[n2][na] for na in self._node_attrs]
					return kn(n1_attrs, n2_attrs)
			# node unlabeled
			else:
# 				for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
# 					if e1[2]['cost'] == e2[2]['cost']:
# 						kernel += 1
				return 0

		# compute graph kernels
		if self._ds_infos['directed']:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
# 					nk11, nk22 = compute_vk(e1[0], e2[0]), compute_vk(e1[1], e2[1])
# 					kn1 = nk11 * nk22
# 					kernel += kn1
					nb_comparison += 2
		else:
			for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
				if e1[2]['cost'] == e2[2]['cost']:
					# each edge walk is counted twice, starting from both its extreme nodes.
# 					nk11, nk12, nk21, nk22 = compute_vk(e1[0], e2[0]), compute_vk(
# 						e1[0], e2[1]), compute_vk(e1[1], e2[0]), compute_vk(e1[1], e2[1])
# 					kn1 = nk11 * nk22
# 					kn2 = nk12 * nk21
# 					kernel += kn1 + kn2
					nb_comparison += 4

		return nb_comparison