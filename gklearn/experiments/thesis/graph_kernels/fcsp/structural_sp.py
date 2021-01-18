#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:59:57 2020

@author: ljia

@references:

    [1] Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For
    Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).
"""
import sys
from itertools import product
from gklearn.utils import get_iters
import numpy as np
import time
import os, errno
import pickle
from pympler import asizeof
import networkx as nx
from gklearn.utils.utils import get_shortest_paths
from gklearn.kernels import StructuralSP


def load_splist(file_name):
	if os.path.isfile(file_name):
		with open(file_name, 'rb') as f:
			return pickle.load(f)
	else:
		results_path = {'splist': [], 'i': -1, 'completed': False}
		return results_path


def load_results(file_name, fcsp):
	if os.path.isfile(file_name):
		with open(file_name, 'rb') as f:
			return pickle.load(f)
	else:
		results = {'nb_v_comparison': [], 'nb_e_comparison': [], 'i': -1, 'j': -1, 'completed': False}
		if fcsp:
			results['vk_dict_mem'] = []
			results['ek_dict_mem'] = []
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


def estimate_ek_memory(obj, nb_nodes1, nb_nodes2):
# asizeof.asized(obj, detail=1).format()
# 	return asizeof.asizeof(obj)
	key, val = next(iter(obj.items()))
# 	key = dict.iterkeys().next()
# 	key_mem = asizeof.asizeof(key)
	dict_flat = sys.getsizeof(obj)
	key_mem = 192

	if isinstance(val, float):
		val_mem = 24
		mem = (key_mem + val_mem) * len(obj) + dict_flat + 28 * (nb_nodes1 + nb_nodes2)
	else: # value is True or False
		mem = (key_mem) * len(obj) + dict_flat + 52 + 28 * (nb_nodes1 + nb_nodes2)

# 	print(mem, asizeof.asizeof(obj), '\n', asizeof.asized(obj, detail=3).format(), '\n')
	return mem


def compute_stats(file_name, results, splist):
	del results['i']
	del results['j']
	results['nb_v_comparison'] = np.mean(results['nb_v_comparison'])
# 	if len(results['nb_e_comparison']) > 0:
	results['nb_e_comparison'] = np.mean(results['nb_e_comparison'])
	results['completed'] = True
	if 'vk_dict_mem' in results and len(results['vk_dict_mem']) > 0:
		results['vk_dict_mem'] = np.mean(results['vk_dict_mem'])
	if 'ek_dict_mem' in results and len(results['ek_dict_mem']) > 0:
		results['ek_dict_mem'] = np.mean(results['ek_dict_mem'])
	results['nb_sp_ave'] = np.mean([len(ps) for ps in splist])
	results['sp_len_ave'] = np.mean([np.mean([len(p) for p in ps]) for ps in splist])
	results['sp_mem_all'] = asizeof.asizeof(splist)
	save_results(file_name, results)


class SSPSpace(StructuralSP):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._file_name = kwargs.get('file_name')

# 	@profile
	def _compute_gm_series(self):
		# get shortest paths of each graph in the graphs.
		fn_paths = os.path.splitext(self._file_name)[0] + '.paths.pkl'
		results_path = load_splist(fn_paths)

		if not results_path['completed']:

			iterator = get_iters(self._graphs, desc='getting sp graphs', file=sys.stdout, verbose=(self._verbose >= 2))
			if self._compute_method == 'trie':
				for g in iterator:
					splist.append(self._get_sps_as_trie(g))
			else:
				time0 = time.time()
				for i, g in enumerate(iterator):
					if i > results_path['i']:
						results_path['splist'].append(get_shortest_paths(g, self._edge_weight, self._ds_infos['directed']))
						results_path['i'] = i

						time1 = time.time()
						if time1 - time0 > 600:
							save_results(fn_paths, results_path)
							time0 = time1

				del results_path['i']
				results_path['completed'] = True
				save_results(fn_paths, results_path)

		#########
		splist = results_path['splist']
		results = load_results(self._file_name, self._fcsp)

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(self._graphs)), 2)
		len_itr = int(len(self._graphs) * (len(self._graphs) + 1) / 2)
		iterator = get_iters(itr, desc='Computing kernels', file=sys.stdout,
					   length=len_itr, verbose=(self._verbose >= 2))
		if self._compute_method == 'trie':
			for i, j in iterator:
				kernel = self._ssp_do_trie(self._graphs[i], self._graphs[j], splist[i], splist[j])
				gram_matrix[i][j] = kernel
				gram_matrix[j][i] = kernel
		else:
			time0 = time.time()
			for i, j in iterator:
				if i > results['i'] or (i == results['i'] and j > results['j']):
					data = self._ssp_do_naive_space(self._graphs[i], self._graphs[j], splist[i], splist[j])
					results['nb_v_comparison'].append(data[0])
					results['nb_e_comparison'].append(data[1])
					if self._fcsp:
						if data[2] != {}:
							results['vk_dict_mem'].append(estimate_vk_memory(data[2],
									    nx.number_of_nodes(self._graphs[i]),
										nx.number_of_nodes(self._graphs[j])))
						if data[3] != {}:
							results['ek_dict_mem'].append(estimate_ek_memory(data[3],
									    nx.number_of_nodes(self._graphs[i]),
										nx.number_of_nodes(self._graphs[j])))
					results['i'] = i
					results['j'] = j

					time1 = time.time()
					if time1 - time0 > 600:
						save_results(self._file_name, results)
						time0 = time1

			compute_stats(self._file_name, results, splist)
			# @todo: may not remove the path file if the program stops exactly here.
			try:
				os.remove(fn_paths)
			except OSError as e:
				if e.errno != errno.ENOENT:
					raise

		return gram_matrix


	def _ssp_do_naive_space(self, g1, g2, spl1, spl2):
		if self._fcsp: # @todo: it may be put outside the _sp_do().
			return self._sp_do_naive_fcsp(g1, g2, spl1, spl2)
		else:
			return self._sp_do_naive_naive(g1, g2, spl1, spl2)


	def _sp_do_naive_fcsp(self, g1, g2, spl1, spl2):

		# First, compute shortest path matrices, method borrowed from FCSP.
		vk_dict, nb_v_comparison = self._get_all_node_kernels(g1, g2)
		# Then, compute kernels between all pairs of edges, which is an idea of
		# extension of FCSP. It suits sparse graphs, which is the most case we
		# went though. For dense graphs, this would be slow.
		ek_dict, nb_e_comparison = self._get_all_edge_kernels(g1, g2)

		return nb_v_comparison, nb_e_comparison, vk_dict, ek_dict


	def _sp_do_naive_naive(self, g1, g2, spl1, spl2):

		nb_v_comparison = 0
		nb_e_comparison = 0

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
# 			# node unlabeled
# 			else:
# 				for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
# 					if e1[2]['cost'] == e2[2]['cost']:
# 						kernel += 1
# 				return kernel

		# Define the function to compute kernels between edges in each condition.
		if len(self._edge_labels) > 0:
			# edge symb and non-synb labeled
			if len(self._edge_attrs) > 0:
				def compute_ek(e1, e2):
					ke = self._edge_kernels['mix']
					e1_labels = [g1.edges[e1][el] for el in self._edge_labels]
					e2_labels = [g2.edges[e2][el] for el in self._edge_labels]
					e1_attrs = [g1.edges[e1][ea] for ea in self._edge_attrs]
					e2_attrs = [g2.edges[e2][ea] for ea in self._edge_attrs]
					return ke(e1_labels, e2_labels, e1_attrs, e2_attrs)
			# edge symb labeled
			else:
				def compute_ek(e1, e2):
					ke = self._edge_kernels['symb']
					e1_labels = [g1.edges[e1][el] for el in self._edge_labels]
					e2_labels = [g2.edges[e2][el] for el in self._edge_labels]
					return ke(e1_labels, e2_labels)
		else:
			# edge non-synb labeled
			if len(self._edge_attrs) > 0:
				def compute_ek(e1, e2):
					ke = self._edge_kernels['nsymb']
					e1_attrs = [g1.edges[e1][ea] for ea in self._edge_attrs]
					e2_attrs = [g2.edges[e2][ea] for ea in self._edge_attrs]
					return ke(e1_attrs, e2_attrs)


		# compute graph kernels
		if len(self._node_labels) > 0 or len(self._node_attrs) > 0:
			if len(self._edge_labels) > 0 or len(self._edge_attrs) > 0:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
# 						nb_v_comparison = len(p1)
# 						nb_e_comparison = len(p1) - 1
						kpath = compute_vk(p1[0], p2[0])
						nb_v_comparison += 1
						if kpath:
							for idx in range(1, len(p1)):
								kpath *= compute_vk(p1[idx], p2[idx]) * \
									compute_ek((p1[idx-1], p1[idx]),
											 (p2[idx-1], p2[idx]))
								nb_v_comparison += 1
								nb_e_comparison += 1
								if not kpath:
									break
# 							kernel += kpath  # add up kernels of all paths
			else:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						kpath = compute_vk(p1[0], p2[0])
						nb_v_comparison += 1
						if kpath:
							for idx in range(1, len(p1)):
								kpath *= compute_vk(p1[idx], p2[idx])
								nb_v_comparison += 1
								if not kpath:
									break
# 							kernel += kpath  # add up kernels of all paths
		else:
			if len(self._edge_labels) > 0 or len(self._edge_attrs) > 0:
				for p1, p2 in product(spl1, spl2):
					if len(p1) == len(p2):
						if len(p1) == 0:
							pass
						else:
							kpath = 1
							for idx in range(0, len(p1) - 1):
								kpath *= compute_ek((p1[idx], p1[idx+1]),
												  (p2[idx], p2[idx+1]))
								nb_e_comparison += 1
								if not kpath:
									break
			else:
				pass
# 				for p1, p2 in product(spl1, spl2):
# 					if len(p1) == len(p2):
# 						kernel += 1
# 		try:
# 			kernel = kernel / (len(spl1) * len(spl2))  # Compute mean average
# 		except ZeroDivisionError:
# 			print(spl1, spl2)
# 			print(g1.nodes(data=True))
# 			print(g1.edges(data=True))
# 			raise Exception

		return nb_v_comparison, nb_e_comparison


	def _get_all_node_kernels(self, g1, g2):
		nb_comparison = 0

		vk_dict = {}  # shortest path matrices dict
		if len(self._node_labels) > 0:
			# node symb and non-synb labeled
			if len(self._node_attrs) > 0:
				kn = self._node_kernels['mix']
				for n1 in g1.nodes(data=True):
					for n2 in g2.nodes(data=True):
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
				pass # @todo: add edge weights.
	# 			for e1 in g1.edges(data=True):
	# 				for e2 in g2.edges(data=True):
	# 					if e1[2]['cost'] == e2[2]['cost']:
	# 						kernel += 1
	# 			return kernel

		return vk_dict, nb_comparison


	def _get_all_edge_kernels(self, g1, g2):
		nb_comparison = 0

		# compute kernels between all pairs of edges, which is an idea of
		# extension of FCSP. It suits sparse graphs, which is the most case we
		# went though. For dense graphs, this would be slow.
		ek_dict = {}  # dict of edge kernels
		if len(self._edge_labels) > 0:
			# edge symb and non-synb labeled
			if len(self._edge_attrs) > 0:
				ke = self._edge_kernels['mix']
				for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
					e1_labels = [e1[2][el] for el in self._edge_labels]
					e2_labels = [e2[2][el] for el in self._edge_labels]
					e1_attrs = [e1[2][ea] for ea in self._edge_attrs]
					e2_attrs = [e2[2][ea] for ea in self._edge_attrs]
					ek_temp = ke(e1_labels, e2_labels, e1_attrs, e2_attrs)
					ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
					ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
					ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
					ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
					nb_comparison += 1
			# edge symb labeled
			else:
				ke = self._edge_kernels['symb']
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						e1_labels = [e1[2][el] for el in self._edge_labels]
						e2_labels = [e2[2][el] for el in self._edge_labels]
						ek_temp = ke(e1_labels, e2_labels)
						ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
						nb_comparison += 1
		else:
			# edge non-synb labeled
			if len(self._edge_attrs) > 0:
				ke = self._edge_kernels['nsymb']
				for e1 in g1.edges(data=True):
					for e2 in g2.edges(data=True):
						e1_attrs = [e1[2][ea] for ea in self._edge_attrs]
						e2_attrs = [e2[2][ea] for ea in self._edge_attrs]
						ek_temp = ke(e1_attrs, e2_attrs)
						ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
						ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
						ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
						nb_comparison += 1
			# edge unlabeled
			else:
				pass

		return ek_dict, nb_comparison