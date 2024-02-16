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
from gklearn.utils import get_iters
import numpy as np
import networkx as nx
from collections import Counter
from itertools import chain
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from gklearn.utils import SpecialLabel
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.utils.utils import find_all_paths, get_mlti_dim_node_attrs
from gklearn.kernels import GraphKernel


class Treelet(GraphKernel):

	def __init__(self, **kwargs):
		"""Initialise a treelet kernel.
		"""
		GraphKernel.__init__(
			self, **{
				k: kwargs.get(k) for k in
				['parallel', 'n_jobs', 'chunksize', 'normalize', 'copy_graphs',
				 'verbose'] if k in kwargs}
		)
		self.node_labels = kwargs.get('node_labels', [])
		self.edge_labels = kwargs.get('edge_labels', [])
		self.sub_kernel = kwargs.get('sub_kernel', None)
		self.ds_infos = kwargs.get('ds_infos', {})
		self.precompute_canonkeys = kwargs.get('precompute_canonkeys', True)
		self.save_canonkeys = kwargs.get('save_canonkeys', True)


	##########################################################################
	# The following is the 1st paradigm to compute kernel matrix, which is
	# compatible with `scikit-learn`.
	# -------------------------------------------------------------------
	# Special thanks to the "GraKeL" library for providing an excellent template!
	##########################################################################

	def clear_attributes(self):
		super().clear_attributes()
		if hasattr(self, '_canonkeys'):
			delattr(self, '_canonkeys')
		if hasattr(self, '_Y_canonkeys'):
			delattr(self, '_Y_canonkeys')
		if hasattr(self, '_dummy_labels_considered'):
			delattr(self, '_dummy_labels_considered')


	def validate_parameters(self):
		"""Validate all parameters for the transformer.

		Returns
		-------
		None.

		"""
		super().validate_parameters()
		if self.sub_kernel is None:
			raise ValueError('Sub-kernel not set.')


	def _compute_kernel_matrix_series(self, Y, X=None, load_canonkeys=True):
		"""Compute the kernel matrix between a given target graphs (Y) and
		the fitted graphs (X / self._graphs) without parallelization.

		Parameters
		----------
		Y : list of graphs, optional
			The target graphs.

		Returns
		-------
		kernel_matrix : numpy array, shape = [n_targets, n_inputs]
			The computed kernel matrix.

		"""
		if_comp_X_canonkeys = True

		# if load saved canonkeys of X from the instance:
		if load_canonkeys:
			# Canonical keys for self._graphs.
			try:
				check_is_fitted(self, ['_canonkeys'])
				canonkeys_list1 = self._canonkeys
				if_comp_X_canonkeys = False
			except NotFittedError:
				import warnings
				warnings.warn(
					'The canonkeys of self._graphs are not computed/saved. The keys of `X` is computed instead.'
				)
				if_comp_X_canonkeys = True

		# get all canonical keys of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.

		# Compute the canonical keys of X.
		if if_comp_X_canonkeys:
			if X is None:
				raise ('X can not be None.')
			# self._add_dummy_labels will modify the input in place.
			self._add_dummy_labels(X)  # for X
			canonkeys_list1 = []
			iterator = get_iters(
				self._graphs, desc='Getting canonkeys for X', file=sys.stdout,
				verbose=(self.verbose >= 2)
			)
			for g in iterator:
				canonkeys_list1.append(self._get_canonkeys(g))

		# Canonical keys for Y.
		# 		Y = [g.copy() for g in Y] # @todo: ?
		self._add_dummy_labels(Y)
		canonkeys_list2 = []
		iterator = get_iters(
			Y, desc='Getting canonkeys for Y', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		for g in iterator:
			canonkeys_list2.append(self._get_canonkeys(g))

		# 		if self.save_canonkeys:
		# 			self._Y_canonkeys = canonkeys_list2

		# compute kernel matrix.
		kernel_matrix = np.zeros((len(Y), len(canonkeys_list1)))

		from itertools import product
		itr = product(range(len(Y)), range(len(canonkeys_list1)))
		len_itr = int(len(Y) * len(canonkeys_list1))
		iterator = get_iters(
			itr, desc='Computing kernels', file=sys.stdout,
			length=len_itr, verbose=(self.verbose >= 2)
		)
		for i_y, i_x in iterator:
			kernel = self._kernel_do(canonkeys_list2[i_y], canonkeys_list1[i_x])
			kernel_matrix[i_y][i_x] = kernel

		return kernel_matrix


	def _compute_kernel_matrix_imap_unordered(self, Y):
		"""Compute the kernel matrix between a given target graphs (Y) and
		the fitted graphs (X / self._graphs) using imap unordered parallelization.

		Parameters
		----------
		Y : list of graphs, optional
			The target graphs.

		Returns
		-------
		kernel_matrix : numpy array, shape = [n_targets, n_inputs]
			The computed kernel matrix.

		"""
		raise Exception('Parallelization for kernel matrix is not implemented.')


	def pairwise_kernel(self, x, y, are_keys=False):
		"""Compute pairwise kernel between two graphs.

		Parameters
		----------
		x, y : NetworkX Graph.
			Graphs bewteen which the kernel is computed.

		are_keys : boolean, optional
			If `True`, `x` and `y` are canonical keys, otherwise are graphs.
			The default is False.

		Returns
		-------
		kernel: float
			The computed kernel.

		"""
		if are_keys:
			# x, y are canonical keys.
			kernel = self._kernel_do(x, y)

		else:
			# x, y are graphs.
			kernel = self._compute_single_kernel_series(x, y)

		return kernel


	def diagonals(self):
		"""Compute the kernel matrix diagonals of the fit/transformed data.

		Returns
		-------
        X_diag : numpy array
            The diagonal of the kernel matrix between the fitted data.
            This consists of each element calculated with itself.

        Y_diag : numpy array
            The diagonal of the kernel matrix, of the transform.
            This consists of each element calculated with itself.

		"""
		# Check if method "fit" had been called.
		check_is_fitted(self, ['_graphs'])

		# Check if the diagonals of X exist.
		try:
			check_is_fitted(self, ['_X_diag'])
		except NotFittedError:
			# Compute diagonals of X.
			self._X_diag = np.empty(shape=(len(self._graphs),))
			try:
				check_is_fitted(self, ['_canonkeys'])
				for i, x in enumerate(self._canonkeys):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_keys=True
					)  # @todo: parallel?
			except NotFittedError:
				for i, x in enumerate(self._graphs):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_keys=False
					)  # @todo: parallel?

		try:
			# If transform has happened, return both diagonals.
			check_is_fitted(self, ['_Y'])
			self._Y_diag = np.empty(shape=(len(self._Y),))
			try:
				check_is_fitted(self, ['_Y_canonkeys'])
				for (i, y) in enumerate(self._Y_canonkeys):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_keys=True
					)  # @todo: parallel?
			except NotFittedError:
				for (i, y) in enumerate(self._Y):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_keys=False
					)  # @todo: parallel?

			return self._X_diag, self._Y_diag

		except NotFittedError:
			# Else just return both X_diag
			return self._X_diag


	##########################################################################
	# The following is the 2nd paradigm to compute kernel matrix. It is
	# simplified and not compatible with `scikit-learn`.
	##########################################################################


	def _compute_gm_series(self, graphs):
		self._add_dummy_labels(graphs)

		# get all canonical keys of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.
		canonkeys = []
		iterator = get_iters(
			graphs, desc='getting canonkeys', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		for g in iterator:
			canonkeys.append(self._get_canonkeys(g))

		if self.save_canonkeys:
			self._canonkeys = canonkeys

		# compute Gram matrix.
		gram_matrix = np.zeros((len(graphs), len(graphs)))

		from itertools import combinations_with_replacement
		itr = combinations_with_replacement(range(0, len(graphs)), 2)
		len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
		iterator = get_iters(
			itr, desc='Computing kernels', file=sys.stdout,
			length=len_itr, verbose=(self.verbose >= 2)
		)
		for i, j in iterator:
			kernel = self._kernel_do(canonkeys[i], canonkeys[j])
			gram_matrix[i][j] = kernel
			gram_matrix[j][i] = kernel  # @todo: no directed graph considered?

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		self._add_dummy_labels(self._graphs)

		# get all canonical keys of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.
		pool = Pool(self.n_jobs)
		itr = zip(self._graphs, range(0, len(self._graphs)))
		if len(self._graphs) < 100 * self.n_jobs:
			chunksize = int(len(self._graphs) / self.n_jobs) + 1
		else:
			chunksize = 100
		canonkeys = [[] for _ in range(len(self._graphs))]
		get_fun = self._wrapper_get_canonkeys
		iterator = get_iters(
			pool.imap_unordered(get_fun, itr, chunksize),
			desc='getting canonkeys', file=sys.stdout,
			length=len(self._graphs), verbose=(self.verbose >= 2)
		)
		for i, ck in iterator:
			canonkeys[i] = ck
		pool.close()
		pool.join()

		if self.save_canonkeys:
			self._canonkeys = canonkeys

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))


		def init_worker(canonkeys_toshare):
			global G_canonkeys
			G_canonkeys = canonkeys_toshare


		do_fun = self._wrapper_kernel_do
		parallel_gm(
			do_fun, gram_matrix, self._graphs, init_worker=init_worker,
			glbv=(canonkeys,), n_jobs=self.n_jobs, verbose=self.verbose
		)

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		# @TODO: Why this is commented out before?
		self._add_dummy_labels(g_list + [g1])

		# get all canonical keys of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.
		canonkeys_1 = self._get_canonkeys(g1)
		canonkeys_list = []
		iterator = get_iters(
			g_list, desc='getting canonkeys', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		for g in iterator:
			canonkeys_list.append(self._get_canonkeys(g))

		# compute kernel list.
		kernel_list = [None] * len(g_list)
		iterator = get_iters(
			range(len(g_list)), desc='Computing kernels', file=sys.stdout,
			length=len(g_list), verbose=(self.verbose >= 2)
		)
		for i in iterator:
			kernel = self._kernel_do(canonkeys_1, canonkeys_list[i])
			kernel_list[i] = kernel

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._add_dummy_labels(g_list + [g1])

		# get all canonical keys of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.
		canonkeys_1 = self._get_canonkeys(g1)
		canonkeys_list = [[] for _ in range(len(g_list))]
		pool = Pool(self.n_jobs)
		itr = zip(g_list, range(0, len(g_list)))
		if len(g_list) < 100 * self.n_jobs:
			chunksize = int(len(g_list) / self.n_jobs) + 1
		else:
			chunksize = 100
		get_fun = self._wrapper_get_canonkeys
		iterator = get_iters(
			pool.imap_unordered(get_fun, itr, chunksize),
			desc='getting canonkeys', file=sys.stdout,
			length=len(g_list), verbose=(self.verbose >= 2)
		)
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
		parallel_me(
			do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(canonkeys_1, canonkeys_list),
			method='imap_unordered',
			n_jobs=self.n_jobs, itr_desc='Computing kernels',
			verbose=self.verbose
		)

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		return itr, self._kernel_do(G_ck_1, G_ck_list[itr])


	def _compute_single_kernel_series(self, g1, g2):
		# @TODO: Why this is commented out before?
		self._add_dummy_labels([g1] + [g2])
		canonkeys_1 = self._get_canonkeys(g1)
		canonkeys_2 = self._get_canonkeys(g2)
		kernel = self._kernel_do(canonkeys_1, canonkeys_2)
		return kernel


	# 	@profile
	def _kernel_do(self, canonkey1, canonkey2):
		"""Compute treelet graph kernel between 2 graphs.

		Parameters
		----------
		canonkey1, canonkey2 : list
			List of canonical keys in 2 graphs, where each key is represented by a string.

		Return
		------
		kernel : float
			Treelet kernel between 2 graphs.
		"""
		keys = set(canonkey1.keys()) | set(canonkey2.keys())  # find same canonical keys in both graphs
		if len(keys) == 0:  # There is nothing in common...
		        return 0

		vector1 = np.array([canonkey1.get(key, 0) for key in keys])
		vector2 = np.array([canonkey2.get(key, 0) for key in keys])

		# 		vector1, vector2 = [], []
		# 		keys1, keys2 = canonkey1, canonkey2
		# 		keys_searched = {}
		# 		for k, v in canonkey1.items():
		# 			if k in keys2:
		# 				vector1.append(v)
		# 				vector2.append(canonkey2[k])
		# 				keys_searched[k] = v

		# 		for k, v in canonkey2.items():
		# 			if k in keys1 and k not in keys_searched:
		# 				vector1.append(canonkey1[k])
		# 				vector2.append(v)

		# 		vector1, vector2 = np.array(vector1), np.array(vector2)

		kernel = self.sub_kernel(vector1, vector2)
		return kernel


	def _wrapper_kernel_do(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do(G_canonkeys[i], G_canonkeys[j])


	def _get_canonkeys(self, G):
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
		patterns = {}  # a dictionary which consists of lists of patterns for all graphlet.
		canonkey = {}  # canonical key, a dictionary which records amount of every tree pattern.

		### structural analysis ###
		### In this section, a list of patterns is generated for each graphlet,
		### where every pattern is represented by nodes ordered by Morgan's
		### extended labeling.
		# linear patterns
		patterns['0'] = list(G.nodes())
		canonkey['0'] = nx.number_of_nodes(G)
		for i in range(1, 6):  # for i in range(1, 6):
			patterns[str(i)] = find_all_paths(G, i, self.ds_infos['directed'])
			canonkey[str(i)] = len(patterns[str(i)])

		# n-star patterns
		patterns['3star'] = [[node] + [neighbor for neighbor in G[node]] for
		                     node in G.nodes() if len(G[node]) == 3]
		patterns['4star'] = [[node] + [neighbor for neighbor in G[node]] for
		                     node in G.nodes() if len(G[node]) == 4]
		patterns['5star'] = [[node] + [neighbor for neighbor in G[node]] for
		                     node in G.nodes() if len(G[node]) == 5]
		# n-star patterns
		canonkey['6'] = len(patterns['3star'])
		canonkey['8'] = len(patterns['4star'])
		canonkey['d'] = len(patterns['5star'])

		# pattern 7
		patterns['7'] = []  # the 1st line of Table 1 in Ref [1]
		for pattern in patterns['3star']:
			for i in range(1, len(pattern)):  # for each neighbor of node 0
				if len(G[pattern[i]]) >= 2:
					pattern_t = pattern[:]
					# set the node with degree >= 2 as the 4th node
					pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
					for neighborx in G[pattern[i]]:
						if neighborx != pattern[0]:
							new_pattern = pattern_t + [neighborx]
							patterns['7'].append(new_pattern)
		canonkey['7'] = len(patterns['7'])

		# pattern 11
		patterns['11'] = []  # the 4th line of Table 1 in Ref [1]
		for pattern in patterns['4star']:
			for i in range(1, len(pattern)):
				if len(G[pattern[i]]) >= 2:
					pattern_t = pattern[:]
					pattern_t[i], pattern_t[4] = pattern_t[4], pattern_t[i]
					for neighborx in G[pattern[i]]:
						if neighborx != pattern[0]:
							new_pattern = pattern_t + [neighborx]
							patterns['11'].append(new_pattern)
		canonkey['b'] = len(patterns['11'])

		# pattern 12
		patterns['12'] = []  # the 5th line of Table 1 in Ref [1]
		rootlist = []  # a list of root nodes, whose extended labels are 3
		for pattern in patterns['3star']:
			if pattern[
				0] not in rootlist:  # prevent to count the same pattern twice from each of the two root nodes
				rootlist.append(pattern[0])
				for i in range(1, len(pattern)):
					if len(G[pattern[i]]) >= 3:
						rootlist.append(pattern[i])
						pattern_t = pattern[:]
						pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
						for neighborx1 in G[pattern[i]]:
							if neighborx1 != pattern[0]:
								for neighborx2 in G[pattern[i]]:
									if neighborx1 > neighborx2 and neighborx2 != \
											pattern[0]:
										new_pattern = pattern_t + [
											neighborx1] + [neighborx2]
										#						 new_patterns = [ pattern + [neighborx1] + [neighborx2] for neighborx1 in G[pattern[i]] if neighborx1 != pattern[0] for neighborx2 in G[pattern[i]] if (neighborx1 > neighborx2 and neighborx2 != pattern[0]) ]
										patterns['12'].append(new_pattern)
		canonkey['c'] = int(len(patterns['12']) / 2)

		# pattern 9
		# todo: this is not correct for self loops, but for now, we simply remove
		# self loops from the graph at the beginning of the model,
		# see GraphKernel._compute_gram_matrix().
		patterns['9'] = []  # the 2nd line of Table 1 in Ref [1]
		for pattern in patterns['3star']:
			for pairs in [[neighbor1, neighbor2] for neighbor1 in G[pattern[0]]
			              if len(G[neighbor1]) >= 2 \
			              for neighbor2 in G[pattern[0]] if
			              len(G[neighbor2]) >= 2 if neighbor1 > neighbor2]:
				pattern_t = pattern[:]
				# move nodes with extended labels 4 to specific position to correspond to their children
				pattern_t[pattern_t.index(pairs[0])], pattern_t[2] = pattern_t[
					2], pattern_t[pattern_t.index(pairs[0])]
				pattern_t[pattern_t.index(pairs[1])], pattern_t[3] = pattern_t[
					3], pattern_t[pattern_t.index(pairs[1])]
				for neighborx1 in G[pairs[0]]:
					if neighborx1 != pattern[0]:
						for neighborx2 in G[pairs[1]]:
							if neighborx2 != pattern[0]:
								new_pattern = pattern_t + [neighborx1] + [
									neighborx2]
								patterns['9'].append(new_pattern)
		canonkey['9'] = len(patterns['9'])

		# pattern 10
		patterns['10'] = []  # the 3rd line of Table 1 in Ref [1]
		for pattern in patterns['3star']:
			for i in range(1, len(pattern)):
				if len(G[pattern[i]]) >= 2:
					for neighborx in G[pattern[i]]:
						if neighborx != pattern[0] and len(G[neighborx]) >= 2:
							pattern_t = pattern[:]
							pattern_t[i], pattern_t[3] = pattern_t[3], \
							pattern_t[i]
							new_patterns = [
								pattern_t + [neighborx] + [neighborxx] for
								neighborxx in G[neighborx] if
								neighborxx != pattern[i]]
							patterns['10'].extend(new_patterns)
		canonkey['a'] = len(patterns['10'])

		### labeling information ###
		### In this section, a list of canonical keys is generated for every
		### pattern obtained in the structural analysis section above, which is a
		### string corresponding to a unique treelet. A dictionary is built to keep
		### track of the amount of every treelet.
		if len(self.node_labels) > 0 or len(self.edge_labels) > 0:
			canonkey_l = {}  # canonical key, a dictionary which keeps track of amount of every treelet.

			# linear patterns
			canonkey_t = Counter(get_mlti_dim_node_attrs(G, self.node_labels))
			for key in canonkey_t:
				canonkey_l[('0', key)] = canonkey_t[key]

			for i in range(1, 6):  # for i in range(1, 6):
				treelet = []
				for pattern in patterns[str(i)]:
					canonlist = []
					for idx, node in enumerate(pattern[:-1]):
						canonlist.append(
							tuple(G.nodes[node][nl] for nl in self.node_labels)
						)
						canonlist.append(
							tuple(
								G[node][pattern[idx + 1]][el] for el in
								self.edge_labels
							)
						)
					canonlist.append(
						tuple(
							G.nodes[pattern[-1]][nl] for nl in self.node_labels
						)
					)
					canonkey_t = canonlist if canonlist < canonlist[
					                                      ::-1] else canonlist[
					                                                 ::-1]
					treelet.append(tuple([str(i)] + canonkey_t))
				canonkey_l.update(Counter(treelet))

			# n-star patterns
			for i in range(3, 6):
				treelet = []
				for pattern in patterns[str(i) + 'star']:
					canonlist = []
					for leaf in pattern[1:]:
						nlabels = tuple(
							G.nodes[leaf][nl] for nl in self.node_labels
						)
						elabels = tuple(
							G[leaf][pattern[0]][el] for el in self.edge_labels
						)
						canonlist.append(tuple((nlabels, elabels)))
					canonlist.sort()
					canonlist = list(chain.from_iterable(canonlist))
					canonkey_t = tuple(
						['d' if i == 5 else str(i * 2)] +
						[tuple(
							G.nodes[pattern[0]][nl] for nl in self.node_labels
						)]
						+ canonlist
						)
					treelet.append(canonkey_t)
				canonkey_l.update(Counter(treelet))

			# pattern 7
			treelet = []
			for pattern in patterns['7']:
				canonlist = []
				for leaf in pattern[1:3]:
					nlabels = tuple(
						G.nodes[leaf][nl] for nl in self.node_labels
					)
					elabels = tuple(
						G[leaf][pattern[0]][el] for el in self.edge_labels
					)
					canonlist.append(tuple((nlabels, elabels)))
				canonlist.sort()
				canonlist = list(chain.from_iterable(canonlist))
				canonkey_t = tuple(
					['7']
					+ [tuple(
						G.nodes[pattern[0]][nl] for nl in self.node_labels
					)] + canonlist
					+ [tuple(
						G.nodes[pattern[3]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[3]][pattern[0]][el] for el in self.edge_labels
					)]
					+ [tuple(
						G.nodes[pattern[4]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[4]][pattern[3]][el] for el in self.edge_labels
					)]
					)
				treelet.append(canonkey_t)
			canonkey_l.update(Counter(treelet))

			# pattern 11
			treelet = []
			for pattern in patterns['11']:
				canonlist = []
				for leaf in pattern[1:4]:
					nlabels = tuple(
						G.nodes[leaf][nl] for nl in self.node_labels
					)
					elabels = tuple(
						G[leaf][pattern[0]][el] for el in self.edge_labels
					)
					canonlist.append(tuple((nlabels, elabels)))
				canonlist.sort()
				canonlist = list(chain.from_iterable(canonlist))
				canonkey_t = tuple(
					['b']
					+ [tuple(
						G.nodes[pattern[0]][nl] for nl in self.node_labels
					)] + canonlist
					+ [tuple(
						G.nodes[pattern[4]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[4]][pattern[0]][el] for el in self.edge_labels
					)]
					+ [tuple(
						G.nodes[pattern[5]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[5]][pattern[4]][el] for el in self.edge_labels
					)]
					)
				treelet.append(canonkey_t)
			canonkey_l.update(Counter(treelet))

			# pattern 10
			treelet = []
			for pattern in patterns['10']:
				canonkey4 = [
					tuple(G.nodes[pattern[5]][nl] for nl in self.node_labels),
					tuple(
						G[pattern[5]][pattern[4]][el] for el in self.edge_labels
					)]
				canonlist = []
				for leaf in pattern[1:3]:
					nlabels = tuple(
						G.nodes[leaf][nl] for nl in self.node_labels
					)
					elabels = tuple(
						G[leaf][pattern[0]][el] for el in self.edge_labels
					)
					canonlist.append(tuple((nlabels, elabels)))
				canonlist.sort()
				canonkey0 = list(chain.from_iterable(canonlist))
				canonkey_t = tuple(
					['a']
					+ [tuple(
						G.nodes[pattern[3]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G.nodes[pattern[4]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[4]][pattern[3]][el] for el in self.edge_labels
					)]
					+ [tuple(
						G.nodes[pattern[0]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[0]][pattern[3]][el] for el in self.edge_labels
					)]
					+ canonkey4 + canonkey0
					)
				treelet.append(canonkey_t)
			canonkey_l.update(Counter(treelet))

			# pattern 12
			treelet = []
			for pattern in patterns['12']:
				canonlist0 = []
				for leaf in pattern[1:3]:
					nlabels = tuple(
						G.nodes[leaf][nl] for nl in self.node_labels
					)
					elabels = tuple(
						G[leaf][pattern[0]][el] for el in self.edge_labels
					)
					canonlist0.append(tuple((nlabels, elabels)))
				canonlist0.sort()
				canonlist0 = list(chain.from_iterable(canonlist0))
				canonlist3 = []
				for leaf in pattern[4:6]:
					nlabels = tuple(
						G.nodes[leaf][nl] for nl in self.node_labels
					)
					elabels = tuple(
						G[leaf][pattern[3]][el] for el in self.edge_labels
					)
					canonlist3.append(tuple((nlabels, elabels)))
				canonlist3.sort()
				canonlist3 = list(chain.from_iterable(canonlist3))

				# 2 possible key can be generated from 2 nodes with extended label 3,
				# select the one with lower lexicographic order.
				canonkey_t1 = tuple(
					['c']
					+ [tuple(
						G.nodes[pattern[0]][nl] for nl in self.node_labels
					)] + canonlist0
					+ [tuple(
						G.nodes[pattern[3]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[3]][pattern[0]][el] for el in self.edge_labels
					)]
					+ canonlist3
					)
				canonkey_t2 = tuple(
					['c']
					+ [tuple(
						G.nodes[pattern[3]][nl] for nl in self.node_labels
					)] + canonlist3
					+ [tuple(
						G.nodes[pattern[0]][nl] for nl in self.node_labels
					)]
					+ [tuple(
						G[pattern[0]][pattern[3]][el] for el in self.edge_labels
					)]
					+ canonlist0
					)
				treelet.append(
					canonkey_t1 if canonkey_t1 < canonkey_t2 else canonkey_t2
				)
			canonkey_l.update(Counter(treelet))

			# pattern 9
			treelet = []
			for pattern in patterns['9']:
				canonkey2 = [
					tuple(G.nodes[pattern[4]][nl] for nl in self.node_labels),
					tuple(
						G[pattern[4]][pattern[2]][el] for el in self.edge_labels
					)]
				canonkey3 = [
					tuple(G.nodes[pattern[5]][nl] for nl in self.node_labels),
					tuple(
						G[pattern[5]][pattern[3]][el] for el in self.edge_labels
					)]
				prekey2 = [
					tuple(G.nodes[pattern[2]][nl] for nl in self.node_labels),
					tuple(
						G[pattern[2]][pattern[0]][el] for el in self.edge_labels
					)]
				prekey3 = [
					tuple(G.nodes[pattern[3]][nl] for nl in self.node_labels),
					tuple(
						G[pattern[3]][pattern[0]][el] for el in self.edge_labels
					)]
				if prekey2 + canonkey2 < prekey3 + canonkey3:
					canonkey_t = [tuple(
						G.nodes[pattern[1]][nl] for nl in self.node_labels
					)] \
					             + [tuple(
						G[pattern[1]][pattern[0]][el] for el in self.edge_labels
					)] \
					             + prekey2 + prekey3 + canonkey2 + canonkey3
				else:
					canonkey_t = [tuple(
						G.nodes[pattern[1]][nl] for nl in self.node_labels
					)] \
					             + [tuple(
						G[pattern[1]][pattern[0]][el] for el in self.edge_labels
					)] \
					             + prekey3 + prekey2 + canonkey3 + canonkey2
				treelet.append(
					tuple(
						['9']
						+ [tuple(
							G.nodes[pattern[0]][nl] for nl in self.node_labels
						)]
						+ canonkey_t
						)
				)
			canonkey_l.update(Counter(treelet))

			return canonkey_l

		return canonkey


	def _wrapper_get_canonkeys(self, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, self._get_canonkeys(g)


	def _add_dummy_labels(self, Gn=None):
		def _add_dummy(Gn):
			if len(self.node_labels) == 0 or (
					len(self.node_labels) == 1 and self.node_labels[
				0] == SpecialLabel.DUMMY):
				for i in range(len(Gn)):
					nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
				self.node_labels = [SpecialLabel.DUMMY]
			if len(self.edge_labels) == 0 or (
					len(self.edge_labels) == 1 and self.edge_labels[
				0] == SpecialLabel.DUMMY):
				for i in range(len(Gn)):
					nx.set_edge_attributes(Gn[i], '0', SpecialLabel.DUMMY)
				self.edge_labels = [SpecialLabel.DUMMY]


		if Gn is None or Gn is self._graphs:
			# Add dummy labels for the copy of self._graphs.
			try:
				check_is_fitted(self, ['_dummy_labels_considered'])
				if not self._dummy_labels_considered:
					Gn = self._graphs  # @todo: ?[g.copy() for g in self._graphs]
					_add_dummy(Gn)
					self._graphs = Gn
					self._dummy_labels_considered = True
			except NotFittedError:
				Gn = self._graphs  # @todo: ?[g.copy() for g in self._graphs]
				_add_dummy(Gn)
				self._graphs = Gn
				self._dummy_labels_considered = True

		else:
			# Add dummy labels for the input.
			_add_dummy(Gn)
