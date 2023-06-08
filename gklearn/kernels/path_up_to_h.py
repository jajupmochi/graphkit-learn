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
from gklearn.utils import get_iters
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np
import networkx as nx
from collections import Counter
from functools import partial
from gklearn.utils import SpecialLabel
from gklearn.utils.parallel import parallel_gm, parallel_me
from gklearn.kernels import GraphKernel
from gklearn.utils import Trie


class PathUpToH(GraphKernel):  # @todo: add function for k_func is None

	def __init__(self, **kwargs):
		GraphKernel.__init__(
			self, **{
				k: kwargs.get(k) for k in
				['parallel', 'n_jobs', 'chunksize', 'normalize', 'copy_graphs',
				 'verbose'] if k in kwargs}
		)
		self._node_labels = kwargs.get('node_labels', [])
		self._edge_labels = kwargs.get('edge_labels', [])
		self._depth = int(kwargs.get('depth', 10))
		self._k_func = kwargs.get('k_func', 'MinMax')
		self._compute_method = kwargs.get('compute_method', 'trie')
		self._ds_infos = kwargs.get('ds_infos', {})
		self._save_paths = kwargs.get('save_paths', True)
		if self._compute_method == 'trie':
			self._path_func = self._find_all_path_as_trie
			self._kernel_do_func = self._kernel_do_trie
		else:
			self._path_func = self._find_all_paths_until_length
			self._kernel_do_func = self._kernel_do_naive


	##########################################################################
	# The following is the 1st paradigm to compute kernel matrix, which is
	# compatible with `scikit-learn`.
	# -------------------------------------------------------------------
	# Special thanks to the "GraKeL" library for providing an excellent template!
	##########################################################################


	def clear_attributes(self):
		super().clear_attributes()
		if hasattr(self, '_all_paths'):
			delattr(self, '_all_paths')
		if hasattr(self, '_Y_all_paths'):
			delattr(self, '_Y_all_paths')


	def validate_parameters(self):
		super().validate_parameters()


	# if self._depth < 1:
	# 	raise ValueError('`depth` must be greater than 0.')
	# if self._k_func not in ['MinMax', 'tanimoto']:
	# 	raise ValueError('`k_func` must be either `MinMax` or `tanimoto`.')
	# if self._compute_method not in ['trie']:
	# 	raise ValueError('`compute_method` must be `trie`.')


	def _compute_kernel_matrix_series(self, Y, X=None, load_paths=True):
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
		if_comp_X_paths = True

		# if load saved paths of X from the instance:
		if load_paths:
			# paths for self._graphs.
			try:
				check_is_fitted(self, ['_all_paths'])
				paths_list1 = self._all_paths
				if_comp_X_paths = False
			except NotFittedError:
				import warnings
				warnings.warn(
					'The paths of self._graphs are not computed/saved. The paths of `X` is computed instead.'
				)
				if_comp_X_paths = True

		# Get all paths of all graphs before computing kernels to save
		# time, but this may cost a lot of memory for large dataset.

		# Compute the paths of X.
		if if_comp_X_paths:
			if X is None:
				raise ('X can not be None.')
			# self._add_dummy_labels will modify the input in place.
			self._add_dummy_labels(X)  # for X
			paths_list1 = []
			iterator = get_iters(
				self._graphs, desc='Getting paths for X',
				file=sys.stdout, verbose=(self.verbose >= 2)
			)
			for g in iterator:
				paths_list1.append(self._path_func(g))

		# Paths for Y.
		# 		Y = [g.copy() for g in Y] # @todo: ?
		self._add_dummy_labels(Y)
		paths_list2 = []
		iterator = get_iters(
			Y, desc='Getting paths for Y', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		for g in iterator:
			paths_list2.append(self._path_func(g))

		# 		if self.save_paths:
		# 			self._Y_paths = paths_list2

		# compute kernel matrix.
		kernel_matrix = np.zeros((len(Y), len(paths_list1)))

		from itertools import product
		itr = product(range(len(Y)), range(len(paths_list1)))
		len_itr = int(len(Y) * len(paths_list1))
		iterator = get_iters(
			itr, desc='Computing kernels', file=sys.stdout,
			length=len_itr, verbose=(self.verbose >= 2)
		)
		for i_y, i_x in iterator:
			kernel = self._kernel_do_func(
				paths_list2[i_y], paths_list1[i_x]
			)
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
		raise NotImplementedError(
			'Parallelization for kernel matrix is not implemented.'
		)


	def pairwise_kernel(self, x, y, are_paths=False):
		"""Compute pairwise kernel between two graphs.

		Parameters
		----------
		x, y : NetworkX Graph.
			Graphs bewteen which the kernel is computed.

		are_keys : boolean, optional
			If `True`, `x` and `y` are paths, otherwise are graphs.
			The default is False.

		Returns
		-------
		kernel: float
			The computed kernel.

		"""
		if are_paths:
			# x, y are canonical keys.
			kernel = self._kernel_do_func(x, y)

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
				check_is_fitted(self, ['_all_paths'])
				for i, x in enumerate(self._all_graphs):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_paths=True
					)  # @todo: parallel?
			except NotFittedError:
				for i, x in enumerate(self._graphs):
					self._X_diag[i] = self.pairwise_kernel(
						x, x, are_paths=False
					)  # @todo: parallel?

		try:
			# If transform has happened, return both diagonals.
			check_is_fitted(self, ['_Y'])
			self._Y_diag = np.empty(shape=(len(self._Y),))
			try:
				check_is_fitted(self, ['_Y_all_paths'])
				for (i, y) in enumerate(self._Y_all_paths):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_paths=True
					)  # @todo: parallel?
			except NotFittedError:
				for (i, y) in enumerate(self._Y):
					self._Y_diag[i] = self.pairwise_kernel(
						y, y, are_paths=False
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

		from itertools import combinations_with_replacement
		itr_kernel = combinations_with_replacement(range(0, len(graphs)), 2)
		iterator_ps = get_iters(
			range(0, len(graphs)), desc='getting paths', file=sys.stdout,
			length=len(graphs), verbose=(self.verbose >= 2)
		)
		len_itr = int(len(graphs) * (len(graphs) + 1) / 2)
		iterator_kernel = get_iters(
			itr_kernel, desc='Computing kernels',
			file=sys.stdout, length=len_itr, verbose=(self.verbose >= 2)
		)

		gram_matrix = np.zeros((len(graphs), len(graphs)))

		all_paths = [self._path_func(graphs[i]) for i in iterator_ps]
		if self._save_paths:
			self._all_paths = all_paths

		for i, j in iterator_kernel:
			kernel = self._kernel_do_func(all_paths[i], all_paths[j])
			gram_matrix[i][j] = kernel
			gram_matrix[j][i] = kernel

		return gram_matrix


	def _compute_gm_imap_unordered(self):
		self._add_dummy_labels(self._graphs)

		# get all paths of all graphs before computing kernels to save time,
		# but this may cost a lot of memory for large datasets.
		pool = Pool(self.n_jobs)
		itr = zip(self._graphs, range(0, len(self._graphs)))
		if len(self._graphs) < 100 * self.n_jobs:
			chunksize = int(len(self._graphs) / self.n_jobs) + 1
		else:
			chunksize = 100
		all_paths = [[] for _ in range(len(self._graphs))]
		if self._compute_method == 'trie' and self._k_func is not None:
			get_ps_fun = self._wrapper_find_all_path_as_trie
		elif self._compute_method != 'trie' and self._k_func is not None:
			get_ps_fun = partial(
				self._wrapper_find_all_paths_until_length, True
			)
		else:
			get_ps_fun = partial(
				self._wrapper_find_all_paths_until_length, False
			)
		iterator = get_iters(
			pool.imap_unordered(get_ps_fun, itr, chunksize),
			desc='getting paths', file=sys.stdout,
			length=len(self._graphs), verbose=(self.verbose >= 2)
		)
		for i, ps in iterator:
			all_paths[i] = ps
		pool.close()
		pool.join()

		if self._save_paths:
			self._all_paths = all_paths

		# compute Gram matrix.
		gram_matrix = np.zeros((len(self._graphs), len(self._graphs)))

		if self._compute_method == 'trie' and self._k_func is not None:
			def init_worker(trie_toshare):
				global G_trie
				G_trie = trie_toshare


			do_fun = self._wrapper_kernel_do_trie
		elif self._compute_method != 'trie' and self._k_func is not None:
			def init_worker(plist_toshare):
				global G_plist
				G_plist = plist_toshare


			do_fun = self._wrapper_kernel_do_naive
		else:
			def init_worker(plist_toshare):
				global G_plist
				G_plist = plist_toshare


			do_fun = self._wrapper_kernel_do_kernelless  # @todo: what is this?
		parallel_gm(
			do_fun, gram_matrix, self._graphs, init_worker=init_worker,
			glbv=(all_paths,), n_jobs=self.n_jobs, verbose=self.verbose
		)

		return gram_matrix


	def _compute_kernel_list_series(self, g1, g_list):
		self._add_dummy_labels(g_list + [g1])

		iterator_ps = get_iters(
			g_list, desc='getting paths', file=sys.stdout,
			verbose=(self.verbose >= 2)
		)
		iterator_kernel = get_iters(
			range(len(g_list)), desc='Computing kernels', file=sys.stdout,
			length=len(g_list), verbose=(self.verbose >= 2)
		)

		kernel_list = [None] * len(g_list)

		if self._compute_method == 'trie':
			paths_g1 = self._find_all_path_as_trie(g1)
			paths_g_list = [self._find_all_path_as_trie(g) for g in iterator_ps]
			for i in iterator_kernel:
				kernel = self._kernel_do_trie(paths_g1, paths_g_list[i])
				kernel_list[i] = kernel
		else:
			paths_g1 = self._find_all_paths_until_length(g1)
			paths_g_list = [self._find_all_paths_until_length(g) for g in
			                iterator_ps]
			for i in iterator_kernel:
				kernel = self._kernel_do_naive(paths_g1, paths_g_list[i])
				kernel_list[i] = kernel

		return kernel_list


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		self._add_dummy_labels(g_list + [g1])

		# get all paths of all graphs before computing kernels to save time,
		# but this may cost a lot of memory for large datasets.
		pool = Pool(self.n_jobs)
		itr = zip(g_list, range(0, len(g_list)))
		if len(g_list) < 100 * self.n_jobs:
			chunksize = int(len(g_list) / self.n_jobs) + 1
		else:
			chunksize = 100
		paths_g_list = [[] for _ in range(len(g_list))]
		if self._compute_method == 'trie' and self._k_func is not None:
			paths_g1 = self._find_all_path_as_trie(g1)
			get_ps_fun = self._wrapper_find_all_path_as_trie
		elif self._compute_method != 'trie' and self._k_func is not None:
			paths_g1 = self._find_all_paths_until_length(g1)
			get_ps_fun = partial(
				self._wrapper_find_all_paths_until_length, True
			)
		else:
			paths_g1 = self._find_all_paths_until_length(g1)
			get_ps_fun = partial(
				self._wrapper_find_all_paths_until_length, False
			)
		iterator = get_iters(
			pool.imap_unordered(get_ps_fun, itr, chunksize),
			desc='getting paths', file=sys.stdout,
			length=len(g_list), verbose=(self.verbose >= 2)
		)
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
		parallel_me(
			do_fun, func_assign, kernel_list, itr, len_itr=len_itr,
			init_worker=init_worker, glbv=(paths_g1, paths_g_list),
			method='imap_unordered', n_jobs=self.n_jobs,
			itr_desc='Computing kernels', verbose=self.verbose
		)

		return kernel_list


	def _wrapper_kernel_list_do(self, itr):
		if self._compute_method == 'trie' and self._k_func is not None:
			return itr, self._kernel_do_trie(G_p1, G_plist[itr])
		elif self._compute_method != 'trie' and self._k_func is not None:
			return itr, self._kernel_do_naive(G_p1, G_plist[itr])
		else:
			return itr, self._kernel_do_kernelless(G_p1, G_plist[itr])


	def _compute_single_kernel_series(self, g1, g2):
		self._add_dummy_labels([g1] + [g2])
		paths_g1 = self._path_func(g1)
		paths_g2 = self._path_func(g2)
		kernel = self._kernel_do_func(paths_g1, paths_g2)
		return kernel


	def _kernel_do_trie(self, trie1, trie2):
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
		if self._k_func == 'tanimoto':
			# traverse all paths in graph1 and search them in graph2. Deep-first
			# search is applied.
			def traverseTrie1t(
					root, trie2, setlist, pcurrent=[]
			):  # @todo: no need to use value (# of occurrence of paths) in this case.
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


			setlist = [0, 0]  # intersection and union of path sets of g1, g2.
			#		print(trie1.root)
			#		print(trie2.root)
			traverseTrie1t(trie1.root, trie2, setlist)
			#		print(setlist)
			traverseTrie2t(trie2.root, trie1, setlist)
			#		print(setlist)
			kernel = setlist[0] / setlist[1]

		elif self._k_func == 'MinMax':  # MinMax kernel
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


			sumlist = [0, 0]  # sum of mins and sum of maxs
			# 			print(trie1.root)
			# 			print(trie2.root)
			traverseTrie1m(trie1.root, trie2, sumlist)
			# 			print(sumlist)
			traverseTrie2m(trie2.root, trie1, sumlist)
			# 			print(sumlist)
			kernel = sumlist[0] / sumlist[1]
		else:
			raise Exception(
				'The given "k_func" cannot be recognized. Possible choices include: "tanimoto", "MinMax".'
			)

		return kernel


	def _wrapper_kernel_do_trie(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do_trie(G_trie[i], G_trie[j])


	def _kernel_do_naive(self, paths1, paths2):
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

		if self._k_func == 'tanimoto':
			length_union = len(set(paths1 + paths2))
			kernel = (len(set(paths1)) + len(set(paths2)) -
			          length_union) / length_union
		#		vector1 = [(1 if path in paths1 else 0) for path in all_paths]
		#		vector2 = [(1 if path in paths2 else 0) for path in all_paths]
		#		kernel_uv = np.dot(vector1, vector2)
		#		kernel = kernel_uv / (len(set(paths1)) + len(set(paths2)) - kernel_uv)

		elif self._k_func == 'MinMax':  # MinMax kernel
			path_count1 = Counter(paths1)
			path_count2 = Counter(paths2)
			vector1 = [(path_count1[key] if (key in path_count1.keys()) else 0)
			           for key in all_paths]
			vector2 = [(path_count2[key] if (key in path_count2.keys()) else 0)
			           for key in all_paths]
			kernel = np.sum(np.minimum(vector1, vector2)) / \
			         np.sum(np.maximum(vector1, vector2))

		elif self._k_func is None:  # no sub-kernel used; compare paths directly.
			path_count1 = Counter(paths1)
			path_count2 = Counter(paths2)
			vector1 = [(path_count1[key] if (key in path_count1.keys()) else 0)
			           for key in all_paths]
			vector2 = [(path_count2[key] if (key in path_count2.keys()) else 0)
			           for key in all_paths]
			kernel = np.dot(vector1, vector2)

		else:
			raise Exception(
				'The given "k_func" cannot be recognized. Possible choices include: "tanimoto", "MinMax" and None.'
			)

		return kernel


	def _wrapper_kernel_do_naive(self, itr):
		i = itr[0]
		j = itr[1]
		return i, j, self._kernel_do_naive(G_plist[i], G_plist[j])


	def _find_all_path_as_trie(self, G):
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
			if len(pcurrent) < self._depth + 1:
				for neighbor in G[root]:
					if neighbor not in pcurrent:
						pcurrent.append(neighbor)
						plstr = self._paths2labelseqs([pcurrent], G)
						ptrie.insertWord(plstr[0])
						traverseGraph(neighbor, ptrie, G, pcurrent)
			del pcurrent[-1]


		ptrie = Trie()
		path_l = [[n] for n in G.nodes]  # paths of length l
		path_l_str = self._paths2labelseqs(path_l, G)
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
		return i, self._find_all_path_as_trie(g)


	# @todo: (can be removed maybe)  this method find paths repetively, it could be faster.
	def _find_all_paths_until_length(self, G, tolabelseqs=True):
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
		# for l in range(1, self._depth + 1):
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
		for l in range(1, self._depth + 1):
			path_lplus1 = []
			for path in path_l:
				for neighbor in G[path[-1]]:
					if neighbor not in path:
						tmp = path + [neighbor]
						#					if tmp[::-1] not in path_lplus1:
						path_lplus1.append(tmp)

			all_paths += path_lplus1
			path_l = [p.copy() for p in path_lplus1]

		# for i in range(0, self._depth + 1):
		#	 new_paths = find_all_paths(G, i)
		#	 if new_paths == []:
		#		 break
		#	 all_paths.extend(new_paths)

		# consider labels
		#	print(paths2labelseqs(all_paths, G, ds_attrs, node_label, edge_label))
		#	print()
		return (
			self._paths2labelseqs(all_paths, G) if tolabelseqs else all_paths)


	def _wrapper_find_all_paths_until_length(self, tolabelseqs, itr_item):
		g = itr_item[0]
		i = itr_item[1]
		return i, self._find_all_paths_until_length(g, tolabelseqs=tolabelseqs)


	def _paths2labelseqs(self, plist, G):
		if len(self._node_labels) > 0:
			if len(self._edge_labels) > 0:
				path_strs = []
				for path in plist:
					pths_tmp = []
					for idx, node in enumerate(path[:-1]):
						pths_tmp.append(
							tuple(G.nodes[node][nl] for nl in self._node_labels)
						)
						pths_tmp.append(
							tuple(
								G[node][path[idx + 1]][el] for el in
								self._edge_labels
							)
						)
					pths_tmp.append(
						tuple(G.nodes[path[-1]][nl] for nl in self._node_labels)
					)
					path_strs.append(tuple(pths_tmp))
			else:
				path_strs = []
				for path in plist:
					pths_tmp = []
					for node in path:
						pths_tmp.append(
							tuple(G.nodes[node][nl] for nl in self._node_labels)
						)
					path_strs.append(tuple(pths_tmp))
			return path_strs
		else:
			if len(self._edge_labels) > 0:
				path_strs = []
				for path in plist:
					if len(path) == 1:
						path_strs.append(tuple())
					else:
						pths_tmp = []
						for idx, node in enumerate(path[:-1]):
							pths_tmp.append(
								tuple(
									G[node][path[idx + 1]][el] for el in
									self._edge_labels
								)
							)
						path_strs.append(tuple(pths_tmp))
				return path_strs
			else:
				return [tuple(['0' for node in path]) for path in plist]


	#			return [tuple([len(path)]) for path in all_paths]

	def _add_dummy_labels(self, Gn):
		if self._k_func is not None:
			if len(self._node_labels) == 0 or (
					len(self._node_labels) == 1 and self._node_labels[
				0] == SpecialLabel.DUMMY):
				for i in range(len(Gn)):
					nx.set_node_attributes(Gn[i], '0', SpecialLabel.DUMMY)
				self._node_labels = [SpecialLabel.DUMMY]
			if len(self._edge_labels) == 0 or (
					len(self._edge_labels) == 1 and self._edge_labels[
				0] == SpecialLabel.DUMMY):
				for i in range(len(Gn)):
					nx.set_edge_attributes(Gn[i], '0', SpecialLabel.DUMMY)
				self._edge_labels = [SpecialLabel.DUMMY]
