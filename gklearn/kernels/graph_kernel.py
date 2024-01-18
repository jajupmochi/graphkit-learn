#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:52:47 2020

@author: ljia
"""
import time
import functools
import multiprocessing

import numpy as np
import networkx as nx
# from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator  # , TransformerMixin
from sklearn.utils.validation import check_is_fitted  # check_X_y, check_array,
from sklearn.exceptions import NotFittedError

from gklearn.utils import normalize_gram_matrix, is_basic_python_type


class GraphKernel(BaseEstimator):  # , ABC):
	"""The basic graph kernel class.

	Attributes
	----------
	_graphs : list
		Stores the input graphs on fit input data.
		Default format of the list objects is `NetworkX` graphs.
		**We don't guarantee that the input graphs remain unchanged during the
		computation.**

	References
	----------
	https://ysig.github.io/GraKeL/0.1a8/_modules/grakel/kernels/kernel.html#Kernel.
	"""


	def __init__(
			self,
			parallel=None,
			n_jobs=None,
			chunksize=None,
			normalize=True,
			copy_graphs=True,  # make sure it is a full deep copy. and faster!
			verbose=2
	):
		"""`__init__` for `GraphKernel` object."""
		# @todo: the default settings of the parameters are different from those in the self.compute method.
		# 		self._graphs = None
		self.parallel = parallel
		self.n_jobs = n_jobs
		self.chunksize = chunksize
		self.normalize = normalize
		self.copy_graphs = copy_graphs
		self.verbose = verbose


	# 		self._run_time = 0
	# 		self._gram_matrix = None
	# 		self._gram_matrix_unnorm = None

	##########################################################################
	# The following is the 1st paradigm to compute kernel matrix, which is
	# compatible with `scikit-learn`.
	# -------------------------------------------------------------------
	# Special thanks to the "GraKeL" library for providing an excellent template!
	##########################################################################

	def fit(self, X, y=None):
		"""Fit a graph dataset for a transformer.

		Parameters
		----------
		X : iterable
			DESCRIPTION.

		y : None, optional
			There is no need of a target in a transformer, yet the `scikit-learn`
			pipeline API requires this parameter.

		Returns
		-------
		object
			Returns self.

		"""
		# 		self._is_tranformed = False

		# Clear any prior attributes stored on the estimator, # @todo: unless warm_start is used;
		self.clear_attributes()

		# Validate parameters for the transformer.
		self.validate_parameters()

		# Validate the input.
		self._graphs = self.validate_input(X)

		# 		self._X = X
		# 		self._kernel = self._get_kernel_instance()

		# Return the transformer.
		return self


	def transform(self, X=None, load_gm_train=False):
		"""Compute the graph kernel matrix between given and fitted data.

		Parameters
		----------
		X : TYPE
			DESCRIPTION.

		Raises
		------
		ValueError
			DESCRIPTION.

		Returns
		-------
		None.

		"""
		# If `load_gm_train`, load Gram matrix of training data.
		if load_gm_train:
			check_is_fitted(self, '_gm_train')
			self._is_transformed = True
			return self._gm_train  # @todo: copy or not?

		# Check if method "fit" had been called.
		check_is_fitted(self, '_graphs')

		# Validate the input.
		Y = self.validate_input(X)

		# Transform: compute the graph kernel matrix.
		kernel_matrix = self.compute_kernel_matrix(Y)
		self._Y = Y

		# Self transform must appear before the diagonal call on normilization.
		self._is_transformed = True
		if self.normalize:
			X_diag, Y_diag = self.diagonals()
			# Catch FloatingPointError: invalid value encountered in sqrt:
			old_settings = np.seterr(invalid='raise')
			try:
				kernel_matrix /= np.sqrt(np.outer(Y_diag, X_diag))
			except:
				raise
			finally:
				np.seterr(**old_settings)

		return kernel_matrix


	def fit_transform(
			self,
			X,
			save_gm_train: bool = False,
			save_mm_train: bool = False,
	):
		"""Fit and transform: compute Gram matrix on the same data.

		Parameters
		----------
		X : list of graphs
			Input graphs.

		Returns
		-------
		gram_matrix : numpy array, shape = [len(X), len(X)]
			The Gram matrix of X.

		"""
		self.fit(X)

		# Transform: compute Gram matrix.
		gram_matrix = self.compute_kernel_matrix()

		# Normalize.
		if self.normalize:
			self._X_diag = np.diagonal(gram_matrix).copy()
			# Catch FloatingPointError: invalid value encountered in sqrt:
			old_settings = np.seterr(invalid='raise')
			try:
				gram_matrix /= np.sqrt(np.outer(self._X_diag, self._X_diag))
			except:
				# print('Error: invalid value encountered in sqrt.')
				# print('self._X_diag =', self._X_diag)
				raise
			finally:
				np.seterr(**old_settings)

		if save_mm_train or save_gm_train:
			self._gm_train = gram_matrix

		return gram_matrix


	def get_params(
			self,
			with_graphs: bool = False,
			with_ndarray: bool = False,
			check_json_serializable: bool = True
	):
		"""Get parameters for this estimator.

		Parameters
		----------
		with_graphs : bool, optional
			Whether to include the graphs. Default: False.

		with_ndarray : bool, optional
			Whether to include the ndarray. Default: False.

		check_json_serializable : bool, optional
			Whether to check if the parameters are JSON serializable. Default: True.
			todo: maybe this needs to be checked in case some important attributes are
			removed.

		Returns
		-------
		params : dict
			Parameter names mapped to their values.

		Todos
		-----
		It may be better to seperate this method with the __str__ method.
		"""
		# loop over attributes in the object:
		params = dict()
		for key, value in self.__dict__.items():
			cur_params = dict()
			# if the attribute is a list of graphs or a graph:
			if (isinstance(value, list) and len(value) > 0 and \
			    isinstance(value[0], nx.Graph)) or \
					isinstance(value, nx.Graph):
				if with_graphs:
					# add the name(s) and params to dict:
					cur_params[key] = dict()
					cur_params[key]['name'] = value[0].__class__.__name__
					cur_params[key]['params'] = value[0].get_params()
				else:
					continue

			# if the attribute is a numpy array:
			elif isinstance(value, np.ndarray):
				if with_ndarray:
					cur_params[key] = value
				else:
					continue

			# If the attribute is a function:
			elif hasattr(value, '__call__'):
				# If it is a partial function, add its `__str__()`:
				if isinstance(value, functools.partial):
					cur_params[key] = str(value)
				# Otherwise, add its name to dict:
				else:
					cur_params[key] = value.__module__ + '.' + value.__name__

			# If the attribute is a class, add its name and params to dict:
			elif hasattr(value, '__dict__'):
				cur_params[key] = dict()
				cur_params[key]['name'] = value.__class__.__name__
				cur_params[key]['params'] = value.get_params()

			# If the attribute is a basic type, add it to dict:
			elif is_basic_python_type(value, deep=True):
				cur_params[key] = value

			# Otherwise, do nothing.
			else:
				continue

			# todo: SpecialLabel.DUMMY (e.g., COX2 + Path)

			if check_json_serializable:
				# If the current params is serializable, add it to params:
				try:
					import json
					json.dumps(cur_params)
				except TypeError:
					continue

			params[key] = cur_params[key]

		return params


	def set_params(self):
		pass


	def clear_attributes(self):
		if hasattr(self, '_X_diag'):
			delattr(self, '_X_diag')
		if hasattr(self, '_graphs'):
			delattr(self, '_graphs')
		if hasattr(self, '_Y'):
			delattr(self, '_Y')
		if hasattr(self, '_run_time'):
			delattr(self, '_run_time')
		if hasattr(self, '_test_run_time'):
			delattr(self, '_test_run_time')


	def validate_parameters(self):
		"""Validate all parameters for the transformer.

		Returns
		-------
		None.

		"""
		if self.parallel == False:
			self.parallel = None
		elif self.parallel == True:
			self.parallel = 'imap_unordered'
		elif self.parallel is not None and self.parallel != 'imap_unordered':
			raise ValueError('Parallel mode is not set correctly.')

		if self.parallel == 'imap_unordered' and self.n_jobs is None:
			self.n_jobs = multiprocessing.cpu_count()


	def validate_input(self, X):
		"""Validate the given input and raise errors if it is invalid.

		Parameters
		----------
		X : list
			The input to check. Should be a list of graph.

		Raises
		------
		ValueError
			Raise if the input is not correct.

		Returns
		-------
		X : list
			The input. A list of graph.

		"""
		if X is None:
			raise ValueError('Please add graphs before computing.')
		elif not isinstance(X, list):
			raise ValueError('Cannot detect graphs.')
		elif len(X) == 0:
			raise ValueError(
				'The graph list given is empty. No computation will be performed.'
			)

		return X


	def compute_kernel_matrix(self, Y=None):
		"""Compute the kernel matrix between a given target graphs (Y) and
		the fitted graphs (X / self._graphs) or the Gram matrix for the fitted
		graphs (X / self._graphs).

		Parameters
		----------
		Y : list of graphs, optional
			The target graphs. The default is None. If None kernel is computed
			between X and itself.

		Returns
		-------
		kernel_matrix : numpy array, shape = [n_targets, n_inputs]
			The computed kernel matrix.

		"""
		if Y is None:
			# Compute Gram matrix for self._graphs (X).
			kernel_matrix = self._compute_gram_matrix()
		# 			self._gram_matrix_unnorm = np.copy(self._gram_matrix)

		else:
			# Compute kernel matrix between Y and self._graphs (X).
			if self.parallel == 'imap_unordered':
				start_time = time.time()
				kernel_matrix = self._compute_kernel_matrix_imap_unordered(Y)

			elif self.parallel is None:
				Y_copy = ([g.copy() for g in Y] if self.copy_graphs else Y)
				graphs_copy = (
					[g.copy() for g in
					 self._graphs] if self.copy_graphs else self._graphs
				)
				start_time = time.time()
				kernel_matrix = self._compute_kernel_matrix_series(
					Y_copy, graphs_copy
				)

			self._test_run_time = time.time() - start_time
			if self.verbose:
				print(
					'Kernel matrix of size (%d, %d) built in %s seconds.'
					% (len(Y), len(self._graphs), self._test_run_time)
				)

		return kernel_matrix


	def _compute_kernel_matrix_series(self, X, Y):
		"""Compute the kernel matrix between two sets of graphs (X and Y) without parallelization.

		Parameters
		----------
		X, Y : list of graphs
			The input graphs.

		Returns
		-------
		kernel_matrix : numpy array, shape = [n_X, n_Y]
			The computed kernel matrix.

		"""
		kernel_matrix = np.zeros((len(X), len(Y)))

		for i_x, g_x in enumerate(X):
			for i_y, g_y in enumerate(Y):
				kernel_matrix[i_x, i_y] = self.pairwise_kernel(g_x, g_y)

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
			graphs = ([g.copy() for g in
			           self._graphs] if self.copy_graphs else self._graphs)
			for i, x in enumerate(graphs):
				self._X_diag[i] = self.pairwise_kernel(x, x)  # @todo: parallel?

		try:
			# If transform has happened, return both diagonals.
			check_is_fitted(self, ['_Y'])
			self._Y_diag = np.empty(shape=(len(self._Y),))
			Y = ([g.copy() for g in self._Y] if self.copy_graphs else self._Y)
			for (i, y) in enumerate(Y):
				self._Y_diag[i] = self.pairwise_kernel(y, y)  # @todo: parallel?

			return self._X_diag, self._Y_diag
		except NotFittedError:
			# Else just return both X_diag
			return self._X_diag


	# 	@abstractmethod
	def pairwise_kernel(self, x, y):
		"""Compute pairwise kernel between two graphs.

		Parameters
		----------
		x, y : NetworkX Graph.
			Graphs bewteen which the kernel is computed.

		Returns
		-------
		kernel: float
			The computed kernel.

# 		Notes
# 		-----
# 		This method is abstract and must be implemented by a subclass.

		"""
		raise NotImplementedError(
			'Pairwise kernel computation is not implemented!'
		)


	##########################################################################
	# The following is the 2nd paradigm to compute kernel matrix. It is
	# simplified and not compatible with `scikit-learn`.
	##########################################################################

	def compute(self, *graphs, **kwargs):
		self.parallel = kwargs.get('parallel', 'imap_unordered')
		self.n_jobs = kwargs.get('n_jobs', multiprocessing.cpu_count())
		self.normalize = kwargs.get('normalize', True)
		self.verbose = kwargs.get('verbose', 2)
		self.copy_graphs = kwargs.get('copy_graphs', True)
		self.save_unnormed = kwargs.get('save_unnormed', True)
		self.validate_parameters()

		# If the inputs is a list of graphs.
		if len(graphs) == 1:
			if not isinstance(graphs[0], list):
				raise Exception('Cannot detect graphs.')
			elif len(graphs[0]) == 0:
				raise Exception(
					'The graph list given is empty. No computation was performed.'
				)
			else:
				if self.copy_graphs:
					self._graphs = [
						g.copy() for g in
						graphs[0]]  # @todo: might be very slow.
				else:
					self._graphs = graphs
				self._gm_train = self._compute_gram_matrix()

				if self.save_unnormed:
					self._gram_matrix_unnorm = np.copy(self._gm_train)
				if self.normalize:
					self._gm_train = normalize_gram_matrix(self._gm_train)
				return self._gm_train, self._run_time

		elif len(graphs) == 2:
			# If the inputs are two graphs.
			if self.is_graph(graphs[0]) and self.is_graph(graphs[1]):
				if self.copy_graphs:
					G0, G1 = graphs[0].copy(), graphs[1].copy()
				else:
					G0, G1 = graphs[0], graphs[1]
				kernel = self._compute_single_kernel(G0, G1)
				return kernel, self._run_time

			# If the inputs are a graph and a list of graphs.
			elif self.is_graph(graphs[0]) and isinstance(graphs[1], list):
				if self.copy_graphs:
					g1 = graphs[0].copy()
					g_list = [g.copy() for g in graphs[1]]
					kernel_list = self._compute_kernel_list(g1, g_list)
				else:
					kernel_list = self._compute_kernel_list(
						graphs[0], graphs[1]
					)
				return kernel_list, self._run_time

			elif isinstance(graphs[0], list) and self.is_graph(graphs[1]):
				if self.copy_graphs:
					g1 = graphs[1].copy()
					g_list = [g.copy() for g in graphs[0]]
					kernel_list = self._compute_kernel_list(g1, g_list)
				else:
					kernel_list = self._compute_kernel_list(
						graphs[1], graphs[0]
					)
				return kernel_list, self._run_time

			else:
				raise Exception('Cannot detect graphs.')

		elif len(graphs) == 0 and self._graphs is None:
			raise Exception('Please add graphs before computing.')

		else:
			raise Exception('Cannot detect graphs.')


	@staticmethod
	def normalize_gm(gram_matrix):
		import warnings
		warnings.warn(
			'gklearn.kernels.graph_kernel.normalize_gm will be deprecated, use '
			'gklearn.utils.normalize_gram_matrix instead',
			DeprecationWarning
		)

		diag = gram_matrix.diagonal().copy()
		for i in range(len(gram_matrix)):
			for j in range(i, len(gram_matrix)):
				gram_matrix[i][j] /= np.sqrt(diag[i] * diag[j])
				gram_matrix[j][i] = gram_matrix[i][j]
		return gram_matrix


	def compute_distance_matrix(self):
		if self._gm_train is None:
			raise Exception(
				'Please compute the Gram matrix before computing distance matrix.'
			)
		dis_mat = np.empty((len(self._gm_train), len(self._gm_train)))
		for i in range(len(self._gm_train)):
			for j in range(i, len(self._gm_train)):
				dis = self._gm_train[i, i] + self._gm_train[j, j] - 2 * \
				      self._gm_train[i, j]
				if dis < 0:
					if dis > -1e-10:
						dis = 0
					else:
						raise ValueError('The distance is negative.')
				dis_mat[i, j] = np.sqrt(dis)
				dis_mat[j, i] = dis_mat[i, j]
		dis_max = np.max(np.max(dis_mat))
		dis_min = np.min(np.min(dis_mat[dis_mat != 0]))
		dis_mean = np.mean(np.mean(dis_mat))
		return dis_mat, dis_max, dis_min, dis_mean


	def _compute_gram_matrix(self):
		if self.parallel == 'imap_unordered':
			start_time = time.time()
			gram_matrix = self._compute_gm_imap_unordered()

		elif self.parallel is None:
			graphs = (
				[g.copy() for g in
				 self._graphs] if self.copy_graphs else self._graphs)

			# todo: this is just a temporary fix for the self loop problem.
			# Remove self loops from the graphs:
			for g in graphs:
				for node in g:
					if g.has_edge(node, node):
						g.remove_edge(node, node)

			start_time = time.time()
			gram_matrix = self._compute_gm_series(graphs)

		else:
			raise Exception('Parallel mode is not set correctly.')

		self._run_time = time.time() - start_time
		if self.verbose:
			print(
				'Gram matrix of size %d built in %s seconds.'
				% (len(self._graphs), self._run_time)
			)

		return gram_matrix


	def _compute_gm_series(self, graphs):
		raise NotImplementedError(
			'The `_compute_gm_series` method needs to be implemented by a subclass.'
		)


	def _compute_gm_imap_unordered(self, graphs):
		raise NotImplementedError(
			'The `_compute_gm_imap_unordered` method needs to be implemented by '
			'a subclass.'
		)


	def _compute_kernel_list(self, g1, g_list):
		start_time = time.time()

		if self.parallel == 'imap_unordered':
			kernel_list = self._compute_kernel_list_imap_unordered(g1, g_list)
		elif self.parallel is None:
			kernel_list = self._compute_kernel_list_series(g1, g_list)
		else:
			raise Exception('Parallel mode is not set correctly.')

		self._run_time = time.time() - start_time
		if self.verbose:
			print(
				'Graph kernel bewteen a graph and a list of %d graphs built in %s seconds.'
				% (len(g_list), self._run_time)
			)

		return kernel_list


	def _compute_kernel_list_series(self, g1, g_list):
		raise NotImplementedError(
			'The `_compute_kernel_list_series` method needs to be implemented by '
			'a subclass.'
		)


	def _compute_kernel_list_imap_unordered(self, g1, g_list):
		raise NotImplementedError(
			'The `_compute_kernel_list_imap_unordered` method needs to be '
			'implemented by a subclass.'
		)


	def _compute_single_kernel(self, g1, g2):
		start_time = time.time()

		kernel = self._compute_single_kernel_series(g1, g2)

		self._run_time = time.time() - start_time
		if self.verbose:
			print(
				'Graph kernel bewteen two graphs built in %s seconds.' % (
					self._run_time)
			)

		return kernel


	def _compute_single_kernel_series(self, g1, g2):
		raise NotImplementedError(
			'The `_compute_single_kernel_series` method needs to be implemented '
			'by a subclass.'
		)


	@staticmethod
	def is_graph(graph):
		if isinstance(graph, nx.Graph):
			return True
		if isinstance(graph, nx.DiGraph):
			return True
		if isinstance(graph, nx.MultiGraph):
			return True
		if isinstance(graph, nx.MultiDiGraph):
			return True
		return False


	@property
	def graphs(self):
		return self._graphs


	# 	@property
	# 	def parallel(self):
	# 		return self.parallel

	# 	@property
	# 	def n_jobs(self):
	# 		return self.n_jobs

	# 	@property
	# 	def verbose(self):
	# 		return self.verbose

	# 	@property
	# 	def normalize(self):
	# 		return self.normalize

	@property
	def run_time(self):
		return self._run_time


	@property
	def test_run_time(self):
		return self._test_run_time


	@property
	def gram_matrix(self):
		return self._gm_train


	@gram_matrix.setter
	def gram_matrix(self, value):
		self._gm_train = value


	@property
	def metric_matrix(self):
		return self._gm_train


	@metric_matrix.setter
	def metric_matrix(self, value):
		self._gm_train = value


	@property
	def gram_matrix_unnorm(self):
		return self._gram_matrix_unnorm


	@gram_matrix_unnorm.setter
	def gram_matrix_unnorm(self, value):
		self._gram_matrix_unnorm = value


	@property
	def n_pairs(self):
		"""
		The number of pairs of graphs between which the kernels are computed.
		"""
		try:
			check_is_fitted(self, '_gm_train')
			return len(self._gm_train) * (len(self._gm_train) + 1) / 2
		except NotFittedError:
			return None
