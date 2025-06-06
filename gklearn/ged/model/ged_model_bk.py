#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:42:30 2022

@author: ljia
"""
import sys
import multiprocessing
import time
import numpy as np
import networkx as nx
from itertools import combinations
import multiprocessing
from multiprocessing import Pool

# from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator  # , TransformerMixin
from sklearn.utils.validation import check_is_fitted  # check_X_y, check_array,
from sklearn.exceptions import NotFittedError

from gklearn.ged.model.distances import euclid_d
from gklearn.ged.util import pairwise_ged, get_nb_edit_operations
# from gklearn.utils import normalize_gram_matrix
from gklearn.utils import get_iters


# @TODO: it should be faster if creating a global env variable.
class GEDModel(BaseEstimator):  # , ABC):
	"""The graph edit distance model class compatible with `scikit-learn`.

	Attributes
	----------
	_graphs : list
		Stores the input graphs on fit input data.
		Default format of the list objects is `NetworkX` graphs.
		**We don't guarantee that the input graphs remain unchanged during the
		computation.**

	Notes
	-----
	This class uses the `gedlibpy` module to compute the graph edit distance.

	References
	----------
	https://ysig.github.io/GraKeL/0.1a8/_modules/grakel/kernels/kernel.html#Kernel.
	"""


	def __init__(
			self,
			ed_method='BIPARTITE',
			edit_cost_fun='CONSTANT',
			init_edit_cost_constants=[3, 3, 1, 3, 3, 1],
			optim_method='init',
			optim_options={'y_distance': euclid_d, 'mode': 'reg'},
			node_labels=[],
			edge_labels=[],
			parallel=None,
			n_jobs=None,
			chunksize=None,
			#			  normalize=True,
			copy_graphs=True,  # make sure it is a full deep copy. and faster!
			verbose=2
	):
		"""`__init__` for `GEDModel` object."""
		# @todo: the default settings of the parameters are different from those in the self.compute method.
		#		self._graphs = None
		self.ed_method = ed_method
		self.edit_cost_fun = edit_cost_fun
		self.init_edit_cost_constants = init_edit_cost_constants
		self.optim_method = optim_method
		self.optim_options = optim_options
		self.node_labels = node_labels
		self.edge_labels = edge_labels
		self.parallel = parallel
		self.n_jobs = (
			(multiprocessing.cpu_count() - 1) if n_jobs is None else n_jobs)
		self.chunksize = chunksize
		#		self.normalize = normalize
		self.copy_graphs = copy_graphs
		self.verbose = verbose


	#		self._run_time = 0
	#		self._gram_matrix = None
	#		self._gram_matrix_unnorm = None

	##########################################################################
	# The following is the 1st paradigm to compute GED distance matrix, which is
	# compatible with `scikit-learn`.
	##########################################################################

	def fit(self, X, y=None, **kwargs):
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
		#		self._is_tranformed = False

		# Clear any prior attributes stored on the estimator, # @todo: unless warm_start is used;
		self.clear_attributes()

		# Validate parameters for the transformer.
		self.validate_parameters()

		# Validate the input.
		self._graphs = self.validate_input(X)
		if y is not None:
			self._targets = y
		# self._targets = self.validate_input(y)

		# Compute edit cost constants.
		self.compute_edit_costs(**kwargs)

		#		self._X = X
		#		self._kernel = self._get_kernel_instance()

		# Return the transformer.
		return self


	def transform(
			self, X=None,
			return_dm_train=False,
			save_dm_test=False,
			return_dm_test=False,
			**kwargs
	):
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
		# If `return_dm_train`, return the fitted GED distance matrix of training data.
		if return_dm_train:
			check_is_fitted(self, '_dm_train')
			self._is_transformed = True
			return self._dm_train  # @TODO: copy or not?

		if return_dm_test:
			check_is_fitted(self, '_dm_test')
			return self._dm_test  # @TODO: copy or not?

		# Check if method "fit" had been called.
		check_is_fitted(self, '_graphs')

		# Validate the input.
		Y = self.validate_input(X)

		# Transform: compute the graph kernel matrix.
		dis_matrix = self.compute_distance_matrix(Y, **kwargs)
		self._Y = Y

		# Self transform must appear before the diagonal call on normalization.
		self._is_transformed = True  # @TODO: When to set this to True? When return dm test?
		# 		if self.normalize:
		# 			X_diag, Y_diag = self.diagonals()
		# 			old_settings = np.seterr(invalid='raise') # Catch FloatingPointError: invalid value encountered in sqrt.
		# 			try:
		# 				kernel_matrix /= np.sqrt(np.outer(Y_diag, X_diag))
		# 			except:
		# 				raise
		# 			finally:
		# 				np.seterr(**old_settings)

		if save_dm_test:
			self._dm_test = dis_matrix
		# If the model is retransformed and the `save_dm_test` flag is not set,
		# then remove the previously computed dm_test to prevent conflicts.
		else:
			if hasattr(self, '_dm_test'):
				delattr(self, '_dm_test')

		return dis_matrix


	def fit_transform(
			self,
			X,
			y=None,
			save_dm_train=False,
			save_mm_train: bool = False,
			**kwargs):
		"""Fit and transform: compute GED distance matrix on the same data.

		Parameters
		----------
		X : list of graphs
			Input graphs.

		Returns
		-------
		dis_matrix : numpy array, shape = [len(X), len(X)]
			The distance matrix of X.

		"""
		self.fit(X, y, **kwargs)

		# Transform: compute Gram matrix.
		dis_matrix = self.compute_distance_matrix(**kwargs)

		#		# Normalize.
		#		if self.normalize:
		#			self._X_diag = np.diagonal(gram_matrix).copy()
		#			old_settings = np.seterr(invalid='raise') # Catch FloatingPointError: invalid value encountered in sqrt.
		#			try:
		#				gram_matrix /= np.sqrt(np.outer(self._X_diag, self._X_diag))
		#			except:
		#				raise
		#			finally:
		#				np.seterr(**old_settings)

		if save_mm_train or save_dm_train:
			self._dm_train = dis_matrix
		# If the model is refitted and the `save_dm_train` flag is not set, then
		# remove the previously computed dm_train to prevent conflicts.
		else:
			if hasattr(self, '_dm_train'):
				delattr(self, '_dm_train')

		return dis_matrix


	def get_params(self):
		pass


	def set_params(self):
		pass


	def clear_attributes(self):  # @todo: update
		#		if hasattr(self, '_X_diag'):
		#			delattr(self, '_X_diag')
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
		if self.parallel is not None and self.parallel != 'imap_unordered':
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
			raise ValueError('Cannot detect graphs. The input must be a list.')
		elif len(X) == 0:
			raise ValueError(
				'The graph list given is empty. No computation will be performed.'
			)

		return X


	def compute_distance_matrix(self, Y=None, **kwargs):
		"""Compute the distance matrix between a given target graphs (Y) and
		the fitted graphs (X / self._graphs) or the distance matrix for the fitted
		graphs (X / self._graphs).

		Parameters
		----------
		Y : list of graphs, optional
			The target graphs. The default is None. If None distance is computed
			between X and itself.

		Returns
		-------
		dis_matrix : numpy array, shape = [n_targets, n_inputs]
			The computed distance matrix.

		"""
		if Y is None:
			# Compute metric matrix for self._graphs (X).
			dis_matrix = self._compute_X_distance_matrix(**kwargs)
		#			self._gram_matrix_unnorm = np.copy(self._gram_matrix)

		else:
			# Compute metric matrix between Y and self._graphs (X).
			Y_copy = ([g.copy() for g in Y] if self.copy_graphs else Y)
			graphs_copy = (
				[g.copy() for g in self._graphs]
				if self.copy_graphs else self._graphs
			)

			start_time = time.time()

			if self.parallel == 'imap_unordered':
				dis_matrix = self._compute_distance_matrix_imap_unordered(Y)

			elif self.parallel is None:
				dis_matrix = self._compute_distance_matrix_series(
					Y_copy, graphs_copy, **kwargs
				)

			self._test_run_time = time.time() - start_time

			if self.verbose:
				print(
					'Distance matrix of size (%d, %d) built in %s seconds.'
					% (len(Y), len(self._graphs), self._test_run_time)
				)

		return dis_matrix


	def _compute_distance_matrix_series(self, X, Y, **kwargs):
		"""Compute the GED distance matrix between two sets of graphs (X and Y)
		without parallelization.

		Parameters
		----------
		X, Y : list of graphs
			The input graphs.

		Returns
		-------
		dis_matrix : numpy array, shape = [n_X, n_Y]
			The computed distance matrix.

		"""
		dis_matrix = np.zeros((len(X), len(Y)))

		for i_x, g_x in enumerate(X):
			for i_y, g_y in enumerate(Y):
				dis_matrix[i_x, i_y], _ = self.compute_ged(g_x, g_y, **kwargs)

		return dis_matrix


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


	#	@abstractmethod
	def pairwise_distance(self, x, y):
		"""Compute pairwise kernel between two graphs.

		Parameters
		----------
		x, y : NetworkX Graph.
			Graphs bewteen which the kernel is computed.

		Returns
		-------
		kernel: float
			The computed kernel.

#		Notes
#		-----
#		This method is abstract and must be implemented by a subclass.

		"""
		raise NotImplementedError(
			'Pairwise kernel computation is not implemented!'
		)


	def compute_edit_costs(self, Y=None, Y_targets=None, **kwargs):
		"""Compute edit cost constants. When optimizing method is `fiited`,
		apply Jia2021's metric learning method by using a given target graphs (Y)
		the fitted graphs (X / self._graphs).

		Parameters
		----------
		Y : TYPE, optional
			DESCRIPTION. The default is None.

		Returns
		-------
		None.

		"""
		# Get or compute.
		if self.optim_method == 'random':
			self._edit_cost_constants = np.random.rand(6)

		elif self.optim_method == 'init':
			self._edit_cost_constants = self.init_edit_cost_constants

		elif self.optim_method == 'expert':
			self._edit_cost_constants = [3, 3, 1, 3, 3, 1]

		elif self.optim_method == 'fitted':  # Jia2021 method
			# Get proper inputs.
			if Y is None:
				check_is_fitted(self, ['_graphs'])
				check_is_fitted(self, ['_targets'])
				graphs = ([g.copy() for g in
				           self._graphs] if self.copy_graphs else self._graphs)
				targets = self._targets
			else:
				graphs = ([g.copy() for g in Y] if self.copy_graphs else Y)
				targets = Y_targets

			# Get optimization options.
			node_labels = self.node_labels
			edge_labels = self.edge_labels
			unlabeled = (len(node_labels) == 0 and len(edge_labels) == 0)
			repeats = kwargs.get('repeats', 1)
			from gklearn.ged.model.optim_costs import compute_optimal_costs
			self._edit_cost_constants = compute_optimal_costs(
				graphs, targets,
				node_labels=node_labels, edge_labels=edge_labels,
				unlabeled=unlabeled,
				init_costs=self.init_edit_cost_constants,
				ed_method=self.ed_method,
				edit_cost_fun=self.edit_cost_fun,
				repeats=repeats,
				rescue_optim_failure=False,
				verbose=(self.verbose >= 2),
				**self.optim_options
			)


	##########################################################################
	# The following is the 2nd paradigm to compute kernel matrix. It is
	# simplified and not compatible with `scikit-learn`.
	##########################################################################

	#	def compute(self, *graphs, **kwargs):
	#		self.parallel = kwargs.get('parallel', 'imap_unordered')
	#		self.n_jobs = kwargs.get('n_jobs', multiprocessing.cpu_count())
	#		self.normalize = kwargs.get('normalize', True)
	#		self.verbose = kwargs.get('verbose', 2)
	#		self.copy_graphs = kwargs.get('copy_graphs', True)
	#		self.save_unnormed = kwargs.get('save_unnormed', True)
	#		self.validate_parameters()

	#		# If the inputs is a list of graphs.
	#		if len(graphs) == 1:
	#			if not isinstance(graphs[0], list):
	#				raise Exception('Cannot detect graphs.')
	#			elif len(graphs[0]) == 0:
	#				raise Exception('The graph list given is empty. No computation was performed.')
	#			else:
	#				if self.copy_graphs:
	#					self._graphs = [g.copy() for g in graphs[0]] # @todo: might be very slow.
	#				else:
	#					self._graphs = graphs
	#				self._gram_matrix = self._compute_gram_matrix()

	#				if self.save_unnormed:
	#					self._gram_matrix_unnorm = np.copy(self._gram_matrix)
	#				if self.normalize:
	#					self._gram_matrix = normalize_gram_matrix(self._gram_matrix)
	#				return self._gram_matrix, self._run_time

	#		elif len(graphs) == 2:
	#			# If the inputs are two graphs.
	#			if self.is_graph(graphs[0]) and self.is_graph(graphs[1]):
	#				if self.copy_graphs:
	#					G0, G1 = graphs[0].copy(), graphs[1].copy()
	#				else:
	#					G0, G1 = graphs[0], graphs[1]
	#				kernel = self._compute_single_kernel(G0, G1)
	#				return kernel, self._run_time

	#			# If the inputs are a graph and a list of graphs.
	#			elif self.is_graph(graphs[0]) and isinstance(graphs[1], list):
	#				if self.copy_graphs:
	#					g1 = graphs[0].copy()
	#					g_list = [g.copy() for g in graphs[1]]
	#					kernel_list = self._compute_kernel_list(g1, g_list)
	#				else:
	#					kernel_list = self._compute_kernel_list(graphs[0], graphs[1])
	#				return kernel_list, self._run_time

	#			elif isinstance(graphs[0], list) and self.is_graph(graphs[1]):
	#				if self.copy_graphs:
	#					g1 = graphs[1].copy()
	#					g_list = [g.copy() for g in graphs[0]]
	#					kernel_list = self._compute_kernel_list(g1, g_list)
	#				else:
	#					kernel_list = self._compute_kernel_list(graphs[1], graphs[0])
	#				return kernel_list, self._run_time

	#			else:
	#				raise Exception('Cannot detect graphs.')

	#		elif len(graphs) == 0 and self._graphs is None:
	#			raise Exception('Please add graphs before computing.')

	#		else:
	#			raise Exception('Cannot detect graphs.')

	#	def normalize_gm(self, gram_matrix):
	#		import warnings
	#		warnings.warn('gklearn.kernels.graph_kernel.normalize_gm will be deprecated, use gklearn.utils.normalize_gram_matrix instead', DeprecationWarning)

	#		diag = gram_matrix.diagonal().copy()
	#		for i in range(len(gram_matrix)):
	#			for j in range(i, len(gram_matrix)):
	#				gram_matrix[i][j] /= np.sqrt(diag[i] * diag[j])
	#				gram_matrix[j][i] = gram_matrix[i][j]
	#		return gram_matrix

	#	def compute_distance_matrix(self):
	#		if self._gram_matrix is None:
	#			raise Exception('Please compute the Gram matrix before computing distance matrix.')
	#		dis_mat = np.empty((len(self._gram_matrix), len(self._gram_matrix)))
	#		for i in range(len(self._gram_matrix)):
	#			for j in range(i, len(self._gram_matrix)):
	#				dis = self._gram_matrix[i, i] + self._gram_matrix[j, j] - 2 * self._gram_matrix[i, j]
	#				if dis < 0:
	#					if dis > -1e-10:
	#						dis = 0
	#					else:
	#						raise ValueError('The distance is negative.')
	#				dis_mat[i, j] = np.sqrt(dis)
	#				dis_mat[j, i] = dis_mat[i, j]
	#		dis_max = np.max(np.max(dis_mat))
	#		dis_min = np.min(np.min(dis_mat[dis_mat != 0]))
	#		dis_mean = np.mean(np.mean(dis_mat))
	#		return dis_mat, dis_max, dis_min, dis_mean

	def _compute_X_distance_matrix(self, **kwargs):
		graphs = ([g.copy() for g in
		           self._graphs] if self.copy_graphs else self._graphs)

		start_time = time.time()

		if self.parallel == 'imap_unordered':
			dis_matrix = self._compute_X_dm_imap_unordered(graphs, **kwargs)
		elif self.parallel is None:
			dis_matrix = self._compute_X_dm_series(graphs, **kwargs)
		else:
			raise Exception('Parallel mode is not set correctly.')

		self._run_time = time.time() - start_time

		if self.verbose:
			print(
				'Distance matrix of size %d built in %s seconds.'
				% (len(self._graphs), self._run_time)
			)

		return dis_matrix


	def _compute_X_dm_series(self, graphs, **kwargs):
		n = len(graphs)
		dis_matrix = np.zeros((n, n))

		iterator = combinations(range(n), 2)
		len_itr = int(n * (n - 1) / 2)
		if self.verbose:
			print('Graphs in total: %d.' % len(graphs))
			print('The total # of pairs is %d.' % len_itr)
		for i, j in get_iters(
				iterator, desc='Computing distance matrix',
				file=sys.stdout, verbose=(self.verbose >= 2), length=len_itr
		):
			g1, g2 = graphs[i], graphs[j]
			dis_matrix[i, j], _ = self.compute_ged(g1, g2, **kwargs)
			dis_matrix[j, i] = dis_matrix[i, j]
		return dis_matrix


	def _compute_X_dm_imap_unordered(self, graphs, **kwargs):
		"""Compute GED distance matrix in parallel using imap_unordered.
		"""
		# This is very slow, maybe because of the Cython is involved.
		from gklearn.utils.parallel import parallel_ged_mat
		n = len(graphs)
		dis_matrix = np.zeros((n, n))
		if self.verbose:
			print('Graphs in total: %d.' % len(graphs))
			print('The total # of pairs is %d.' % int(n * (n + 1) / 2))

		do_fun = self._wrapper_compute_ged
		parallel_ged_mat(
			do_fun, dis_matrix, graphs, init_worker=_init_worker_ged_mat,
			glbv=(graphs,), n_jobs=self.n_jobs, verbose=self.verbose
		)


	def _wrapper_compute_ged(self, itr):
		i = itr[0]
		j = itr[1]
		# @TODO: repeats are not considered here.
		dis, _ = self.compute_ged(G_gn[i], G_gn[j])
		return i, j, dis


	# # imap_unordered returns an iterator of the results in the order
	# # in which the function calls are started.
	# # Note that imap_unordered may end up consuming all of the
	# # available memory if the iterable is too large.
	# n = len(graphs)
	# dis_matrix = np.zeros((n, n))
	# iterator = combinations(range(n), 2)
	# len_itr = int(n * (n + 1) / 2)
	# pool = Pool(processes=self.n_jobs)
	# for i, j in get_iters(
	# 		iterator, desc='Computing distance matrix',
	# 		file=sys.stdout, verbose=(self.verbose >= 2), length=len_itr
	# ):
	# 	g1, g2 = graphs[i], graphs[j]
	# 	dis_matrix[i, j], _ = pool.apply_async(
	# 		self.compute_ged, (g1, g2)
	# 	).get()
	# 	dis_matrix[j, i] = dis_matrix[i, j]
	# pool.close()
	# return dis_matrix

	def compute_ged(self, Gi, Gj, **kwargs):
		"""
		Compute GED between two graph according to edit_cost.
		"""
		ged_options = {
			'edit_cost': self.edit_cost_fun,
			'method': self.ed_method,
			'edit_cost_constants': self._edit_cost_constants
		}
		repeats = kwargs.get('repeats', 1)
		dis, pi_forward, pi_backward = pairwise_ged(
			Gi, Gj, ged_options, repeats=repeats
		)
		# @TODO: Better to have a if here.
		# if self.compute_n_eo:
		# 	n_eo_tmp = get_nb_edit_operations(
		# 		Gi, Gj, pi_forward, pi_backward,
		# 		edit_cost=self.edit_cost_fun,
		# 		node_labels=self.node_labels, edge_labels=self.edge_labels
		# 	)
		# else:
		# 	n_eo_tmp = None
		# return dis, n_eo_tmp
		return dis, None


	# 	def _compute_kernel_list(self, g1, g_list):
	# 		start_time = time.time()

	# 		if self.parallel == 'imap_unordered':
	# 			kernel_list = self._compute_kernel_list_imap_unordered(g1, g_list)
	# 		elif self.parallel is None:
	# 			kernel_list = self._compute_kernel_list_series(g1, g_list)
	# 		else:
	# 			raise Exception('Parallel mode is not set correctly.')

	# 		self._run_time = time.time() - start_time
	# 		if self.verbose:
	# 			print('Graph kernel bewteen a graph and a list of %d graphs built in %s seconds.'
	# 			  % (len(g_list), self._run_time))

	# 		return kernel_list

	# 	def _compute_kernel_list_series(self, g1, g_list):
	# 		pass

	# 	def _compute_kernel_list_imap_unordered(self, g1, g_list):
	# 		pass

	# 	def _compute_single_kernel(self, g1, g2):
	# 		start_time = time.time()

	# 		kernel = self._compute_single_kernel_series(g1, g2)

	# 		self._run_time = time.time() - start_time
	# 		if self.verbose:
	# 			print('Graph kernel bewteen two graphs built in %s seconds.' % (self._run_time))

	# 		return kernel

	# 	def _compute_single_kernel_series(self, g1, g2):
	# 		pass

	def is_graph(self, graph):
		if isinstance(graph, nx.Graph):
			return True
		if isinstance(graph, nx.DiGraph):
			return True
		if isinstance(graph, nx.MultiGraph):
			return True
		if isinstance(graph, nx.MultiDiGraph):
			return True
		return False


	def __repr__(self):
		return (
			f"{self.__class__.__name__}("
			f"optim_method={self.optim_method}, "
			f"ed_method={self.ed_method}, "
			f"edit_cost_fun={self.edit_cost_fun}, "
			f"node_labels={self.node_labels}, "
			f"edge_labels={self.edge_labels}, "
			f"optim_options={self.optim_options}, "
			f"init_edit_cost_constants={self.init_edit_cost_constants}, "
			f"copy_graphs={self.copy_graphs}, "
			f"parallel={self.parallel}, "
			f"n_jobs={self.n_jobs}, "
			f"verbose={self.verbose}, "
			f"normalize={self.normalize}, "
			f"run_time={self.run_time}"
			f")"
		)


	@property
	def graphs(self):
		return self._graphs


	#	@property
	#	def parallel(self):
	#		return self.parallel

	#	@property
	#	def n_jobs(self):
	#		return self.n_jobs

	#	@property
	#	def verbose(self):
	#		return self.verbose

	#	@property
	#	def normalize(self):
	#		return self.normalize

	@property
	def run_time(self):
		return self._run_time


	@property
	def test_run_time(self):
		return self._test_run_time


	@property
	def dis_matrix(self):
		return self._dm_train

	@dis_matrix.setter
	def dis_matrix(self, value):
		self._dm_train = value


	@property
	def metric_matrix(self):
		return self._dm_train


	@metric_matrix.setter
	def metric_matrix(self, value):
		self._dm_train = value


	@property
	def edit_cost_constants(self):
		return self._edit_cost_constants


# 	@property
# 	def gram_matrix_unnorm(self):
# 		return self._gram_matrix_unnorm

# 	@gram_matrix_unnorm.setter
# 	def gram_matrix_unnorm(self, value):
# 		self._gram_matrix_unnorm = value

	@property
	def n_pairs(self):
		"""
		The number of pairs of graphs between which the GEDs are computed.
		"""
		try:
			check_is_fitted(self, '_dm_train')
			return len(self._dm_train) * (len(self._dm_train) - 1) / 2
		except NotFittedError:
			return None


def _init_worker_ged_mat(gn_toshare):
	global G_gn
	G_gn = gn_toshare
