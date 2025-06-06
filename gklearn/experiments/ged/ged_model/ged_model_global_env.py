"""
ged_model_global_env.py

The GEDModel class using a GEDEnv as a global environment inside the class for testing purposes.

@Author: jajupmochi
@Date: June 4 2025
"""
import multiprocessing
import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from itertools import combinations, product
from multiprocessing import shared_memory

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from gklearn.experiments.ged.ged_model.profile_ged_model import INFO_TAG
from gklearn.ged.model.distances import euclid_d
from gklearn.ged.util.util import ged_options_to_string
from gklearn.utils import get_iters


class GEDModel(BaseEstimator):  # , ABC):
	"""The graph edit distance model class compatible with `scikit-learn`.

	Attributes
	----------
	_graphs : list
		Stores the input graphs on fit input data.
		Default format of the list objects is `NetworkX` graphs.
		**We don't guarantee that the input graphs remain unchanged during the computation.**

	Notes
	-----
	This class uses the `gedlibpy` module to compute the graph edit distance.

	References
	----------
	https://ysig.github.io/GraKeL/0.1a8/_modules/grakel/kernels/kernel.html#Kernel.
	"""


	def __init__(
			self,
			env_type: str | None = None,
			ed_method='BIPARTITE',
			edit_cost_fun='CONSTANT',
			init_edit_cost_constants=[3, 3, 1, 3, 3, 1],
			edit_cost_config: dict = {},
			optim_method='init',
			optim_options={'y_distance': euclid_d, 'mode': 'reg'},
			ged_init_options: dict | None = None,
			node_labels=[],
			edge_labels=[],
			parallel=None,
			n_jobs=None,
			chunksize=None,
			#			  normalize=True,
			copy_graphs=True,  # make sure it is a full deep copy. and faster!
			verbose=2
	):
		"""`__init__` for `GEDModel` object.

		Parameters
		----------
		env_type : str, optional
			The type of the GED environment. Default is None. If None, try to determine
			the type automatically based on the given graph node / edge labels.

			Available types are:

			- 'attr': Attribute-based environment (with complex node and edge labels).
			Each node or edge can have multiple key-value label pairs, and each value can
			be of the following types: int, float, str, list/np.ndarray of int or float.
			This is the default type if no node or edge labels are provided.

			- 'gxl' or 'str': GXLLabel environment (with string labels). Each node or
			edge can have multiple key-value label pairs, but all values must be strings.
			The type will be set to GXL only if at least one node or edge label is
			provided.
		"""
		# @todo: the default settings of the parameters are different from those in the self.compute method.
		#		self._graphs = None
		self.env_type = env_type
		self.ed_method = ed_method
		self.edit_cost_fun = edit_cost_fun
		self.init_edit_cost_constants = init_edit_cost_constants
		self.edit_cost_config = edit_cost_config
		self.optim_method = optim_method
		self.optim_options = optim_options
		self.ged_init_options = ged_init_options
		self.node_labels = node_labels
		self.edge_labels = edge_labels
		self.parallel = parallel
		self.n_jobs = ((multiprocessing.cpu_count() - 1) if n_jobs is None else n_jobs)
		self.chunksize = chunksize
		#		self.normalize = normalize
		self.copy_graphs = copy_graphs
		self.verbose = verbose

		self._ged_env = None  # The GED environment to use for the model.
		self._graphs = None  # The input graphs to the model.
		self._is_transformed = False  # Whether the model has been transformed.
		self._run_time = 0  # The run time of the last computation.
		self._Y = None  # The target graphs for the model.
		self._dm_train = None  # The distance matrix of the training data.
		self._dm_test = None  # The distance matrix of the test data.
		self._edit_cost_constants = None  # The edit cost constants for the model.
		self._X_diag = None  # The diagonal of the metric matrix for the training data (0's in this case).
		self._Y_diag = None  # The diagonal of the metric matrix for the test data (0's in this case).
		self._targets = None  # The targets for the model, if any.

		self.env_stats = {}  # Store environment stats for the model.


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

		kwargs : dict, optional
			Additional parameters for the transformer. The following parameters can be included:

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

		# Create the GED environment if not set:
		# Only do this if no parallelization will be used. Otherwise, a separate GEDEnv will be
		# created in each worker in transforming.
		# todo: we plan to refactor this in the future for better performance.
		if self.parallel is None:
			# `self._edit_cost_constants` is needed from `self.compute_edit_costs` to initialize the
			# GED environment.
			self._ged_env, env_setting_time = self.create_and_setup_ged_env(
				self.env_type, graph=X[0],
				**{
					'ed_method': self.ed_method,
					'edit_cost_fun': self.edit_cost_fun,
					'edit_cost_constants': self._edit_cost_constants,
					'edit_cost_config': self.edit_cost_config,
				}
			)
			self.env_stats['env_setting_time'] = env_setting_time
			# Add graphs to the environment:
			graphs_adding_time = self.add_graphs_to_ged_env(
				self._graphs, self._ged_env, self.verbose, **{'copy_graphs': self.copy_graphs}
			)
			self.env_stats['graphs_adding_time'] = graphs_adding_time

		#		self._X = X
		#		self._kernel = self._get_kernel_instance()

		# Return the transformer.
		return self


	def transform(
			self,
			X=None,
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
			**kwargs
	):
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
			self._run_time = 0
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
		if self.parallel is not None and self.parallel not in [
			'imap_unordered', 'multiprocessing', 'joblib', 'concurrent'
		]:
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
			dis_matrix = self._compute_self_distance_matrix(**kwargs)

		else:
			# This will be done when loading the graphs into the GEDEnv.
			# # Compute metric matrix between Y and self._graphs (X).
			# Y_copy = ([g.copy() for g in Y] if self.copy_graphs else Y)
			# graphs_copy = (
			# 	[g.copy() for g in self._graphs]
			# 	if self.copy_graphs else self._graphs
			# )

			dis_matrix = self._compute_cross_distance_matrix(Y, **kwargs)

		return dis_matrix


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
		# todo: this function is not optimized to use global environment.
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


	# %% Self distance matrix computation methods:

	def _compute_self_distance_matrix(self, **kwargs):
		# # Useless if graphs were loaded into GEDEnv beforehand:
		# graphs = ([g.copy() for g in self._graphs] if self.copy_graphs else self._graphs)

		start_time = time.time()

		# if self.parallel == 'imap_unordered':
		#     dis_matrix = self._compute_X_dm_imap_unordered(graphs, **kwargs)
		if self.parallel in ['imap_unordered', 'joblib', 'concurrent', 'multiprocessing']:
			dis_matrix = self._compute_self_distance_matrix_parallel(**kwargs)
		elif self.parallel is None:
			# dis_matrix = self._compute_X_dm_series(graphs, **kwargs)
			dis_matrix = self._compute_self_distance_matrix_series(**kwargs)
		else:
			raise Exception('Parallel mode is not set correctly.')

		self._run_time += time.time() - start_time

		if self.verbose:
			print(
				'Distance matrix of size %d built in %s seconds.'
				% (len(self._graphs), self._run_time)
			)

		return dis_matrix


	def _compute_self_distance_matrix_series(self, **kwargs):
		# We put the initialization of the GED environment here for these reasons:
		# 1. To process the computation of costs between labels separately for series and parallel mode.
		# 2. To include the time of this initialization in the total run time.
		# 3. For cross distance matrix, target graphs (Y) need to be added to the environment.
		eager_label_cost_computing_time = self.init_ged_env_and_method(
			self._ged_env, **{'ged_init_options': self.ged_init_options}
		)
		if eager_label_cost_computing_time is not None:
			self.env_stats['eger_label_cost_computing_time'] = eager_label_cost_computing_time

		graph_ids = self._ged_env.get_all_graph_ids()
		n = len(graph_ids)
		if n != len(self._graphs):
			raise ValueError(
				f'Number of graphs in the GEDEnv ({n}) does not match '
				f'number of input graphs in the GEDModel ({len(self._graphs)}).'
			)

		dis_matrix = np.zeros((n, n))
		iterator = combinations(range(n), 2)
		len_itr = int(n * (n - 1) / 2)
		if self.verbose:
			print('Graphs in total: %d.' % n)
			print('The total # of pairs is %d.' % len_itr)
		self.env_stats['ged_computing_time'] = []
		for i, j in get_iters(
				iterator, desc='Computing distance matrix',
				file=sys.stdout, verbose=(self.verbose >= 2), length=len_itr
		):
			gid1, gid2 = graph_ids[i], graph_ids[j]
			dis_matrix[i, j], stats = GEDModel.pairwise_ged_with_gids(
				gid1, gid2, self._ged_env, self._graphs, **kwargs
			)
			dis_matrix[j, i] = dis_matrix[i, j]
			self.env_stats['ged_computing_time'].append(stats['ged_computing_time'])
		return dis_matrix


	# todo: this is not refactored yet.
	def _compute_self_distance_matrix_parallel(self, **kwargs):
		"""
		Highly optimized parallelized version of distance matrix computation between graphs.

		Parameters:
		-----------
		graphs : list
			List of graph objects to compute pairwise distances
		n_jobs : int, default=-1
			Number of parallel jobs. -1 means using all available cores.
		chunk_size : int, default=None
			Number of tasks per chunk. If None, will be auto-calculated.
		memory_limit : str or int, default='auto'
			Memory limit per worker in MB or 'auto' to determine automatically.
		method : str, default='joblib'
			Parallelization backend: 'joblib', 'concurrent', or 'multiprocessing'

		Returns:
		--------
		np.ndarray
			Distance matrix of shape (n, n)
		"""
		n = len(self._graphs)

		# Get all pairs of indices
		pairs = list(combinations(range(n), 2))
		len_itr = len(pairs)

		n_jobs = self.n_jobs
		chunksize = self.chunksize
		method = self.parallel
		memory_limit = kwargs.get('memory_limit', 'auto')

		if self.verbose:
			print('Graphs in total: %d.' % n)
			print('The total # of pairs is %d.' % len_itr)

		# Determine the number of processes:
		if n_jobs == -1:
			n_jobs = os.cpu_count() - 1
		n_jobs = min(n_jobs, os.cpu_count(), len_itr)

		# Auto-calculate optimal chunk size if not provided
		if chunksize is None:
			# # this seems to be slightly faster when using `test_ged_model.py`
			# # with 100 graphs (0.0012 s vs 0.0016 s per pair). Yet gets slower with
			# # larger number of graphs (e.g., 1000) (~ 31 mins vs ~ 40 mins in total).
			# if len_itr < 100 * n_jobs:
			#     chunksize = int(len_itr / n_jobs) + 1
			# else:
			#     chunksize = 100

			# Balancing chunk size: larger chunks reduce overhead but limit load balancing
			# A good heuristic is sqrt(len_itr / n_jobs) * 4
			chunksize = max(1, int(np.sqrt(len_itr / n_jobs) * 4))

		if self.verbose >= 2:
			print(
				f"Running with {n_jobs} parallel processes and chunk size of {chunksize}"
			)

		# # For networkx graphs, we need to use a Manager to share them between processes:
		# with Manager() as manager:
		# 	# Create a managed shared list for the graphs
		# 	# todo:
		# 	# 1. This operation will serialize the graphs, which will make a deep copy of each graph,
		# 	# so it is not efficient.
		# 	#
		# 	# 2. When using multiprocessing.Manager to share graphs, a separate manager process is launched
		# 	# to hold the shared objects. Accessing these shared graphs from other processes involves
		# 	# serialization (pickling), inter-process communication (IPC), and deserialization (unpickling),
		# 	# which can be very costly for large NetworkX graphs.
		# 	#
		# 	# In contrast, if we use per-process global variables initialized via init_worker(),
		# 	# each process gets a local copy of the graph data, which avoids the IPC overhead,
		# 	# but requires duplicating memory (one full copy per worker).
		# 	#
		# 	# To compare the overheads:
		# 	#   - Using a Manager:
		# 	#     -- Every access to a graph (e.g., shared_graphs[i]) involves: pickle → IPC → unpickle.
		# 	#     -- Graphs are not truly shared in memory; they are proxied through the manager process.
		# 	#     Since we then create GEDEnv graphs, so no more pickling is needed.
		# 	#
		# 	#   - Using global variables in worker init:
		# 	#     -- Graphs are copied once to each worker during process start (via memory fork or pickle).
		# 	#     -- After that, all access is purely local (no IPC, no further serialization).
		# 	#     -- This is faster at runtime but uses more memory.
		# 	#
		# 	# 3. Since manager uses a proxy object, it may cause issues when trying to modify
		# 	# the graphs.
		# 	shared_graphs = manager.list(self._graphs)

		# Get a function reference to compute_ged that can be pickled
		# Using a Python trick to make the instance method picklable
		compute_ged_func = partial(GEDModel.pairwise_ged_with_gids_parallel, **kwargs)

		# Create a shared memory array for results
		with numpy_shared_memory((n, n), dtype=np.float64) as (dis_matrix, shm_name):

			# Create a partial function with fixed arguments - must use module-level function
			worker = partial(
				self._process_pair_worker,
				shm_name=shm_name,
				matrix_shape=(n, n),
				compute_ged_func=compute_ged_func,
				**kwargs
			)

			try:
				# Three different parallelization options for different scenarios
				if method == 'joblib':
					raise NotImplementedError(
						'Joblib parallelization is not implemented yet. '
						'Please use "multiprocessing".'
					)

				elif method == 'concurrent':
					# Option 2: ProcessPoolExecutor - cleaner API, slightly faster for CPU-bound tasks
					# Use thread instead of the process to support shared memory for pre-created
					# Cython objects:
					raise NotImplementedError(
						'concurrent parallelization is not implemented yet. '
						'Please use "multiprocessing".'
					)
				# if self.verbose >= 2:
				# 	print(f'Using ThreadPoolExecutor.')
				#
				# with ThreadPoolExecutor(max_workers=n_jobs) as executor:
				# 	futures = [executor.submit(worker, pair) for pair in pairs]
				#
				# 	# Track progress if verbose
				# 	if self.verbose >= 2:
				# 		results = []
				# 		# When `as_completed` is used, the order of results is not guaranteed:
				# 		for f in tqdm(
				# 				as_completed(futures), total=len(futures),
				# 				# futures, total=len(futures),
				# 				desc='Computing distance matrix', file=sys.stdout
				# 		):
				# 			results.append(f.result())
				# 	else:
				# 		results = [f.result() for f in as_completed(futures)]

				# # This does not guarantee the order of results:
				# self.env_stats['ged_computing_time'] = [
				# 	stats['ged_computing_time'] for _, _, _, stats in results
				# ]

				elif method in ['imap_unordered' or 'multiprocessing']:
					# Option 3: multiprocessing.Pool with imap_unordered - more control, classic approach
					# Does not work with pre-created GEDEnv Cython objects:
					# TypeError: no default __reduce__ due to non-trivial __cinit__
					# So create a GEDEnv for each worker during the initialization.
					# todo: maybe it is better to
					# parallelize directly in C++ with pybind with e.g., openmp

					if self.verbose >= 2:
						print(f'Using multiprocessing imap_unordered.')

					init_kwargs = {
						'ed_method': self.ed_method,
						'edit_cost_fun': self.edit_cost_fun,
						'edit_cost_constants': self._edit_cost_constants,
						'edit_cost_config': self.edit_cost_config,
						'ged_init_options': self.ged_init_options,
						'copy_graphs': False,
						# Do not copy graphs here, they are already copied in the worker
					}


					# todo: we can actually control the part of graphs that each worker will process,
					# but it is not worth the effort for now.
					def init_worker_self_metric_matrix(graphs):
						"""Initialize each worker process with a GED environment"""
						global g_ged_env  # <- This will be created for each worker
						global g_graphs
						global g_env_stats
						global g_stats_reported
						g_graphs = graphs  # Set the graphs for the worker
						g_stats_reported = False  # Reset the stats reported flag
						(
							g_ged_env, env_setting_time, graphs_adding_time,
							eager_label_cost_computing_time
						) = GEDModel.create_and_init_ged_env_for_parallel(g_graphs, **init_kwargs)
						g_env_stats = {
							'env_setting_time': env_setting_time,
							'graphs_adding_time': graphs_adding_time
						}
						if eager_label_cost_computing_time is not None:
							g_env_stats[
								'eger_label_cost_computing_time'] = eager_label_cost_computing_time


					with multiprocessing.Pool(
							processes=n_jobs, initializer=init_worker_self_metric_matrix,
							initargs=(self._graphs,)
					) as pool:
						if self.verbose >= 2:
							results = list(
								tqdm(
									pool.imap_unordered(worker, pairs, chunksize=chunksize),
									total=len_itr,
									desc='Computing distance matrix',
									file=sys.stdout
								)
							)
						else:
							results = list(
								pool.imap_unordered(worker, pairs, chunksize=chunksize)
							)

						stats = [stats for _, _, _, stats, _ in results]
						init_stats = [init_stats for _, _, _, _, init_stats in results if
						              init_stats is not None]
						if len(init_stats) != n_jobs:
							raise ValueError(
								f'Number of init_stats ({len(init_stats)}) does not match '
								f'number of workers ({n_jobs}).'
							)
						else:
							print(f'Number of init_stats: {len(init_stats)}.')
						for s in init_stats:
							for k, v in s.items():
								if f'{k}_parallel' not in self.env_stats:
									self.env_stats[f'{k}_parallel'] = []
								self.env_stats[f'{k}_parallel'].append(v)
						# print(stats)
						for s in stats:
							for k, v in s.items():
								if f'{k}_parallel' not in self.env_stats:
									self.env_stats[f'{k}_parallel'] = []
								self.env_stats[f'{k}_parallel'].append(v)
				# print(self.env_stats)

				else:
					raise ValueError(
						f"Unsupported parallelization method: {method}."
					)

				# Copy the result from shared memory to a regular numpy array
				result = dis_matrix.copy()

			except Exception as e:
				# Make sure we log any errors that occur during parallel execution
				if self.verbose:
					print(f"Error during parallel execution: {e}.")
				raise

		# At this point, the Manager will automatically clean up shared resources

		return result


	@staticmethod
	def _process_pair_worker(
			pair, shm_name, matrix_shape,
			compute_ged_func, **kwargs
	):
		"""Worker function that processes a pair of graphs and updates the shared matrix.
		Must be defined at module level to be picklable."""
		# # test only:
		# print(f'[{multiprocessing.current_process().name}] Processing pair: {pair}.')

		i, j = pair

		try:
			# Access the shared memory
			existing_shm = shared_memory.SharedMemory(name=shm_name)
			shared_matrix = np.ndarray(
				matrix_shape, dtype=np.float64, buffer=existing_shm.buf
			)

			# Compute distance using the function reference
			distance, stats, init_stats = compute_ged_func(i, j, **kwargs)

			# Update the matrix
			shared_matrix[i, j] = distance
			shared_matrix[j, i] = distance

		finally:
			# Clean up local shared memory reference
			if 'existing_shm' in locals():
				existing_shm.close()

		return i, j, distance, stats, init_stats  # Return for progress tracking


	# %% Cross distance matrix computation methods:


	def _compute_cross_distance_matrix(self, graphs_t: nx.Graph, **kwargs):
		start_time = time.time()

		if self.parallel in ['imap_unordered', 'joblib', 'concurrent', 'multiprocessing']:
			dis_matrix = self._compute_distance_matrix_parallel_unified(
				self._graphs, graphs_t, **kwargs
			)

		elif self.parallel is None:
			dis_matrix = self._compute_cross_distance_matrix_series(
				self._graphs, graphs_t, **kwargs
			)
		else:
			raise Exception('Parallel mode is not set correctly.')

		self._run_time += time.time() - start_time

		if self.verbose:
			print(
				'Distance matrix of size (%d, %d) built in %s seconds.'
				% (len(graphs_t), len(self._graphs), self._run_time)
			)

		return dis_matrix


	def _compute_cross_distance_matrix_series(
			self, graphs_f: list[nx.Graph], graphs_t: list[nx.Graph], **kwargs):
		"""Compute the GED distance matrix between two sets of graphs (X and Y)
		without parallelization.

		Parameters
		----------
		graphs_f : list of graphs
			The fitted graphs (X / self._graphs).

		graphs_t : list of graphs
			The target graphs (Y).

		Returns
		-------
		dis_matrix : numpy array, shape = [n_Y, n_X]
			The computed distance matrix.
		"""
		# Add graphs to the environment:
		graphs_adding_time = self.add_graphs_to_ged_env(
			graphs_t, self._ged_env, self.verbose, **{'copy_graphs': self.copy_graphs}
		)
		self.env_stats['graphs_adding_time'] += graphs_adding_time
		# We put the initialization of the GED environment here for these reasons:
		# 1. To process the computation of costs between labels separately for series and parallel mode.
		# 2. To include the time of this initialization in the total run time.
		# 3. For cross distance matrix, target graphs (Y) need to be added to the environment.
		eager_label_cost_computing_time = self.init_ged_env_and_method(
			self._ged_env, **{'ged_init_options': self.ged_init_options}
		)
		if eager_label_cost_computing_time is not None:
			self.env_stats['eger_label_cost_computing_time'] = eager_label_cost_computing_time

		n_f = len(graphs_f)
		n_t = len(graphs_t)
		n_graphs_in_env = self._ged_env.get_num_graphs()
		if n_graphs_in_env != n_f + n_t:
			raise ValueError(
				f'Number of graphs in the GEDEnv ({n_graphs_in_env}) does not match '
				f'the total number of fitted and target graphs in the GEDModel ({n_f} + {n_t} = {n_f + n_t}).'
			)

		# Initialize distance matrix with zeros
		dis_matrix = np.zeros((n_t, n_f))
		iterator = product(range(n_f), range(n_t))
		len_itr = n_f * n_t
		if self.verbose:
			print(f'Computing distances between {n_t} and {n_f} graphs.')
			print(f'The total # of pairs is {len_itr}.')

		self.env_stats['ged_computing_time'] = []
		for i_f, j_t in get_iters(
				iterator, desc='Computing distance matrix',
				file=sys.stdout, verbose=(self.verbose >= 2), length=len_itr
		):
			gid_f, gid_t = i_f, j_t
			dis_matrix[j_t, i_f], stats = self.pairwise_ged_with_gids(
				gid_f, gid_t, self._ged_env, graphs_f + graphs_t, **kwargs
			)
			self.env_stats['ged_computing_time'].append(stats['ged_computing_time'])

		return dis_matrix


	def _compute_distance_matrix_parallel_unified(
			self, graphs_f, graphs_t: nx.Graph | None = None, **kwargs
	):
		"""Compute the GED distance matrix between two sets of graphs (X and Y)
		with parallelization.

		Parameters
		----------
		graphs_f : list of graphs
			The fitted graphs (X).

		graphs_t : list of graphs
			The target graphs (Y). If None, the distance is computed between
			the fitted graphs (X) and itself.


		Returns
		-------
		dis_matrix : numpy array, shape = [n_Y, n_X]
			The computed distance matrix.

		References
		----------
		This method is written with the help of the Claude 3.7 Sonnet AI, accessed on 2025.05.15.

		todo: this can be merged with the _compute_X_dm_parallel method.
		"""
		# Handle the case where graphs2 is not provided
		is_same_set = graphs_t is None
		if is_same_set:
			graphs_t = graphs_f

		n_f = len(graphs_f)
		n_t = len(graphs_t)

		# Get all pairs of indices to compute
		if is_same_set:
			# Only compute the upper triangular portion for efficiency when comparing within same set
			pairs = list(combinations(range(n_f), 2))
		else:
			# Compute all pairs when comparing between different sets:
			# Notice this has different order (fiited / col first) as the matrix (target / row first):
			pairs = list(product(range(n_f), range(n_t)))

		len_itr = len(pairs)

		n_jobs = self.n_jobs
		chunksize = self.chunksize
		method = self.parallel
		# memory_limit = kwargs.get('memory_limit', 'auto')

		if self.verbose:
			if is_same_set:
				print(f'Graphs in total: {n_f}.')
			else:
				print(f'Computing distances between {n_t} and {n_f} graphs.')
			print(f'The total # of pairs is {len_itr}.')

		# Determine the number of workers:
		if n_jobs == -1 or n_jobs is None:
			n_jobs = os.cpu_count() - 1
		n_jobs = min(n_jobs, os.cpu_count(), len_itr)

		# Auto-calculate optimal chunk size if not provided
		if chunksize is None:
			# # this seems to be slightly faster when using `test_ged_model.py`
			# # with 100 graphs (0.0012 s vs 0.0016 s per pair). Yet gets slower with
			# # larger number of graphs (e.g., 1000) (~ 31 mins vs ~ 40 mins in total).
			# if len_itr < 100 * n_jobs:
			#     chunksize = int(len_itr / n_jobs) + 1
			# else:
			#     chunksize = 100

			# Balancing chunk size: larger chunks reduce overhead but limit load balancing
			# A good heuristic is sqrt(len_itr / n_jobs) * 4
			chunksize = max(1, int(np.sqrt(len_itr / n_jobs) * 4))

		if self.verbose >= 2:
			print(
				f"Running with {n_jobs} parallel processes and chunk size of {chunksize}..."
			)

		# Get a function reference to compute_ged that can be pickled
		# Using a Python trick to make the instance method picklable
		compute_ged_func = partial(
			GEDModel.pairwise_ged_with_gids_parallel, is_same_set=is_same_set, **kwargs
		)

		# Create a shared memory array for results
		with numpy_shared_memory((n_t, n_f), dtype=np.float64) as (dis_matrix, shm_name):
			# Create a partial function with fixed arguments - MUST NOT use
			# inline function here, as it won't be picklable:
			worker = partial(
				self._process_pair_worker_unified,
				shm_name=shm_name,
				matrix_shape=(n_t, n_f),
				compute_ged_func=compute_ged_func,
				is_same_set=is_same_set,
				**kwargs
			)

			try:
				# Three different parallelization options for different scenarios
				if method == 'joblib':
					raise NotImplementedError(
						'Joblib parallelization is not implemented yet. '
						'Please use "multiprocessing".'
					)

				elif method == 'concurrent':
					# Option 2: ProcessPoolExecutor - cleaner API, slightly faster for CPU-bound tasks
					raise NotImplementedError(
						'concurrent parallelization is not implemented yet. '
						'Please use "multiprocessing".'
					)

				elif method in ['imap_unordered' or 'multiprocessing']:
					# Option 3: multiprocessing.Pool with imap_unordered - more control, classic approach
					if self.verbose >= 2:
						print(f'Using multiprocessing imap_unordered.')

					init_kwargs = {
						'ed_method': self.ed_method,
						'edit_cost_fun': self.edit_cost_fun,
						'edit_cost_constants': self._edit_cost_constants,
						'edit_cost_config': self.edit_cost_config,
						'ged_init_options': self.ged_init_options,
						'copy_graphs': False,
						# Do not copy graphs here, they are already copied in the worker
					}


					def init_worker_cross_metric_matrix(graphs_f, graphs_t):
						"""Initialize each worker process with a GED environment"""
						global g_ged_env  # <- This will be created for each worker
						global g_graphs_f
						global g_graphs_t
						global g_env_stats
						global g_stats_reported
						g_graphs_f = graphs_f  # Set the graphs for the worker
						g_graphs_t = graphs_t
						g_stats_reported = False  # Reset the stats reported flag
						(
							g_ged_env, env_setting_time, graphs_adding_time,
							eager_label_cost_computing_time
						) = GEDModel.create_and_init_ged_env_for_parallel(
							g_graphs_f + g_graphs_t, **init_kwargs
						)
						g_env_stats = {
							'env_setting_time': env_setting_time,
							'graphs_adding_time': graphs_adding_time
						}
						if eager_label_cost_computing_time is not None:
							g_env_stats[
								'eger_label_cost_computing_time'] = eager_label_cost_computing_time


					with multiprocessing.Pool(
							processes=n_jobs, initializer=init_worker_cross_metric_matrix,
							initargs=(graphs_f, graphs_t,)
					) as pool:
						if self.verbose >= 2:
							results = list(
								tqdm(
									pool.imap_unordered(worker, pairs, chunksize=chunksize),
									total=len_itr,
									desc='Computing distance matrix',
									file=sys.stdout
								)
							)
						else:
							results = list(pool.imap_unordered(worker, pairs, chunksize=chunksize))

						stats = [stats for _, _, _, stats, _ in results]
						init_stats = [
							init_stats for _, _, _, _, init_stats in results if
							init_stats is not None
						]
						if len(init_stats) != n_jobs:
							raise ValueError(
								f'Number of init_stats ({len(init_stats)}) does not match '
								f'number of workers ({n_jobs}).'
							)
						else:
							print(f'Number of init_stats: {len(init_stats)}.')
						for s in init_stats:
							for k, v in s.items():
								if f'{k}_parallel' not in self.env_stats:
									self.env_stats[f'{k}_parallel'] = []
								self.env_stats[f'{k}_parallel'].append(v)
						# print(stats)
						for s in stats:
							for k, v in s.items():
								if f'{k}_parallel' not in self.env_stats:
									self.env_stats[f'{k}_parallel'] = []
								self.env_stats[f'{k}_parallel'].append(v)
				# print(self.env_stats)

				else:
					raise ValueError(
						f"Unsupported parallelization method: {method}."
					)

				# Copy the result from shared memory to a regular numpy array
				result = dis_matrix.copy()

			except Exception as e:
				# Make sure we log any errors that occur during parallel execution
				if self.verbose:
					print(f"Error during parallel execution: {e}.")
				raise

		# At this point, the Manager will automatically clean up shared resources

		return result


	# %% Parallelization methods:

	@staticmethod
	def create_and_init_ged_env_for_parallel(graphs: list[nx.Graph], **kwargs):
		"""Create and initialize a GED environment for parallel processing."""
		# Create a new GEDEnv instance for each worker
		ged_env, env_setting_time = GEDModel.create_and_setup_ged_env(graph=graphs[0], **kwargs)
		# print(f'[{multiprocessing.current_process().name}] ')
		# print(ged_env)

		# Add all graphs to the environment:
		graphs_adding_time = GEDModel.add_graphs_to_ged_env(graphs, ged_env, verbose=0, **kwargs)
		# print('fnished adding graphs to the GEDEnv in worker.')
		# print(ged_env.get_all_graph_ids())
		eager_label_cost_computing_time = GEDModel.init_ged_env_and_method(ged_env, **kwargs)

		graph_ids = ged_env.get_all_graph_ids()
		n = len(graph_ids)
		if n != len(graphs):
			raise ValueError(
				f'Number of graphs in the GEDEnv ({n}) does not match '
				f'number of graphs set from GEDModel to the worker ({len(graphs)}).'
			)
		return ged_env, env_setting_time, graphs_adding_time, eager_label_cost_computing_time


	# @staticmethod
	# def _thread_pair_worker(pair, distance_matrix, compute_ged_func, **kwargs):
	# 	"""
	# 	Worker function that processes a pair of graphs and updates the shared matrix.
	# 	Used for processing pre-created Cython objects (GEDEnv) in threads (with shared memory).
	# 	Please make sure that the Cython objects are thread-safe!!!
	# 	"""
	# 	# # test only:
	# 	# print(f'[{threading.current_thread().name}] Processing pair: {pair}.')
	# 	# # Sleep for 1 second to simulate work:
	# 	# time.sleep(1)
	#
	# 	i, j = pair
	#
	# 	# Compute distance using the function reference
	# 	distance, stats = compute_ged_func(i, j, **kwargs)
	# 	# Update the matrix
	# 	distance_matrix[i, j] = distance
	# 	distance_matrix[j, i] = distance
	#
	# 	return i, j, distance, stats  # Return for progress tracking

	@staticmethod
	def _process_pair_worker_unified(
			pair, shm_name, matrix_shape, compute_ged_func, is_same_set=True, **kwargs
	):
		"""Worker function that processes a pair of graphs and updates the shared matrix.
		Must be defined at module level to be picklable."""
		i_f, j_t = pair  # Indices of the fitted and target graphs in the original lists in GEDModel

		try:
			# Access the shared memory
			existing_shm = shared_memory.SharedMemory(name=shm_name)
			shared_matrix = np.ndarray(matrix_shape, dtype=np.float64, buffer=existing_shm.buf)

			# Compute distance using the function reference
			distance, stats, init_stats = compute_ged_func(i_f, j_t, **kwargs)

			# Update the matrix
			shared_matrix[j_t, i_f] = distance

			# If computing within the same set, update symmetric position:
			if is_same_set and i_f != j_t:
				shared_matrix[i_f, j_t] = distance

		finally:
			# Clean up local shared memory reference
			if 'existing_shm' in locals():
				existing_shm.close()

		return i_f, j_t, distance, stats, init_stats  # Return for progress tracking


	@staticmethod
	def pairwise_ged_with_gids_parallel(
			graph_id_f: int, graph_id_t: int, is_same_set: bool = True, **kwargs
	):
		global g_ged_env  # <- Use the global GEDEnv created in the worker initializer
		if is_same_set:
			global g_graphs
			graphs1, graphs2 = g_graphs, None
		else:
			global g_graphs_f, g_graphs_t
			graphs1, graphs2 = g_graphs_f, g_graphs_t

		dis, stats = GEDModel.pairwise_ged_with_gids(
			graph_id_f, graph_id_t, g_ged_env, graphs1,
			is_same_set=is_same_set, graphs2=graphs2, **kwargs
		)

		global g_stats_reported
		# print(g_stats_reported)
		if not g_stats_reported:
			# Report the stats only once per worker
			g_stats_reported = True
			global g_env_stats
			return dis, stats, g_env_stats
		else:
			return dis, stats, None  # Return None for env_stats if already reported


	# %% GEDEnv related methods:

	@staticmethod
	def get_env_type(graph: nx.Graph | None = None):
		"""
		Check the environment type of the graph.
		If `env_type` is set on initialization, return it.
		Otherwise, check the given graph's node and edge labels to determine the type.

		Only one node and one edge are checked to determine the type.
		This function expects that all nodes have the same type of labels, so as all
		edges.
		"""
		if graph is None:
			raise ValueError(
				'Graph is not provided while `env_type` not set on initialization. '
				'Cannot determine environment type.'
			)
		# Use 'gxl' env type only if all nodes and edge labes are strings, and at least one
		# node or edge label is present:
		one_n_labels = graph.nodes[list(graph.nodes)[0]]
		for k, v in one_n_labels.items():
			if not isinstance(v, str):
				return 'attr'
		if nx.number_of_edges(graph) != 0:
			one_e_labels = graph.edges[list(graph.edges)[0]]
			for k, v in one_e_labels.items():
				if not isinstance(v, str):
					return 'attr'
		if len(one_n_labels) > 0 or (
				nx.number_of_edges(graph) != 0 and len(one_e_labels) > 0
		):
			return 'gxl'
		return 'attr'


	@staticmethod
	def create_and_setup_ged_env(env_type: str | None = None, graph: nx.Graph = None, **kwargs):
		"""
		Create and set up the GED environment.

		Notes
		-----
		`GEDENV.init()` and `GEDENV.init_method()` must be called after all graphs are added
		to the GEDEnv. They are not called here.
		"""
		env_setting_time = time.time()

		from gklearn.gedlib import gedlibpy

		if env_type is None:
			env_type = GEDModel.get_env_type(graph=graph)
		ged_options = {
			'env_type': env_type,
			'edit_cost': kwargs['edit_cost_fun'],
			'method': kwargs['ed_method'],
			'edit_cost_constants': kwargs['edit_cost_constants'],
			'edit_cost_config': kwargs['edit_cost_config'],
		}

		ged_env = gedlibpy.GEDEnv(env_type=ged_options.get('env_type', 'attr'), verbose=False)
		ged_env.set_edit_cost(
			ged_options['edit_cost'],
			edit_cost_constant=ged_options['edit_cost_constants'],
			**ged_options.get('edit_cost_config') and {
				'edit_cost_config': ged_options['edit_cost_config']
			} or {}
		)

		ged_env.set_method(ged_options['method'], ged_options_to_string(ged_options))

		env_setting_time = time.time() - env_setting_time

		return ged_env, env_setting_time


	@staticmethod
	def add_graphs_to_ged_env(graphs: list[nx.Graph], ged_env, verbose: int = 1, **kwargs):
		# `init()` and `init_method()` must be called after all graphs are added to the GEDEnv.

		iterator = enumerate(graphs)
		if verbose >= 2:
			iterator = tqdm(
				iterator, desc='Adding graphs to the GED environment',
				file=sys.stdout, total=len(graphs)
			)
		graphs_adding_time = []
		for i, g in iterator:
			graph_adding_start_time = time.time()
			GEDModel.add_graph_to_ged_env(g.copy() if kwargs['copy_graphs'] else g, ged_env=ged_env)
			graphs_adding_time.append(time.time() - graph_adding_start_time)

		return graphs_adding_time


	@staticmethod
	def add_graph_to_ged_env(graph: nx.Graph, ged_env):
		ged_env.add_nx_graph(graph, '', ignore_duplicates=True)


	@staticmethod
	def init_ged_env_and_method(ged_env, **kwargs):
		# `init()` must be called after all graphs are added to the GEDEnv:
		# todo: determine which is faster: lazy or eager. Maybe do this automatically.
		# (eager can not show progress bar):
		init_options = 'LAZY_WITHOUT_SHUFFLED_COPIES' if kwargs['ged_init_options'] is None else \
			kwargs['ged_init_options']
		if init_options.startswith('EAGER_'):
			eager_label_cost_computing_time = time.time()
			print(f'{INFO_TAG}Starting eager label cost computing. This may take a while...')
			ged_env.init(init_options)
			print(f'{INFO_TAG}Eager label cost computing finished.')
			eager_label_cost_computing_time = time.time() - eager_label_cost_computing_time
		else:
			ged_env.init(init_options)
			eager_label_cost_computing_time = None
		# `init_method()` must be called after `init()`:
		ged_env.init_method()
		return eager_label_cost_computing_time


	@staticmethod
	def pairwise_ged_with_gids(
			graph_id1: int, graph_id2: int, ged_env, graphs: list[nx.Graph],
			is_same_set: bool = True, graphs2: list[nx.Graph] | None = None, **kwargs
	):
		"""
		Compute pairwise GED between two graphs using their IDs in the GEDEnv.

		This method uses the GEDEnv member globally available in the class.

		Parameters
		----------
		graph_id1 : int
			ID of the first graph in the GEDEnv. If `is_same_set` is False, it refers to the fitted
			(reference) graph.

		graph_id2 : int
			ID of the second graph in the GEDEnv. If `is_same_set` is False, it refers to the target
			graph.

		Notes
		-----
		- Be careful with the order between `graph_id1` and `graph_id2`. When `is_same_set` = False,
		`graph_id1` is the fitted (reference) graph and `graph_id2` is the target graph.

		Todo
		----
		- Since GED is not normally symmetric, maybe add an option to compute the average of the two
		- distances (forward and backward) or the minimum of the two distances.
		"""
		ged_computing_time = time.time()

		repeats = kwargs.get('repeats', 1)

		dis_min = np.inf

		if is_same_set:
			graph_id2_env = graph_id2
		else:
			graph_id2_env = len(graphs) + graph_id2  # Both graph lists were added to the GEDEnv.

		for i in range(0, repeats):
			ged_env.run_method(graph_id1, graph_id2_env)
			upper = ged_env.get_upper_bound(graph_id1, graph_id2_env)
			dis = upper
			# 		print(dis)
			if dis < dis_min:
				dis_min = dis
				pi_forward = ged_env.get_forward_map(graph_id1, graph_id2_env)
				pi_backward = ged_env.get_backward_map(graph_id1, graph_id2_env)
		# 			lower = ged_env.get_lower_bound(g, h)

		ged_computing_time = time.time() - ged_computing_time

		# make the map label correct (label remove mappings as np.inf):
		if is_same_set:
			g1, g2 = graphs[graph_id1], graphs[graph_id2]
		else:
			g1, g2 = graphs[graph_id1], graphs2[graph_id2]
		nodes1 = [n for n in g1.nodes()]
		nodes2 = [n for n in g2.nodes()]
		nb1 = nx.number_of_nodes(g1)
		nb2 = nx.number_of_nodes(g2)
		pi_forward = [nodes2[pi] if pi < nb2 else np.inf for pi in pi_forward]
		pi_backward = [nodes1[pi] if pi < nb1 else np.inf for pi in pi_backward]
		#		print(pi_forward)

		stats = {
			'ged_computing_time': ged_computing_time
		}

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
		return dis, stats


	# %%

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
				f"verbose={self.verbose}, " +
				(f"normalize={self.normalize}, " if hasattr(self, 'normalize') else "") +
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


# Context manager for shared memory with automatic cleanup
@contextmanager
def numpy_shared_memory(shape, dtype=np.float64):
	"""Create a numpy array in shared memory that automatically cleans up."""
	size = int(np.prod(shape)) * np.dtype(dtype).itemsize
	shm = shared_memory.SharedMemory(create=True, size=size)
	try:
		array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
		array.fill(0)  # Initialize with zeros
		yield array, shm.name
	finally:
		shm.close()
		shm.unlink()
