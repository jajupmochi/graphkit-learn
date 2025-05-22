"""
basic

@Author: jajupmochi
@Date: May 22 2025
"""
import gc
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from itertools import combinations, product
from multiprocessing import shared_memory, Manager

import joblib
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from gklearn.ged.model.distances import euclid_d
from gklearn.ged.util import pairwise_ged
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

			if self.parallel in [
				'imap_unordered', 'joblib', 'concurrent', 'multiprocessing'
			]:
				dis_matrix = self._compute_cross_distance_matrix_parallel(
					Y_copy, graphs_copy, **kwargs
				)

			elif self.parallel is None:
				dis_matrix = self._compute_cross_distance_matrix_series(
					Y_copy, graphs_copy, **kwargs
				)
			else:
				raise Exception('Parallel mode is not set correctly.')

			self._run_time = time.time() - start_time

			if self.verbose:
				print(
					'Distance matrix of size (%d, %d) built in %s seconds.'
					% (len(Y), len(self._graphs), self._run_time)
				)

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


	def _compute_X_distance_matrix(self, **kwargs):
		graphs = (
			[
				g.copy() for g in self._graphs
			] if self.copy_graphs else self._graphs
		)

		start_time = time.time()

		# if self.parallel == 'imap_unordered':
		#     dis_matrix = self._compute_X_dm_imap_unordered(graphs, **kwargs)
		if self.parallel in [
			'imap_unordered', 'joblib', 'concurrent', 'multiprocessing'
		]:
			dis_matrix = self._compute_X_dm_parallel(graphs, **kwargs)
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


	# todo: this is not refactored yet.
	def _compute_X_dm_parallel(self, graphs, **kwargs):
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
		n = len(graphs)

		# Get all pairs of indices
		pairs = list(combinations(range(n), 2))
		len_itr = len(pairs)

		n_jobs = self.n_jobs
		chunksize = self.chunksize
		method = self.parallel
		memory_limit = kwargs.get('memory_limit', 'auto')

		if self.verbose:
			print('Graphs in total: %d.' % len(graphs))
			print('The total # of pairs is %d.' % len_itr)

		# Determine number of processes
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

		# For networkx graphs, we need to use a Manager to share them between processes
		with Manager() as manager:
			# Create a managed shared list for the graphs
			shared_graphs = manager.list(graphs)

			# Get a function reference to compute_ged that can be pickled
			# Using a Python trick to make the instance method picklable
			compute_ged_func = self.compute_ged

			# Create a shared memory array for results
			with numpy_shared_memory((n, n), dtype=np.float64) as (
					dis_matrix, shm_name
			):

				# # Define worker function that updates the shared matrix directly
				# def process_pair(pair):
				#     i, j = pair
				#     g1, g2 = graphs[i], graphs[j]
				#
				#     try:
				#         # Access the shared memory
				#         existing_shm = shared_memory.SharedMemory(name=shm_name)
				#         shared_matrix = np.ndarray(
				#             (n, n), dtype=np.float64, buffer=existing_shm.buf
				#         )
				#
				#         # Compute distance - use graph indices to avoid serializing graphs
				#         distance = self.compute_ged(g1, g2, **kwargs)
				#
				#         # Update the matrix with thread/process-safe approach
				#         # We're only writing to unique cells so no locking needed
				#         shared_matrix[i, j] = distance
				#         shared_matrix[j, i] = distance
				#
				#     finally:
				#         # Clean up local shared memory reference
				#         if 'existing_shm' in locals():
				#             existing_shm.close()
				#
				#     return i, j, distance  # Return for progress tracking

				# Create a partial function with fixed arguments - must use module-level function
				worker = partial(
					self._process_pair_worker,
					graphs_manager=shared_graphs,
					shm_name=shm_name,
					matrix_shape=(n, n),
					compute_ged_func=compute_ged_func,
					**kwargs
				)

				try:
					# Force garbage collection before starting parallel processing
					gc.collect()

					# Three different parallelization options for different scenarios
					if method == 'joblib':
						if memory_limit == 'auto':
							# Set max_nbytes according to the size of the shared memory:
							# Get the size of the shared memory in bytes:
							shm = shared_memory.SharedMemory(name=shm_name)
							memory_limit = shm.size
							shm.close()
							if self.verbose >= 2:
								print(
									f"Setting memory limit to {memory_limit} bytes per process."
								)

						# Option 1: joblib - great for large datasets, memory control, possible caching
						with joblib.parallel_backend(
								'loky', n_jobs=n_jobs, inner_max_num_threads=1,
								mmap_mode='r', temp_folder='/tmp'
						):
							results = joblib.Parallel(
								verbose=self.verbose >= 2,
								prefer="processes",
								batch_size=chunksize,
								pre_dispatch='2*n_jobs',
								max_nbytes=memory_limit
							)(
								joblib.delayed(worker)(pair) for pair in pairs
							)

					elif method == 'concurrent':
						# Option 2: ProcessPoolExecutor - cleaner API, slightly faster for CPU-bound tasks
						with ProcessPoolExecutor(
								max_workers=n_jobs
						) as executor:
							futures = [executor.submit(worker, pair) for pair
							           in pairs]

							# Track progress if verbose
							if self.verbose >= 2:
								results = []
								for f in tqdm(
										futures, total=len(futures),
										desc='Computing distance matrix'
								):
									results.append(f.result())
							else:
								results = [f.result() for f in futures]

					elif method in ['imap_unordered' or 'multiprocessing']:
						# Option 3: multiprocessing.Pool with imap_unordered - more control, classic approach
						with multiprocessing.Pool(processes=n_jobs) as pool:
							if self.verbose >= 2:
								results = list(
									tqdm(
										pool.imap_unordered(
											worker, pairs,
											chunksize=chunksize
										),
										total=len_itr,
										desc='Computing distance matrix',
										file=sys.stdout
									)
								)
							else:
								results = list(
									pool.imap_unordered(
										worker, pairs, chunksize=chunksize
									)
								)

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
			pair, graphs_manager, shm_name, matrix_shape,
			compute_ged_func, **kwargs
	):
		"""Worker function that processes a pair of graphs and updates the shared matrix.
		Must be defined at module level to be picklable."""
		i, j = pair

		# Access the shared graphs from the manager
		g1 = graphs_manager[i]
		g2 = graphs_manager[j]

		try:
			# Access the shared memory
			existing_shm = shared_memory.SharedMemory(name=shm_name)
			shared_matrix = np.ndarray(
				matrix_shape, dtype=np.float64, buffer=existing_shm.buf
			)

			# Compute distance using the function reference
			distance, _ = compute_ged_func(g1, g2, **kwargs)

			# Update the matrix
			shared_matrix[i, j] = distance
			shared_matrix[j, i] = distance

		finally:
			# Clean up local shared memory reference
			if 'existing_shm' in locals():
				existing_shm.close()

		return i, j, distance  # Return for progress tracking


	def _compute_cross_distance_matrix_series(self, graphs1, graphs2, **kwargs):
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
		n1 = len(graphs1)
		n2 = len(graphs2)

		# Initialize distance matrix with zeros
		dis_matrix = np.zeros((n1, n2))

		# Cross set case: compute all pairs between the two sets
		iterator = product(range(n1), range(n2))
		len_itr = n1 * n2

		if self.verbose:
			print(f'Computing distances between {n1} and {n2} graphs.')
			print(f'The total # of pairs is {len_itr}.')

		for i, j in get_iters(
				iterator, desc='Computing distance matrix',
				file=sys.stdout, verbose=(self.verbose >= 2), length=len_itr
		):
			g1, g2 = graphs1[i], graphs2[j]
			dis_matrix[i, j], _ = self.compute_ged(g1, g2, **kwargs)

		return dis_matrix


	def _compute_cross_distance_matrix_parallel(
			self, graphs1, graphs2, **kwargs
	):
		"""Compute the GED distance matrix between two sets of graphs (X and Y)
		with parallelization.

		Parameters
		----------
		X, Y : list of graphs
			The input graphs.

		Returns
		-------
		dis_matrix : numpy array, shape = [n_X, n_Y]
			The computed distance matrix.

		References
		----------
		This method is written with the help of the Claude 3.7 Sonnet AI,
		accessed on 2025.05.15.


		todo: this can be merged with the _compute_X_dm_parallel method.
		"""
		# Handle the case where graphs2 is not provided
		is_same_set = graphs2 is None
		if is_same_set:
			graphs2 = graphs1

		n1 = len(graphs1)
		n2 = len(graphs2)

		# Get all pairs of indices to compute
		if is_same_set:
			# Only compute upper triangular portion for efficiency when comparing within same set
			pairs = list(combinations(range(n1), 2))
		else:
			# Compute all pairs when comparing between different sets
			pairs = list(product(range(n1), range(n2)))

		len_itr = len(pairs)

		n_jobs = self.n_jobs
		chunksize = self.chunksize
		method = self.parallel
		memory_limit = kwargs.get('memory_limit', 'auto')

		if self.verbose:
			if is_same_set:
				print(f'Graphs in total: {n1}.')
			else:
				print(f'Computing distances between {n1} and {n2} graphs.')
			print(f'The total # of pairs is {len_itr}.')

		# Determine number of processes
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

		# For networkx graphs, we need to use a Manager to share them between processes
		with Manager() as manager:
			# Create managed shared lists for both graph sets
			shared_graphs1 = manager.list(graphs1)
			shared_graphs2 = manager.list(graphs2)

			# Get a function reference to compute_ged that can be pickled
			# Using a Python trick to make the instance method picklable
			compute_ged_func = self.compute_ged

			# Create a shared memory array for results
			with numpy_shared_memory((n1, n2), dtype=np.float64) as (
					dis_matrix, shm_name
			):
				# Create a partial function with fixed arguments - MUST NOT use
				# inline function here, as it won't be picklable:
				worker = partial(
					self._process_pair_worker_cross,
					graphs1_manager=shared_graphs1,
					graphs2_manager=shared_graphs2,
					shm_name=shm_name,
					matrix_shape=(n1, n2),
					compute_ged_func=compute_ged_func,
					is_same_set=is_same_set,
					**kwargs
				)

				try:
					# Force garbage collection before starting parallel processing
					gc.collect()

					# Three different parallelization options for different scenarios
					if method == 'joblib':
						if memory_limit == 'auto':
							# Set max_nbytes according to the size of the shared memory:
							# Get the size of the shared memory in bytes:
							shm = shared_memory.SharedMemory(name=shm_name)
							memory_limit = shm.size
							shm.close()
							if self.verbose >= 2:
								print(
									f"Setting memory limit to {memory_limit} bytes per process."
								)

						# Option 1: joblib - great for large datasets, memory control, possible caching
						with joblib.parallel_backend(
								'loky', n_jobs=n_jobs, inner_max_num_threads=1,
								mmap_mode='r', temp_folder='/tmp'
						):
							results = joblib.Parallel(
								verbose=self.verbose >= 2,
								prefer="processes",
								batch_size=chunksize,
								pre_dispatch='2*n_jobs',
								max_nbytes=memory_limit
							)(
								joblib.delayed(worker)(pair) for pair in pairs
							)

					elif method == 'concurrent':
						# Option 2: ProcessPoolExecutor - cleaner API, slightly faster for CPU-bound tasks
						with ProcessPoolExecutor(
								max_workers=n_jobs
						) as executor:
							futures = [executor.submit(worker, pair) for pair
							           in pairs]

							# Track progress if verbose
							if self.verbose >= 2:
								results = []
								for f in tqdm(
										futures, total=len(futures),
										desc='Computing distance matrix'
								):
									results.append(f.result())
							else:
								results = [f.result() for f in futures]

					elif method in ['imap_unordered' or 'multiprocessing']:
						# Option 3: multiprocessing.Pool with imap_unordered - more control, classic approach
						with multiprocessing.Pool(processes=n_jobs) as pool:
							if self.verbose >= 2:
								results = list(
									tqdm(
										pool.imap_unordered(
											worker, pairs,
											chunksize=chunksize
										),
										total=len_itr,
										desc='Computing distance matrix',
										file=sys.stdout
									)
								)
							else:
								results = list(
									pool.imap_unordered(
										worker, pairs, chunksize=chunksize
									)
								)

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
	def _process_pair_worker_cross(
			pair, graphs1_manager, graphs2_manager, shm_name, matrix_shape,
			compute_ged_func, is_same_set=False, **kwargs
	):
		"""Worker function that processes a pair of graphs and updates the shared matrix.
		Must be defined at module level to be picklable."""
		i, j = pair

		# Access the shared graphs from the manager
		g1 = graphs1_manager[i]
		g2 = graphs2_manager[j]

		try:
			# Access the shared memory
			existing_shm = shared_memory.SharedMemory(name=shm_name)
			shared_matrix = np.ndarray(
				matrix_shape, dtype=np.float64, buffer=existing_shm.buf
			)

			# Compute distance using the function reference
			distance, _ = compute_ged_func(g1, g2, **kwargs)

			# Update the matrix
			shared_matrix[i, j] = distance

			# If computing within the same set, update symmetric position:
			if is_same_set and i != j:
				shared_matrix[j, i] = distance

		finally:
			# Clean up local shared memory reference
			if 'existing_shm' in locals():
				existing_shm.close()

		return i, j, distance  # Return for progress tracking


	def _wrapper_compute_ged(self, itr):
		i = itr[0]
		j = itr[1]
		# @TODO: repeats are not considered here.
		dis, _ = self.compute_ged(G_gn[i], G_gn[j])
		return i, j, dis


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
			f"normalize={self.normalize}, " if hasattr(
				self, 'normalize'
			) else ""
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


