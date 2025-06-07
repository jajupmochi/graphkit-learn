"""
@File: compare_gedlib_with_coords_in_string_and_attr_format.py

@Author: jajupmochi
@Date: May 22 2025
"""
from typing import List

import networkx as nx
import numpy as np

ISSUE_TAG = "\033[91m[issue]\033[0m "  # Red
INFO_TAG = "\033[94m[info]\033[0m "  # Blue
SUCCESS_TAG = "\033[92m[success]\033[0m "  # Green


def fit_model_ged(
		graphs_X: List[nx.Graph],
		graphs_Y: List[nx.Graph] = None,
		ged_options: dict = None,
		parallel: bool = None,
		n_jobs: int = None,
		chunksize: int = None,
		copy_graphs: bool = True,
		read_resu_from_file: int = 1,
		output_dir: str = None,
		params_idx: str = None,
		reorder_graphs: bool = False,
		verbose: int = 2,
		**kwargs
):
	# if read_resu_from_file >= 1:
	#     fn_model = os.path.join(
	#         output_dir, 'metric_model.params_{}.pkl'.format(
	#             params_idx
	#         )
	#     )
	#     # Load model from file if it exists:
	#     if os.path.exists(fn_model) and os.path.getsize(fn_model) > 0:
	#         print('\nLoading model from file...')
	#         resu = pickle.load(open(fn_model, 'rb'))
	#         return resu['model'], resu['history'], resu['model'].dis_matrix

	# Reorder graphs if specified:
	if reorder_graphs:
		graphs_X = reorder_graphs_by_index(graphs_X, idx_key='id')
		if graphs_Y is not None:
			graphs_Y = reorder_graphs_by_index(graphs_Y, idx_key='id')

	# Compute metric matrix otherwise:
	print(f'{INFO_TAG}Computing metric matrix...')
	all_graphs = graphs_X + graphs_Y if graphs_Y else graphs_X
	nl_names = list(
		all_graphs[0].nodes[list(all_graphs[0].nodes)[0]].keys()
	) if graphs_X else []
	if not all_graphs:
		el_names = []
	else:
		idx_edge = (
			np.where(np.array([nx.number_of_edges(g) for g in all_graphs]) > 0)[0]
		)
		if len(idx_edge) == 0:
			el_names = []
		else:
			el_names = list(
				all_graphs[idx_edge[0]].edges[
					list(all_graphs[idx_edge[0]].edges)[0]].keys()
			)

	from gklearn.experiments.ged.ged_model.ged_model_parallel import GEDModel

	if parallel is False:
		parallel = None
	elif parallel is True:
		parallel = 'imap_unordered'

	model = GEDModel(
		# env_type=ged_options['env_type'],
		ed_method=ged_options['method'],
		edit_cost_fun=ged_options['edit_cost_fun'],
		init_edit_cost_constants=ged_options['edit_costs'],
		edit_cost_config=ged_options.get('edit_cost_config', {}),
		optim_method=ged_options['optim_method'],
		node_labels=nl_names, edge_labels=el_names,
		parallel=parallel,
		n_jobs=n_jobs,
		chunksize=chunksize,
		copy_graphs=copy_graphs,
		# make sure it is a full deep copy. and faster!
		verbose=verbose
	)

	# Train model.
	try:
		if graphs_Y is None:
			# Compute the distance matrix for the same set of graphs:
			matrix = model.fit_transform(
				graphs_X, y=graphs_Y,
				save_dm_train=True, repeats=ged_options['repeats'],
			)
		else:
			model.fit(graphs_X, repeats=ged_options['repeats'])
			matrix = model.transform(
				graphs_Y,
				save_dm_test=True, repeats=ged_options['repeats'],
			)

	except OSError as exception:
		if 'GLIBC_2.23' in exception.args[0]:
			msg = \
				'This error is very likely due to the low version of GLIBC ' \
				'on your system. ' \
				'The required version of GLIBC is 2.23. This may happen on the ' \
				'CentOS 7 system, where the highest version of GLIBC is 2.17. ' \
				'You may check your CLIBC version by bash command `rpm -q glibc`. ' \
				'The `graphkit-learn` library comes with GLIBC_2.23, which you can ' \
				'install by enable the `--build-gedlib` option: ' \
				'`python3 setup.py install --build-gedlib`. This will compile the C++ ' \
				'module `gedlib`, which requires a C++ compiler and CMake.'
			raise AssertionError(msg) from exception
		else:
			assert False, exception
	except Exception as exception:
		assert False, exception

	# Save history:
	# For graph kernels it is n * (n - 1) / 2:
	if graphs_Y is None:
		n_pairs = len(graphs_X) * (len(graphs_X) - 1) / 2
	else:
		n_pairs = len(graphs_X) * len(graphs_Y)
	# history = {'run_time': AverageMeter()}
	# history['run_time'].update(model.run_time / n_pairs, n_pairs)

	# # Save model and history to file:
	# if read_resu_from_file >= 1:
	#     os.makedirs(os.path.dirname(fn_model), exist_ok=True)
	#     pickle.dump({'model': model, 'history': history}, open(fn_model, 'wb'))

	# Print out the information:
	params_msg = f' for parameters {params_idx}' if params_idx else ''
	print(
		f'{SUCCESS_TAG}Computed metric matrix of size {matrix.shape} in {model.run_time:.3f} '
		f'seconds ({(model.run_time / n_pairs):.9f} s per pair){params_msg}.'
	)

	stats = {
		'n_pairs': n_pairs,
		'matrix_shape': matrix.shape,
		'run_time': model.run_time,
		'run_time_per_pair': model.run_time / n_pairs,
	}

	return model, matrix, stats


def show_some_graphs(graphs):
	"""
	Show some graphs from the list of graphs.
	"""
	print(f'{INFO_TAG}Showing some graphs:')
	for i, g in enumerate(graphs[:3]):
		print(f'Graph {i}:')
		print('Number of nodes:', g.number_of_nodes())
		print('Number of edges:', g.number_of_edges())
		print('Nodes:', g.nodes(data=True))
		print('Edges:', g.edges(data=True))
		print()


def convert_graphs_coords_from_attr_to_string(graphs: List[nx.Graph]):
	"""
	Convert the coordinates of nodes in graphs from the attribute format `AttrLabel` to the string format `GXLLabel`.
	"""
	for g in graphs:
		for node in g.nodes(data=True):
			if 'coords' in node[1]:
				# Convert the coordinates to string format and store them in "x" and "y" keys:
				coords = node[1]['coords']
				node[1]['x'] = str(coords[0])
				node[1]['y'] = str(coords[1])
				for idx in range(2, len(coords)):
					# If there are more than 2 coordinates, store them with extra keys:
					node[1][f'coord_{idx}'] = str(coords[idx])
				del node[1]['coords']
	print(f'{INFO_TAG}Converted coordinates from attribute format to string format.')


def fit_model_attr_version(
		seed: int = 42, n_graphs: int = 100, n_emb_dim: int = 2, parallel: bool = False
) -> (np.array, float):
	"""
	Fit the GED model with graphs that have coordinates on nodes in attribute format `AttrLabel`.
	"""
	print(
		f'\n{INFO_TAG}Fitting model with graphs with coordinates in attribute format...'
	)

	from gklearn.experiments.ged.ged_model.graph_generator import GraphGenerator
	generator = GraphGenerator(
		num_graphs=n_graphs,
		max_num_nodes=20,
		min_num_nodes=10,
		max_num_edges=50,
		min_num_edges=20,
		node_feat_type='float',
		edge_feat_type=None,
		with_discrete_n_features=False,
		with_discrete_e_features=False,
		with_continuous_n_features=True,
		with_continuous_e_features=False,
		continuous_n_feature_key='coords',
		continuous_n_feature_dim=n_emb_dim,
		continuous_e_feature_dim=0,
		seed=seed
	)
	graphs = generator.generate_graphs()
	# Check graph node label format:
	one_n_labels = graphs[0].nodes[list(graphs[0].nodes)[0]]
	assert 'coords' in one_n_labels and isinstance(one_n_labels['coords'], np.ndarray) and (
		len(one_n_labels['coords']) > 0 and one_n_labels['coords'].dtype in [
			np.float64, np.float32]
	), (
		'The node labels should contain "coords" key with a numpy array as value.'
	)
	print(
		f'{INFO_TAG}Generated {len(graphs)} graphs with coordinates in string format.'
	)
	show_some_graphs(graphs)

	# Set GED options:
	ged_options = {
		'env_type': 'attr',  # Use the attribute-based environment
		'method': 'BIPARTITE',
		'edit_cost_fun': 'GEOMETRIC',
		'edit_costs': [3, 3, 1, 3, 3, 1],
		'edit_cost_config': {
			'node_coord_metric': 'euclidean',
			'node_embed_metric': 'cosine_distance',
			'edge_weight_metric': 'euclidean',
			'edge_embed_metric': 'cosine_distance',
		},
		'optim_method': 'init',
		'repeats': 1,
	}

	fit_settings = {
		'parallel': parallel,  # Use parallel processing if specified
		'n_jobs': 10,  # min(12, max(os.cpu_count() - 2, 0)),
		'chunksize': None,  # None == automatic determination
		'copy_graphs': True,
		'reorder_graphs': False,
	}

	# Fit model and compute GED matrix:
	model, matrix, stats = fit_model_ged(
		graphs,
		graphs_Y=None,
		ged_options=ged_options,
		read_resu_from_file=0,
		output_dir=None,
		params_idx=None,
		verbose=2,
		**fit_settings
	)
	print("Model:", model)
	print("Matrix shape:", matrix.shape)
	print("Run time:", stats['run_time'])

	return matrix, stats['run_time']


def fit_model_string_version(
		seed: int = 42, n_graphs: int = 100, n_emb_dim: int = 2, parallel: bool = False
) -> (np.array, float):
	"""
	Fit the GED model with graphs that have coordinates on nodes in string format `GXLLabel`.
	"""
	print(f'\n{INFO_TAG}Fitting model with graphs with coordinates in string format...')

	from gklearn.experiments.ged.ged_model.graph_generator import GraphGenerator
	generator = GraphGenerator(
		num_graphs=n_graphs,
		max_num_nodes=20,
		min_num_nodes=10,
		max_num_edges=50,
		min_num_edges=20,
		node_feat_type='float',
		edge_feat_type=None,
		with_discrete_n_features=False,
		with_discrete_e_features=False,
		with_continuous_n_features=True,
		with_continuous_e_features=False,
		continuous_n_feature_key='coords',
		continuous_n_feature_dim=n_emb_dim,
		continuous_e_feature_dim=0,
		seed=seed
	)
	graphs = generator.generate_graphs()
	convert_graphs_coords_from_attr_to_string(graphs)
	# Check graph node label format:
	one_n_labels = graphs[0].nodes[list(graphs[0].nodes)[0]]
	assert 'x' in one_n_labels and 'y' in one_n_labels and isinstance(
		one_n_labels['x'], str) and isinstance(one_n_labels['y'], str), (
		'The node labels should contain "x" and "y" keys with string values.'
	)
	print(
		f'{INFO_TAG}Generated {len(graphs)} graphs with coordinates in string format.'
	)
	show_some_graphs(graphs)

	# Set GED options:
	ged_options = {
		'env_type': 'gxl',  # Use the GXLLabel environment
		'method': 'BIPARTITE',
		'edit_cost_fun': 'NON_SYMBOLIC',
		'edit_costs': [3, 3, 1, 3, 3, 1],
		'optim_method': 'init',
		'repeats': 1
	}

	fit_settings = {
		'parallel': parallel,  # Use parallel processing if specified
		'n_jobs': 10,  # min(12, max(os.cpu_count() - 2, 0)),
		'chunksize': None,  # None == automatic determination
		'copy_graphs': True,
		'reorder_graphs': False,
	}

	# Fit model and compute GED matrix:
	model, matrix, stats = fit_model_ged(
		graphs,
		graphs_Y=None,
		ged_options=ged_options,
		read_resu_from_file=0,
		output_dir=None,
		params_idx=None,
		verbose=2,
		**fit_settings
	)
	print("Model:", model)
	print("Matrix shape:", matrix.shape)
	print("Run time:", stats['run_time'])

	return matrix, stats['run_time']


def compare_gedlib_with_coords_in_string_and_attr_format(
		seed: int = 42, n_graphs: int = 100, n_emb_dim: int = 2, parallel: bool = False
) -> (np.array, np.array):
	"""
	Compare the output and the performance of GEDLIB with the same graphs with coordinates on nodes,
	but one is in string format `GXLLabel` and the other is in the complex attribute format `AttrLabel`.
	"""
	cost_matrix_a, run_time_a = fit_model_attr_version(
		seed=seed, n_graphs=n_graphs, n_emb_dim=n_emb_dim, parallel=parallel
	)
	cost_matrix_s, run_time_s = fit_model_string_version(
		seed=seed, n_graphs=n_graphs, n_emb_dim=n_emb_dim, parallel=parallel
	)
	if not np.allclose(cost_matrix_s, cost_matrix_a, rtol=1e-9):
		print(
			f'{ISSUE_TAG}The cost matrices are not equal! '
			f'String version: {cost_matrix_s.shape}, '
			f'Attribute version: {cost_matrix_a.shape}, '
			f'Relevant tolerance: 1e-9.'
		)
	else:
		print(
			f'{SUCCESS_TAG}The cost matrices are equal! '
			f'String version: {cost_matrix_s.shape}, '
			f'Attribute version: {cost_matrix_a.shape}, '
			f'Relevant tolerance: 1e-9.'
		)

	# Print the first 5 rows and columns of the matrices:
	print('First 5 rows and columns of the string version cost matrix:')
	print(cost_matrix_s[:5, :5])
	print('First 5 rows and columns of the attribute version cost matrix:')
	print(cost_matrix_a[:5, :5])

	# Print the run times:
	print(f'String version run time: {run_time_s:.3f} seconds.')
	print(f'Attribute version run time: {run_time_a:.3f} seconds.')

	# Print the run time per pair:
	n_pairs = cost_matrix_s.shape[0] * (cost_matrix_s.shape[0] - 1) / 2
	print(
		f'String version run time per pair: {run_time_s / n_pairs:.9f} seconds.'
	)
	print(
		f'Attribute version run time per pair: {run_time_a / n_pairs:.9f} seconds.'
	)

	return cost_matrix_s, cost_matrix_a


if __name__ == '__main__':
	# Test the class
	# feat_type = 'str'
	seed = 42
	n_graphs = 500
	n_emb_dim = 100
	parallel = True
	compare_gedlib_with_coords_in_string_and_attr_format(
		seed=seed, n_graphs=n_graphs, n_emb_dim=n_emb_dim, parallel=parallel
	)

	# # Comparison of the two versions:
	#
	# General Settings:
	# - n_graphs: 500
	# - node numbers: 10-20
	# - edge numbers: 20-50
	# - Regenerate GEDEnv for each pair of computation (not optimized).
	# - Coordinates as labels of strings in GXLLabel or one label of np.array in AttrLabel,
	#   where the latter is optimized by the Eigen C++ library for vectorized operations.
	#
	# ## Without parallelization:
	#
	# ### n_emb_dim = 2:
	# - String version run time: 7.4e-4 s per pair (92.3 s total).
	# - Attribute version run time: 5.0e-4 s per pair (62.4 s total).
	# The Attr version is ~ 1.5x faster than the String version.
	#
	# ### n_emb_dim = 20:
	# - String version run time: 5.4e-3 s per pair (675.1 s total).
	# - Attribute version run time: 5.5e-4 s per pair (69.0 s total).
	# The Attr version is ~ 10x faster than the String version.
	#
	# ### n_emb_dim = 100:
	# - String version run time: too long to compute (over 1 h ~ 3698.5 s).
	# - Attribute version run time: 8.0e-4 s per pair (99.9 s total).
	# The Attr version is ~ 37x faster than the String version.
	#
	# ### Conclusion:
	# - The Attribute version is faster than the String version.
	# - With the increase of the dimensionality of the coordinates (n_emb_dim):
	#   -- Attribute version takes almost the same level of time to compute pairwise
	#      distances (e.g., ~ 1.6x slower when n_emb_dim = 100 than 2).
	#   -- String version becomes unusable (~ 40x slower when n_emb_dim = 100 than 2),
	#      and ~ 37x slower than the Attribute version with n_emb_dim = 100.
	#
	# ## With parallelization (n_jobs=10):
	#
	# ### n_emb_dim = 2:
	# - String version run time: 3.6e-4 s per pair (45.3 s total).
	# - Attribute version run time: 3.6e-4 s per pair (45.3 s total).
	# The two versions are almost equal in terms of run time.
	#
	# ### n_emb_dim = 20:
	# - String version run time: 9.8e-4 s per pair (122.4 s total).
	# - Attribute version run time: 4.1e-4 s per pair (50.7 s total).
	# The Attribute version is ~ 2.4x faster than the String version.
	#
	# ### n_emb_dim = 100:
	# - String version run time: 5.3e-3 s per pair (664.3 s total).
	# - Attribute version run time: 4.4e-4 s per pair (54.3 s total).
	# The Attribute version is ~ 12.2x faster than the String version.
	#
	# ### Conclusion:
	# - The Attribute version is still faster than the String version.
	# - The parallelization helps to reduce the run time of both versions,
	#   but the improvement on the String version is much more significant,
	#   e.g., ~ x faster than the non-parallelized version with n_emb_dim = 100
	# - On the other hand, the improvement brought by parallelization is not so significant
	#   for the Attribute version, e.g., ~ 1.8x faster than the non-parallelized version
	#   with n_emb_dim = 100.
	#   -- I assume the reason is that the construction of the GEDEnvAttr and the
	#      Python-C++ interface conversion becomes the bottleneck of the process.
