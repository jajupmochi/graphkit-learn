"""
@File: profile_ged_model_cross_matrix.py

@Author: jajupmochi
@Date: June 3 2025
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
		use_global_env: bool = True,
		**kwargs
):
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

	if use_global_env:
		from gklearn.experiments.ged.ged_model.ged_model_global_env import GEDModel
		print(f'>>> {INFO_TAG}Using global GEDEnv for all pairs of graphs.')
	else:
		from gklearn.experiments.ged.ged_model.ged_model_parallel import GEDModel
		print(f'>>> {INFO_TAG}Using local GEDEnv for each pair of graphs.')

	if parallel is False:
		parallel = None
	elif parallel is True:
		# Set automatically: the global version uses 'concurrent', and local version 'multiprocessing':
		parallel = True

	model = GEDModel(
		# env_type=ged_options['env_type'],
		ed_method=ged_options['method'],
		edit_cost_fun=ged_options['edit_cost_fun'],
		init_edit_cost_constants=ged_options['edit_costs'],
		edit_cost_config=ged_options.get('edit_cost_config', {}),
		optim_method=ged_options['optim_method'],
		ged_init_options=ged_options['init_options'],
		node_labels=nl_names, edge_labels=el_names,
		parallel=parallel,
		n_jobs=n_jobs,  # fixme: None
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
		'env_stats': model.env_stats if use_global_env else model.pairwise_stats,
	}

	return model, matrix, stats


def show_some_graphs(graphs):
	"""
	Show some graphs from the list of graphs.
	"""
	print(f'{INFO_TAG}Showing some graphs:')
	for i, g in enumerate(graphs[:2]):
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


def fit_model_global_version(
		seed: int = 42, n_graphs: int = 100, n_emb_dim: int = 2, parallel: bool = False,
		ged_init_mode: str = 'eager'
) -> (np.array, float):
	"""
	Fit the GED model with GEDEnv as a member of the model globally available in the class.
	"""
	print(
		f'\n{INFO_TAG}Fitting GEDModel with GEDEnv as a member of the model...'
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
		f'{INFO_TAG}Generated {len(graphs)} graphs with coordinates in np.array format.'
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
		'init_options': 'EAGER_WITHOUT_SHUFFLED_COPIES' if ged_init_mode == 'eager' else 'LAZY_WITHOUT_SHUFFLED_COPIES',
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
		use_global_env=True,  # Use local GEDEnv for each pair
		verbose=2,
		**fit_settings
	)
	print("Model:", model)
	print("Matrix shape:", matrix.shape)
	print("Run time:", stats['run_time'])

	return matrix, stats


def fit_model_local_version(
		seed: int = 42, n_graphs: int = 100, n_emb_dim: int = 2, parallel: bool = False,
		ged_init_mode: str = 'eager'
) -> (np.array, float):
	"""
	Fit the GED model with GEDEnv locally created for each pair of graphs.
	"""
	print(
		f'\n{INFO_TAG}Fitting GEDModel with GEDEnv created locally for each pair of graphs...'
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
		f'{INFO_TAG}Generated {len(graphs)} graphs with coordinates in np.array format.'
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
		'init_options': 'EAGER_WITHOUT_SHUFFLED_COPIES' if ged_init_mode == 'eager' else 'LAZY_WITHOUT_SHUFFLED_COPIES',
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
		use_global_env=False,  # Use local GEDEnv for each pair
		verbose=2,
		**fit_settings
	)
	print("Model:", model)
	print("Matrix shape:", matrix.shape)
	print("Run time:", stats['run_time'])

	return matrix, stats


def print_stats_local_version(stats: dict):
	# Print the run times:
	print(
		f'{INFO_TAG}The total run time for the GEDModel: '
		f'{stats["run_time"]:.3f} s / {stats["run_time_per_pair"]:.9f} s per pair.'
	)

	pairwise_stats = stats['env_stats']
	keys = ['pairwise_total_time', 'env_setting_time', 'graphs_adding_time', 'ged_computing_time']
	n_pairs = len(pairwise_stats)

	time_stats = {}
	for key in keys:
		time_stats[key] = sum([pair[key] for pair in pairwise_stats])
		print(
			f'{INFO_TAG}{key.replace("_", " ")}: '
			f'{time_stats[key]:.3f} s / {time_stats[key] / n_pairs:.9f} s per pair. '
			f'({time_stats[key] / stats["run_time"] * 100:.2f}% of total run time).'
		)


def print_stats_global_version(stats: dict):
	# Print the run times:
	print(
		f'{INFO_TAG}The total run time for the GEDModel: '
		f'{stats["run_time"]:.3f} s / {stats["run_time_per_pair"]:.9f} s per pair.'
	)

	env_stats = stats['env_stats']
	time_stats = {}
	keys = ['env_setting_time', 'eager_label_cost_computing_time']
	for key in keys:
		if key not in env_stats:
			continue
		time_stats[key] = env_stats[key]
		print(
			f'{INFO_TAG}{key.replace("_", " ")}: '
			f'{time_stats[key]:.3f} s. '
			f'({time_stats[key] / stats["run_time"] * 100:.2f}% of total run time).'
		)

	keys = ['env_setting_time_parallel']#  'eager_label_cost_computing_time']
	for key in keys:
		if key not in env_stats:
			continue
		time_stats[key] = sum(env_stats[key])
		n_ele = len(env_stats[key])
		print(
			f'{INFO_TAG}{key.replace("_", " ")}: '
			f'{time_stats[key]:.3f} s / {time_stats[key] / n_ele:.9f} s per worker. '
			f'({time_stats[key] / stats["run_time"] * 100:.2f}% of total run time).'
		)

	keys = ['graphs_adding_time_parallel']
	for key in keys:
		if key not in env_stats:
			continue
		np_time = np.array(env_stats[key])
		time_per_worker = np_time.sum(axis=1)
		n_workers = np_time.shape[0]
		time_stats[key] = np.sum(time_per_worker)
		n_ele = n_workers * np_time.shape[1]
		print(
			f'{INFO_TAG}{key.replace("_", " ")}: '
			f'{time_stats[key]:.3f} s / {time_stats[key] / n_ele:.9f} s per worker per graph. '
			f'({time_stats[key] / stats["run_time"] * 100:.2f}% of total run time). '
			f'Time per worker: {time_per_worker}.'
		)

	keys = ['graphs_adding_time', 'ged_computing_time', 'ged_computing_time_parallel']
	elements = ['graph', 'pair', 'pair']
	for key, ele in zip(keys, elements):
		if key not in env_stats:
			continue
		time_stats[key] = sum(env_stats[key])
		n_ele = len(env_stats[key])
		print(
			f'{INFO_TAG}{key.replace("_", " ")}: '
			f'{time_stats[key]:.3f} s / {time_stats[key] / n_ele:.9f} s per {ele}. '
			f'({time_stats[key] / stats["run_time"] * 100:.2f}% of total run time).'
		)


def compare_ged_model_with_global_and_local_env(
		seed: int = 42, n_graphs: int = 100, n_emb_dim: int = 2, parallel: bool = False,
		ged_init_mode: str = 'eager'
) -> (np.array, np.array):
	"""
	Compare the output and the performance of the following two GEDModel versions:
	- `GEDModel` with a GEDEnv as its global variable, which will be created along with the model.
	- `GEDModel` without a GEDEnv. GEDEnv will be created for each pair of graphs inside the pairwise
	computation.
	Both versions use `AttrLabel` as the node and edge labels format.
	"""
	cost_matrix_g, stats_g = fit_model_global_version(
		seed=seed, n_graphs=n_graphs, n_emb_dim=n_emb_dim, parallel=parallel,
		ged_init_mode=ged_init_mode
	)
	cost_matrix_l, stats_l = fit_model_local_version(
		seed=seed, n_graphs=n_graphs, n_emb_dim=n_emb_dim, parallel=parallel,
		ged_init_mode=ged_init_mode
	)

	if not np.allclose(cost_matrix_g, cost_matrix_l, rtol=1e-9):
		print(
			f'{ISSUE_TAG}The cost matrices are not equal! '
			f'String version: {cost_matrix_g.shape}, '
			f'Attribute version: {cost_matrix_l.shape}, '
			f'Relevant tolerance: 1e-9.'
		)
	else:
		print(
			f'{SUCCESS_TAG}The cost matrices are equal! '
			f'String version: {cost_matrix_g.shape}, '
			f'Attribute version: {cost_matrix_l.shape}, '
			f'Relevant tolerance: 1e-9.'
		)

	# Print the first 5 rows and columns of the matrices:
	print('\nFirst 5 rows and columns of the global version cost matrix:')
	print(cost_matrix_g[:5, :5])
	print('\nFirst 5 rows and columns of the local version cost matrix:')
	print(cost_matrix_l[:5, :5])

	print(f'\n{INFO_TAG}Global version stats:')
	print_stats_global_version(stats_g)
	print(f'\n{INFO_TAG}Local version stats:')
	print_stats_local_version(stats_l)

	return cost_matrix_g, cost_matrix_l


if __name__ == '__main__':
	# Test the class
	# feat_type = 'str'
	seed = 42
	n_graphs = 1000
	n_emb_dim = 200
	parallel = True
	ged_init_mode = 'lazy'  # 'eager' or 'lazy'
	compare_ged_model_with_global_and_local_env(
		seed=seed, n_graphs=n_graphs, n_emb_dim=n_emb_dim, parallel=parallel,
		ged_init_mode=ged_init_mode
	)

# 1. Profiling results:
#
# The following experiment pairs return the same cost matrix:
# - global v.s. local version, no parallelization, eager initialization.
# - global v.s. local version, no parallelization, lazy initialization.
# - global with Multiprocessing v.s. local with Multiprocessing, lazy initialization.
#
#
# 2. Analysis:
#
# # Comparison of the two versions:
#
# General Settings:
# - n_graphs: 1000
# - node numbers: 10-20
# - edge numbers: 20-50
# - n_emb_dim: 200
# - Coordinates as one label of np.array in AttrLabel,
#   which is optimized by the Eigen C++ library for vectorized operations.
#
# ## Without parallelization:
#
# ### local version (GEDEnv created for each pair of graphs):
#
# [info] The total run time for the GEDModel: 1040.524 s / 0.002083132 s per pair.
# [info] pairwise total time: 1020.674 s / 0.002043391 s per pair. (98.09% of total run time).
# [info] env setting time: 213.584 s / 0.000427595 s per pair. (20.53% of total run time).
# [info] graphs adding time: 692.614 s / 0.001386615 s per pair. (66.56% of total run time).
# [info] ged computing time: 105.811 s / 0.000211835 s per pair. (10.17% of total run time).
#
# The actual ged computation only takes 10.17% of the total run time, which is only 49.5% of the
# env setting time and 15.3% of the graphs adding time.
#
# Notice that in this version `init_option` is set to `LAZY_WITHOUT_SHUFFLED_COPIES`, so the costs
# between labels are actually computed when calling `GEDEnv.init()` method. This time is included in
# the `env setting time`. `ged computing time` only contains the time to fetch these costs and
# compute the ged.
#
# There is a huge gap that can be optimized!
#
# ### global version (GEDEnv created once for the model):
#
# #### Using `LAZY_WITHOUT_SHUFFLED_COPIES` init option (computing costs when actually needed):
#
# [info] The total run time for the GEDModel: 199.630 s / 0.000399659 s per pair.
# [info] env setting time: 0.034 s. (0.02% of total run time).
# [info] graphs adding time: 0.900 s / 0.000900076 s per graph. (0.45% of total run time).
# [info] ged computing time: 191.513 s / 0.000383410 s per pair. (95.93% of total run time).
#
# The total run time is significantly reduced to 199.630 s, which is only 19.2% of the local
# version. It even beats the local version with parallelization (~ 2.3x faster). The ged computing
# time (191.513 s) is 1.8x slower than the local version (105.811 s), which may be due to the
# lazy initialization.
#
# #### Using `EAGER_WITH_SHUFFLED_COPIES` init option (computing costs before the ged computation):
#
# [info] The total run time for the GEDModel: 258.232 s / 0.000516981 s per pair.
# [info] env setting time: 0.038 s. (0.01% of total run time).
# [info] eager label cost computing time: 151.453 s. (58.65% of total run time).
# [info] graphs adding time: 1.018 s / 0.001017674 s per graph. (0.39% of total run time).
# [info] ged computing time: 100.500 s / 0.000201202 s per pair. (38.92% of total run time).
#
# Section conclusion:
# - The lazy initialization of the costs may avoid unnecessary computations between label pairs,
#   e.g., if the node is inserted or deleted. Meanwhile, the eager version can avoid the multi-time
#   computation of the costs between the same label pairs. The problem is that it is done inside
#   C++ implementation, so it cannot be progressed by tqdm directly.
#   I have not tested how many of these cases exist during the computation.
# - At least for this specific case, the eager version is slower than the lazy version.
#
#
# ## With parallelization (n_jobs=10):
#
# ### local version (GEDEnv created for each pair of graphs):
#
# [info] The total run time for the GEDModel: 460.347 s / 0.000921615 s per pair.
# [info] pairwise total time: 1424.233 s / 0.002851317 s per pair. (309.38% of total run time).
# [info] env setting time: 285.454 s / 0.000571480 s per pair. (62.01% of total run time).
# [info] graphs adding time: 962.572 s / 0.001927071 s per pair. (209.10% of total run time).
# [info] ged computing time: 161.337 s / 0.000322997 s per pair. (35.05% of total run time).
#
# Similar to the non-parallelized version, the actual ged computation only takes 35.05% of the total
# run time, which is only 56.5% of the env setting time and 16.7% of the graphs adding time.
#
# ### global version (GEDEnv created once for the model):
#
# #### ✅ Using `LAZY_WITHOUT_SHUFFLED_COPIES` init option (computing costs when actually needed):
#
# [info] The total run time for the GEDModel: 18.410 s / 0.000036857 s per pair.
# [info] env setting time parallel: 1.428 s / 0.142769814 s per worker. (7.76% of total run time).
# [info] graphs adding time parallel: 6.165 s / 0.000616510 s per worker per graph. (33.49% of total run time).
#   Time per worker: [0.44879055 0.52007318 0.71203756 0.58835745 0.50336409 0.69282722
#   0.69138193 0.7250886 0.59162402 0.69155192].
# [info] ged computing time parallel: 139.406 s / 0.000279092 s per pair. (757.23% of total run time).
#
# #### Using `EAGER_WITH_SHUFFLED_COPIES` init option (computing costs before the ged computation):
#
# Slow, around 240 seconds for distance computation part, so not included here.
#
# ### Conclusion:
# - ✅ With parallelization, the global version with `LAZY_WITHOUT_SHUFFLED_COPIES` init option is
#   460.347 / 18.410 = 25.0x faster than the local version. Meanwhile, it is 199.630 / 18.410 = 10.8x
#   faster than the global version without parallelization!!! 🎉🎉🎉
# - ❌ the global version with `EAGER_WITH_SHUFFLED_COPIES` init option is much slower for unknown
#   reasons. It should be further investigated.
# - In the local version, env setting and graphs adding are performed N * (N - 1) / 2 times,
#   which is (N - 1) / 2 times more than the global version.
