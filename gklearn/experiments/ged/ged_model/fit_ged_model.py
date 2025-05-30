"""
fit_ged_model

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

	from gklearn.experiments.ged.ged_model.parallel_version import GEDModel

	if parallel is False:
		parallel = None
	elif parallel is True:
		parallel = 'imap_unordered'

	model = GEDModel(
		ed_method=ged_options['method'],
		edit_cost_fun=ged_options['edit_cost_fun'],
		init_edit_cost_constants=ged_options['edit_costs'],
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


def fit_model_ged_test(feat_type: str = 'str'):
	# Example usage:
	from gklearn.experiments.ged.ged_model.graph_generator import GraphGenerator
	if feat_type in ['str', 'int']:
		generator = GraphGenerator(
			num_graphs=10,
			max_num_nodes=5,
			min_num_nodes=3,
			max_num_edges=10,
			min_num_edges=5,
			node_feat_type=feat_type,
			edge_feat_type=feat_type,
			with_discrete_n_features=True,
			with_discrete_e_features=True,
			with_continuous_n_features=False,
			with_continuous_e_features=False,
			continuous_n_feature_dim=1,
			continuous_e_feature_dim=1,

		)
	else:
		generator = GraphGenerator(
			num_graphs=10,
			max_num_nodes=5,
			min_num_nodes=3,
			max_num_edges=10,
			min_num_edges=5,
			with_discrete_n_features=True,
			with_discrete_e_features=True,
			with_continuous_n_features=True,
			with_continuous_e_features=True,
			continuous_n_feature_dim=5,
			continuous_e_feature_dim=3,
			# 	node_features=['color', 'shape'],
			# 	edge_features=['weight'],
			# 	node_feature_values={'color': ['red', 'blue'], 'shape': ['circle', 'square']},
			# 	edge_feature_values={'weight': [1, 2, 3]},
		)
	run_fit(generator)


def run_fit(graph_generator):
	graphs = graph_generator.generate_graphs()

	# Set GED options:
	ged_options = {
		'method': 'BIPARTITE',
		'edit_cost_fun': 'NON_SYMBOLIC',
		'edit_costs': [3, 3, 1, 3, 3, 1],
		'optim_method': 'init',
		'repeats': 1
	}

	fit_settings = {
		'parallel': None,
		'n_jobs': 1, # min(12, max(os.cpu_count() - 2, 0)),
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



if __name__ == '__main__':
	# Test the class
	# feat_type = 'str'
	feat_type = 'int'
	fit_model_ged_test(feat_type=feat_type)
