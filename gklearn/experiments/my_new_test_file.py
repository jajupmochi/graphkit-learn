"""Tests of graph kernels.
"""

import multiprocessing
import numpy as np


##############################################################################

def chooseDataset(ds_name):
	"""Choose dataset according to name.
	"""
	from gklearn.dataset import Dataset
	import os

	current_path = os.path.dirname(os.path.realpath(__file__)) + '/'
	root = current_path + '../../datasets/'

	# no labels at all.
	if ds_name == 'Alkane_unlabeled':
		dataset = Dataset('Alkane_unlabeled', root=root)
		dataset.trim_dataset(edge_required=False)
		dataset.cut_graphs(range(1, 10))
	# node symbolic labels only.
	elif ds_name == 'Acyclic':
		dataset = Dataset('Acyclic', root=root)
		dataset.trim_dataset(edge_required=False)
	# node non-symbolic labels only.
	elif ds_name == 'Letter-med':
		dataset = Dataset('Letter-med', root=root)
		dataset.trim_dataset(edge_required=False)
	# node symbolic + non-symbolic labels + edge symbolic labels.
	elif ds_name == 'AIDS':
		dataset = Dataset('AIDS', root=root)
		dataset.trim_dataset(edge_required=False)
	# node non-symbolic labels + edge non-symbolic labels.
	elif ds_name == 'Fingerprint':
		dataset = Dataset('Fingerprint', root=root)
		dataset.trim_dataset(edge_required=True)
	# edge symbolic only.
	elif ds_name == 'MAO':
		dataset = Dataset('MAO', root=root)
		dataset.trim_dataset(edge_required=True)
		irrelevant_labels = {'node_labels': ['atom_symbol'], 'node_attrs': ['x', 'y']}
		dataset.remove_labels(**irrelevant_labels)
	# edge non-symbolic labels only.
	elif ds_name == 'Fingerprint_edge':
		dataset = Dataset('Fingerprint', root=root)
		dataset.trim_dataset(edge_required=True)
		irrelevant_labels = {'edge_attrs': ['orient', 'angle']}
		dataset.remove_labels(**irrelevant_labels)
	# node symbolic and non-symbolic labels + edge symbolic and non-symbolic labels.
	elif ds_name == 'Cuneiform':
		dataset = Dataset('Cuneiform', root=root)
		dataset.trim_dataset(edge_required=True)

	dataset.cut_graphs(range(0, 3))

	return dataset


def assert_equality(compute_fun, **kwargs):
	"""Check if outputs are the same using different methods to compute.

	Parameters
	----------
	compute_fun : function
		The function to compute the kernel, with the same key word arguments as
		kwargs.
	**kwargs : dict
		The key word arguments over the grid of which the kernel results are
		compared.

	Returns
	-------
	None.
	"""
	from sklearn.model_selection import ParameterGrid
	param_grid = ParameterGrid(kwargs)

	result_lists = [[], [], []]
	for params in list(param_grid):
		results = compute_fun(**params)
		for rs, lst in zip(results, result_lists):
			lst.append(rs)
	for lst in result_lists:
		for i in range(len(lst[:-1])):
			assert np.array_equal(lst[i], lst[i + 1])


# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def my_test_CommonWalk(ds_name, weight, compute_method):
	"""Test common walk kernel.
	"""
	def compute(parallel=None):
		from gklearn.kernels import CommonWalk
		import networkx as nx

		dataset = chooseDataset(ds_name)
		dataset.load_graphs([g for g in dataset.graphs if nx.number_of_nodes(g) > 1])

		try:
			graph_kernel = CommonWalk(node_labels=dataset.node_labels,
						edge_labels=dataset.edge_labels,
						ds_infos=dataset.get_dataset_infos(keys=['directed']),
						weight=weight,
						compute_method=compute_method)
			gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
				parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
			kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
				parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
			kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
				parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel

	assert_equality(compute, parallel=[None, 'imap_unordered'])


if __name__ == "__main__":
 	# test_list_graph_kernels()
#	test_spkernel('Alkane_unlabeled', 'imap_unordered')
 	# test_ShortestPath('Alkane_unlabeled')
# 	test_StructuralSP('Fingerprint_edge', 'imap_unordered')
 	# test_StructuralSP('Acyclic')
# 	test_StructuralSP('Cuneiform', None)
#  	test_WLSubtree('MAO') # 'Alkane_unlabeled', 'Acyclic', 'AIDS'
#	test_RandomWalk('Acyclic', 'sylvester', None, 'imap_unordered')
#	test_RandomWalk('Acyclic', 'conjugate', None, 'imap_unordered')
#	test_RandomWalk('Acyclic', 'fp', None, None)
#	test_RandomWalk('Acyclic', 'spectral', 'exp', 'imap_unordered')
 	# test_CommonWalk('Acyclic', 0.01, 'geo')
	my_test_CommonWalk('AIDS', 0.01, 'geo')
    # test_Marginalized('Acyclic', False)
 	# test_ShortestPath('Acyclic')
# 	 test_PathUpToH('Acyclic', 'MinMax')
 	# test_Treelet('AIDS')
# 	test_SylvesterEquation('Acyclic')
# 	test_ConjugateGradient('Acyclic')
# 	test_FixedPoint('Acyclic')
# 	test_SpectralDecomposition('Acyclic', 'exp')