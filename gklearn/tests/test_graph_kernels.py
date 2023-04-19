"""Tests of graph kernels.
"""

import multiprocessing

import numpy as np
import pytest


##############################################################################

def test_list_graph_kernels():
	"""
	"""
	from gklearn.kernels import GRAPH_KERNELS, list_of_graph_kernels
	assert list_of_graph_kernels() == [i for i in GRAPH_KERNELS]


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
		irrelevant_labels = {
			'node_labels': ['atom_symbol'],
			'node_attrs': ['x', 'y']
		}
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


@pytest.mark.parametrize('ds_name', ['Alkane_unlabeled', 'AIDS'])
@pytest.mark.parametrize('weight,compute_method', [(0.01, 'geo'), (1, 'exp')])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_CommonWalk(ds_name, weight, compute_method):
	"""Test common walk kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import CommonWalk
		import networkx as nx

		dataset = chooseDataset(ds_name)
		dataset.load_graphs(
			[g for g in dataset.graphs if nx.number_of_nodes(g) > 1]
		)

		try:
			graph_kernel = CommonWalk(
				node_labels=dataset.node_labels,
				edge_labels=dataset.edge_labels,
				ds_infos=dataset.get_dataset_infos(keys=['directed']),
				weight=weight,
				compute_method=compute_method
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel, n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0], dataset.graphs[1:],
				parallel=parallel, n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0], dataset.graphs[1],
				parallel=parallel, n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(compute, parallel=[None, 'imap_unordered'])


@pytest.mark.parametrize('ds_name', ['Alkane_unlabeled', 'AIDS'])
@pytest.mark.parametrize('remove_totters', [False])  # [True, False])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_Marginalized(ds_name, remove_totters):
	"""Test marginalized kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import Marginalized

		dataset = chooseDataset(ds_name)

		try:
			graph_kernel = Marginalized(
				node_labels=dataset.node_labels,
				edge_labels=dataset.edge_labels,
				ds_infos=dataset.get_dataset_infos(
					keys=['directed']
				),
				p_quit=0.5,
				n_iteration=2,
				remove_totters=remove_totters
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(compute, parallel=[None, 'imap_unordered'])


@pytest.mark.parametrize('ds_name', ['Acyclic'])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_SylvesterEquation(ds_name):
	"""Test sylvester equation kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import SylvesterEquation

		dataset = chooseDataset(ds_name)

		try:
			graph_kernel = SylvesterEquation(
				ds_infos=dataset.get_dataset_infos(keys=['directed']),
				weight=1e-3,
				p=None,
				q=None,
				edge_weight=None
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(compute, parallel=[None, 'imap_unordered'])


@pytest.mark.parametrize('ds_name', ['Acyclic', 'AIDS'])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_ConjugateGradient(ds_name):
	"""Test conjugate gradient kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import ConjugateGradient
		from gklearn.utils.kernels import kronecker_delta_kernel, \
			gaussian_kernel, \
			kernelproduct
		import functools

		dataset = chooseDataset(ds_name)

		mixkernel = functools.partial(
			kernelproduct, kronecker_delta_kernel,
			gaussian_kernel
		)
		sub_kernels = {
			'symb': kronecker_delta_kernel, 'nsymb': gaussian_kernel,
			'mix': mixkernel
		}

		try:
			graph_kernel = ConjugateGradient(
				node_labels=dataset.node_labels,
				node_attrs=dataset.node_attrs,
				edge_labels=dataset.edge_labels,
				edge_attrs=dataset.edge_attrs,
				ds_infos=dataset.get_dataset_infos(keys=['directed']),
				weight=1e-3,
				p=None,
				q=None,
				edge_weight=None,
				node_kernels=sub_kernels,
				edge_kernels=sub_kernels
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(compute, parallel=[None, 'imap_unordered'])


@pytest.mark.parametrize('ds_name', ['Acyclic', 'AIDS'])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_FixedPoint(ds_name):
	"""Test fixed point kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import FixedPoint
		from gklearn.utils.kernels import deltakernel, gaussiankernel, \
			kernelproduct
		import functools

		dataset = chooseDataset(ds_name)

		mixkernel = functools.partial(
			kernelproduct, deltakernel,
			gaussiankernel
		)
		sub_kernels = {
			'symb': deltakernel, 'nsymb': gaussiankernel,
			'mix': mixkernel
		}

		try:
			graph_kernel = FixedPoint(
				node_labels=dataset.node_labels,
				node_attrs=dataset.node_attrs,
				edge_labels=dataset.edge_labels,
				edge_attrs=dataset.edge_attrs,
				ds_infos=dataset.get_dataset_infos(keys=['directed']),
				weight=1e-3,
				p=None,
				q=None,
				edge_weight=None,
				node_kernels=sub_kernels,
				edge_kernels=sub_kernels
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(compute, parallel=[None, 'imap_unordered'])


@pytest.mark.parametrize('ds_name', ['Acyclic'])
@pytest.mark.parametrize('sub_kernel', ['exp', 'geo'])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_SpectralDecomposition(ds_name, sub_kernel):
	"""Test spectral decomposition kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import SpectralDecomposition

		dataset = chooseDataset(ds_name)

		try:
			graph_kernel = SpectralDecomposition(
				ds_infos=dataset.get_dataset_infos(keys=['directed']),
				weight=1e-3,
				p=None,
				q=None,
				edge_weight=None,
				sub_kernel=sub_kernel
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(compute, parallel=[None, 'imap_unordered'])


# @pytest.mark.parametrize(
#		'compute_method,ds_name,sub_kernel',
#		[
#			('sylvester', 'Alkane_unlabeled', None),
#			('conjugate', 'Alkane_unlabeled', None),
#			('conjugate', 'AIDS', None),
#			('fp', 'Alkane_unlabeled', None),
#			('fp', 'AIDS', None),
#			('spectral', 'Alkane_unlabeled', 'exp'),
#			('spectral', 'Alkane_unlabeled', 'geo'),
#		]
# )
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
# def test_RandomWalk(ds_name, compute_method, sub_kernel, parallel):
#	"""Test random walk kernel.
#	"""
#	from gklearn.kernels import RandomWalk
#	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
#	import functools
#
#	dataset = chooseDataset(ds_name)

#	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
#	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
# #	try:
#	graph_kernel = RandomWalk(node_labels=dataset.node_labels,
#						node_attrs=dataset.node_attrs,
#						edge_labels=dataset.edge_labels,
#						edge_attrs=dataset.edge_attrs,
#						ds_infos=dataset.get_dataset_infos(keys=['directed']),
#						compute_method=compute_method,
#						weight=1e-3,
#						p=None,
#						q=None,
#						edge_weight=None,
#						node_kernels=sub_kernels,
#						edge_kernels=sub_kernels,
#						sub_kernel=sub_kernel)
#	gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
#		parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
#	kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
#		parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
#	kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
#		parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)

#	except Exception as exception:
#		assert False, exception


@pytest.mark.parametrize(
	'ds_name',
	['Alkane_unlabeled', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint']
)
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_ShortestPath(ds_name):
	"""Test shortest path kernel.
	"""


	def compute(parallel=None, fcsp=None):
		from gklearn.kernels import ShortestPath
		from gklearn.utils.kernels import deltakernel, gaussiankernel, \
			kernelproduct
		import functools

		dataset = chooseDataset(ds_name)

		mixkernel = functools.partial(
			kernelproduct, deltakernel,
			gaussiankernel
		)
		sub_kernels = {
			'symb': deltakernel, 'nsymb': gaussiankernel,
			'mix': mixkernel
		}
		try:
			graph_kernel = ShortestPath(
				node_labels=dataset.node_labels,
				node_attrs=dataset.node_attrs,
				ds_infos=dataset.get_dataset_infos(
					keys=['directed']
				),
				fcsp=fcsp,
				node_kernels=sub_kernels
			)

			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(
		compute, parallel=[None, 'imap_unordered'], fcsp=[True, False]
	)


# @pytest.mark.parametrize('ds_name', ['Alkane_unlabeled', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint'])
@pytest.mark.parametrize(
	'ds_name', [
		'Alkane_unlabeled', 'Acyclic', 'Letter-med', 'AIDS',
		'Fingerprint', 'Fingerprint_edge', 'Cuneiform'
	]
)
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_StructuralSP(ds_name):
	"""Test structural shortest path kernel.
	"""


	def compute(parallel=None, fcsp=None):
		from gklearn.kernels import StructuralSP
		from gklearn.utils.kernels import deltakernel, gaussiankernel, \
			kernelproduct
		import functools

		dataset = chooseDataset(ds_name)

		mixkernel = functools.partial(
			kernelproduct, deltakernel,
			gaussiankernel
		)
		sub_kernels = {
			'symb': deltakernel, 'nsymb': gaussiankernel,
			'mix': mixkernel
		}
		try:
			graph_kernel = StructuralSP(
				node_labels=dataset.node_labels,
				edge_labels=dataset.edge_labels,
				node_attrs=dataset.node_attrs,
				edge_attrs=dataset.edge_attrs,
				ds_infos=dataset.get_dataset_infos(
					keys=['directed']
				),
				fcsp=fcsp,
				node_kernels=sub_kernels,
				edge_kernels=sub_kernels
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(
		compute, parallel=[None, 'imap_unordered'], fcsp=[True, False]
	)


@pytest.mark.parametrize('ds_name', ['Alkane_unlabeled', 'AIDS'])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
# @pytest.mark.parametrize('k_func', ['MinMax', 'tanimoto', None])
@pytest.mark.parametrize('k_func', ['MinMax', 'tanimoto'])
# @pytest.mark.parametrize('compute_method', ['trie', 'naive'])
def test_PathUpToH(ds_name, k_func):
	"""Test path kernel up to length $h$.
	"""


	def compute(parallel=None, compute_method=None):
		from gklearn.kernels import PathUpToH

		dataset = chooseDataset(ds_name)

		try:
			graph_kernel = PathUpToH(
				node_labels=dataset.node_labels,
				edge_labels=dataset.edge_labels,
				ds_infos=dataset.get_dataset_infos(
					keys=['directed']
				),
				depth=2, k_func=k_func,
				compute_method=compute_method
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(
		compute, parallel=[None, 'imap_unordered'],
		compute_method=['trie', 'naive']
	)


@pytest.mark.parametrize('ds_name', ['Alkane_unlabeled', 'AIDS'])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_Treelet(ds_name):
	"""Test treelet kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import Treelet
		from gklearn.utils.kernels import polynomialkernel
		import functools

		dataset = chooseDataset(ds_name)

		pkernel = functools.partial(polynomialkernel, d=2, c=1e5)
		try:
			graph_kernel = Treelet(
				node_labels=dataset.node_labels,
				edge_labels=dataset.edge_labels,
				ds_infos=dataset.get_dataset_infos(keys=['directed']),
				sub_kernel=pkernel
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel, n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0], dataset.graphs[1:],
				parallel=parallel, n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0], dataset.graphs[1],
				parallel=parallel, n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	assert_equality(compute, parallel=[None, 'imap_unordered'])


@pytest.mark.parametrize(
	'ds_name', ['Alkane_unlabeled', 'Acyclic', 'MAO', 'AIDS']
)
# @pytest.mark.parametrize('base_kernel', ['subtree', 'sp', 'edge'])
# @pytest.mark.parametrize('base_kernel', ['subtree'])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_WLSubtree(ds_name):
	"""Test Weisfeiler-Lehman subtree kernel.
	"""


	def compute(parallel=None):
		from gklearn.kernels import WLSubtree

		dataset = chooseDataset(ds_name)

		try:
			graph_kernel = WLSubtree(
				node_labels=dataset.node_labels,
				edge_labels=dataset.edge_labels,
				ds_infos=dataset.get_dataset_infos(
					keys=['directed']
				),
				height=2
			)
			gram_matrix, run_time = graph_kernel.compute(
				dataset.graphs,
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel_list, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1:],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)
			kernel, run_time = graph_kernel.compute(
				dataset.graphs[0],
				dataset.graphs[1],
				parallel=parallel,
				n_jobs=multiprocessing.cpu_count(),
				verbose=True
			)

		except Exception as exception:
			print(repr(exception))
			assert False, exception
		else:
			return gram_matrix, kernel_list, kernel


	# assert_equality(compute, parallel=[None, 'imap_unordered'])
	assert_equality(compute, parallel=[None])  # @TODO: parallel returns different results.


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
	test_CommonWalk('Alkane_unlabeled', 0.01, 'geo')
# test_Marginalized('Acyclic', False)
# test_ShortestPath('Acyclic')
# 	 test_PathUpToH('Acyclic', 'MinMax')
# test_Treelet('AIDS')
# 	test_SylvesterEquation('Acyclic')
# 	test_ConjugateGradient('Acyclic')
# 	test_FixedPoint('Acyclic')
# 	test_SpectralDecomposition('Acyclic', 'exp')
