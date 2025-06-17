"""Tests of graph kernels.
"""

import pytest
import multiprocessing


def chooseDataset(ds_name):
	"""Choose dataset according to name.
	"""
	from gklearn.utils import Dataset
	
	dataset = Dataset()

	# no node labels (and no edge labels).
	if ds_name == 'Alkane':
		dataset.load_predefined_dataset(ds_name)
		dataset.trim_dataset(edge_required=False)
		irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']}
		dataset.remove_labels(**irrelevant_labels)
	# node symbolic labels.
	elif ds_name == 'Acyclic':
		dataset.load_predefined_dataset(ds_name)
		dataset.trim_dataset(edge_required=False)
		irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']}
		dataset.remove_labels(**irrelevant_labels)
	# node non-symbolic labels.
	elif ds_name == 'Letter-med':
		dataset.load_predefined_dataset(ds_name)
		dataset.trim_dataset(edge_required=False)
	# node symbolic and non-symbolic labels (and edge symbolic labels).
	elif ds_name == 'AIDS':
		dataset.load_predefined_dataset(ds_name)
		dataset.trim_dataset(edge_required=False)
	# edge non-symbolic labels (no node labels).
	elif ds_name == 'Fingerprint_edge':
		dataset.load_predefined_dataset('Fingerprint')
		dataset.trim_dataset(edge_required=True)
		irrelevant_labels = {'edge_attrs': ['orient', 'angle']}
		dataset.remove_labels(**irrelevant_labels)
	# edge non-symbolic labels (and node non-symbolic labels).
	elif ds_name == 'Fingerprint':
		dataset.load_predefined_dataset(ds_name)
		dataset.trim_dataset(edge_required=True)
	# edge symbolic and non-symbolic labels (and node symbolic and non-symbolic labels).
	elif ds_name == 'Cuneiform':
		dataset.load_predefined_dataset(ds_name)
		dataset.trim_dataset(edge_required=True)
		
	dataset.cut_graphs(range(0, 3))
	
	return dataset


@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('weight,compute_method', [(0.01, 'geo'), (1, 'exp')])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_CommonWalk(ds_name, parallel, weight, compute_method):
	"""Test common walk kernel.
	"""
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
		assert False, exception
		
		
@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('remove_totters', [False]) #[True, False])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_Marginalized(ds_name, parallel, remove_totters):
	"""Test marginalized kernel.
	"""
	from gklearn.kernels import Marginalized
	
	dataset = chooseDataset(ds_name)
	
	try:
		graph_kernel = Marginalized(node_labels=dataset.node_labels,
					edge_labels=dataset.edge_labels,
					ds_infos=dataset.get_dataset_infos(keys=['directed']),
					p_quit=0.5,
					n_iteration=2,
					remove_totters=remove_totters)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)

	except Exception as exception:
		assert False, exception
		
		
@pytest.mark.parametrize('ds_name', ['Acyclic'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_SylvesterEquation(ds_name, parallel):
	"""Test sylvester equation kernel.
	"""
	from gklearn.kernels import SylvesterEquation
	
	dataset = chooseDataset(ds_name)

	try:
		graph_kernel = SylvesterEquation(
							ds_infos=dataset.get_dataset_infos(keys=['directed']),
							weight=1e-3,
							p=None,
							q=None,
							edge_weight=None)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
	
	except Exception as exception:
		assert False, exception
		
		
@pytest.mark.parametrize('ds_name', ['Acyclic', 'AIDS'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_ConjugateGradient(ds_name, parallel):
	"""Test conjugate gradient kernel.
	"""
	from gklearn.kernels import ConjugateGradient
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	import functools
	
	dataset = chooseDataset(ds_name)
	
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}

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
							edge_kernels=sub_kernels)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
	
	except Exception as exception:
		assert False, exception
		
		
@pytest.mark.parametrize('ds_name', ['Acyclic', 'AIDS'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_FixedPoint(ds_name, parallel):
	"""Test fixed point kernel.
	"""
	from gklearn.kernels import FixedPoint
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	import functools
	
	dataset = chooseDataset(ds_name)
	
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}

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
							edge_kernels=sub_kernels)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
	
	except Exception as exception:
		assert False, exception
		
		
@pytest.mark.parametrize('ds_name', ['Acyclic'])
@pytest.mark.parametrize('sub_kernel', ['exp', 'geo'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_SpectralDecomposition(ds_name, sub_kernel, parallel):
	"""Test spectral decomposition kernel.
	"""
	from gklearn.kernels import SpectralDecomposition
	
	dataset = chooseDataset(ds_name)

	try:
		graph_kernel = SpectralDecomposition(
							ds_infos=dataset.get_dataset_infos(keys=['directed']),
							weight=1e-3,
							p=None,
							q=None,
							edge_weight=None,
							sub_kernel=sub_kernel)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
	
	except Exception as exception:
		assert False, exception
		
		
# @pytest.mark.parametrize(
#		'compute_method,ds_name,sub_kernel',
#		[
#			('sylvester', 'Alkane', None),
#			('conjugate', 'Alkane', None),
#			('conjugate', 'AIDS', None),
#			('fp', 'Alkane', None),
#			('fp', 'AIDS', None),
#			('spectral', 'Alkane', 'exp'),
#			('spectral', 'Alkane', 'geo'),
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

		
@pytest.mark.parametrize('ds_name', ['Alkane', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_ShortestPath(ds_name, parallel):
	"""Test shortest path kernel.
	"""
	from gklearn.kernels import ShortestPath
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	import functools
	
	dataset = chooseDataset(ds_name)
	
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	try:
		graph_kernel = ShortestPath(node_labels=dataset.node_labels,
					node_attrs=dataset.node_attrs,
					ds_infos=dataset.get_dataset_infos(keys=['directed']),
					node_kernels=sub_kernels)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)

	except Exception as exception:
		assert False, exception


#@pytest.mark.parametrize('ds_name', ['Alkane', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint'])
@pytest.mark.parametrize('ds_name', ['Alkane', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint', 'Fingerprint_edge', 'Cuneiform'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_StructuralSP(ds_name, parallel):
	"""Test structural shortest path kernel.
	"""
	from gklearn.kernels import StructuralSP
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	import functools
	
	dataset = chooseDataset(ds_name)
	
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	try:
		graph_kernel = StructuralSP(node_labels=dataset.node_labels,
					 edge_labels=dataset.edge_labels, 
					 node_attrs=dataset.node_attrs,
					 edge_attrs=dataset.edge_attrs,
					 ds_infos=dataset.get_dataset_infos(keys=['directed']),
					 node_kernels=sub_kernels,
					 edge_kernels=sub_kernels)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)

	except Exception as exception:
		assert False, exception


@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
#@pytest.mark.parametrize('k_func', ['MinMax', 'tanimoto', None])
@pytest.mark.parametrize('k_func', ['MinMax', 'tanimoto'])
@pytest.mark.parametrize('compute_method', ['trie', 'naive'])
def test_PathUpToH(ds_name, parallel, k_func, compute_method):
	"""Test path kernel up to length $h$.
	"""
	from gklearn.kernels import PathUpToH
	
	dataset = chooseDataset(ds_name)
	
	try:
		graph_kernel = PathUpToH(node_labels=dataset.node_labels,
					 edge_labels=dataset.edge_labels,
					 ds_infos=dataset.get_dataset_infos(keys=['directed']),
					 depth=2, k_func=k_func, compute_method=compute_method)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
	except Exception as exception:
		assert False, exception
	
	
@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_Treelet(ds_name, parallel):
	"""Test treelet kernel.
	"""
	from gklearn.kernels import Treelet
	from gklearn.utils.kernels import polynomialkernel
	import functools
	
	dataset = chooseDataset(ds_name)

	pkernel = functools.partial(polynomialkernel, d=2, c=1e5)	
	try:
		graph_kernel = Treelet(node_labels=dataset.node_labels,
					 edge_labels=dataset.edge_labels,
					 ds_infos=dataset.get_dataset_infos(keys=['directed']),
					 sub_kernel=pkernel)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
	except Exception as exception:
		assert False, exception
		
		
@pytest.mark.parametrize('ds_name', ['Acyclic'])
#@pytest.mark.parametrize('base_kernel', ['subtree', 'sp', 'edge'])
# @pytest.mark.parametrize('base_kernel', ['subtree'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_WLSubtree(ds_name, parallel):
	"""Test Weisfeiler-Lehman subtree kernel.
	"""
	from gklearn.kernels import WLSubtree
	
	dataset = chooseDataset(ds_name)

	try:
		graph_kernel = WLSubtree(node_labels=dataset.node_labels,
					 edge_labels=dataset.edge_labels,
					 ds_infos=dataset.get_dataset_infos(keys=['directed']),
					 height=2)
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel_list, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1:],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
		kernel, run_time = graph_kernel.compute(dataset.graphs[0], dataset.graphs[1],
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True)
	except Exception as exception:
		assert False, exception
		

if __name__ == "__main__":
#	test_spkernel('Alkane', 'imap_unordered')
# 	test_StructuralSP('Fingerprint_edge', 'imap_unordered')
	test_WLSubtree('Acyclic', 'imap_unordered')
#	test_RandomWalk('Acyclic', 'sylvester', None, 'imap_unordered')
#	test_RandomWalk('Acyclic', 'conjugate', None, 'imap_unordered')
#	test_RandomWalk('Acyclic', 'fp', None, None)
#	test_RandomWalk('Acyclic', 'spectral', 'exp', 'imap_unordered')