"""Tests of graph kernels.
"""

import multiprocessing

import pytest


def chooseDataset(ds_name):
	"""Choose dataset according to name.
	"""
	from gklearn.utils.graphfiles import loadDataset

	# no node labels (and no edge labels).
	if ds_name == 'Alkane':
		ds_file = 'datasets/Alkane/dataset.ds'
		ds_y = 'datasets/Alkane/dataset_boiling_point_names.txt'
		Gn, y = loadDataset(ds_file, filename_y=ds_y)
		for G in Gn:
			for node in G.nodes:
				del G.nodes[node]['attributes']
	# node symbolic labels.
	elif ds_name == 'Acyclic':
		ds_file = 'datasets/acyclic/dataset_bps.ds'
		Gn, y = loadDataset(ds_file)
		for G in Gn:
			for node in G.nodes:
				del G.nodes[node]['attributes']
	# node non-symbolic labels.
	elif ds_name == 'Letter-med':
		ds_file = 'datasets/Letter-med/Letter-med_A.txt'
		Gn, y = loadDataset(ds_file)
	# node symbolic and non-symbolic labels (and edge symbolic labels).
	elif ds_name == 'AIDS':
		ds_file = 'datasets/AIDS/AIDS_A.txt'
		Gn, y = loadDataset(ds_file)

	# edge non-symbolic labels (no node labels).
	elif ds_name == 'Fingerprint_edge':
		import networkx as nx
		ds_file = 'datasets/Fingerprint/Fingerprint_A.txt'
		Gn, y = loadDataset(ds_file)
		Gn = [(idx, G) for idx, G in enumerate(Gn) if
		      nx.number_of_edges(G) != 0]
		idx = [G[0] for G in Gn]
		Gn = [G[1] for G in Gn]
		y = [y[i] for i in idx]
		for G in Gn:
			G.graph['node_attrs'] = []
			for node in G.nodes:
				del G.nodes[node]['attributes']
				del G.nodes[node]['x']
				del G.nodes[node]['y']
	# edge non-symbolic labels (and node non-symbolic labels).
	elif ds_name == 'Fingerprint':
		import networkx as nx
		ds_file = 'datasets/Fingerprint/Fingerprint_A.txt'
		Gn, y = loadDataset(ds_file)
		Gn = [(idx, G) for idx, G in enumerate(Gn) if
		      nx.number_of_edges(G) != 0]
		idx = [G[0] for G in Gn]
		Gn = [G[1] for G in Gn]
		y = [y[i] for i in idx]
	# edge symbolic and non-symbolic labels (and node symbolic and non-symbolic labels).
	elif ds_name == 'Cuneiform':
		import networkx as nx
		ds_file = 'datasets/Cuneiform/Cuneiform_A.txt'
		Gn, y = loadDataset(ds_file)

	Gn = Gn[0:3]
	y = y[0:3]

	return Gn, y


@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('weight,compute_method', [(0.01, 'geo'), (1, 'exp')])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_commonwalkkernel(ds_name, weight, compute_method):
	"""Test common walk kernel.
	"""
	from gklearn.kernels.commonWalkKernel import commonwalkkernel

	Gn, y = chooseDataset(ds_name)

	try:
		Kmatrix, run_time, idx = commonwalkkernel(
			Gn,
			node_label='atom',
			edge_label='bond_type',
			weight=weight,
			compute_method=compute_method,
			#                                             parallel=parallel,
			n_jobs=multiprocessing.cpu_count(),
			verbose=True
		)
	except Exception as exception:
		assert False, exception


@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('remove_totters', [True, False])
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_marginalizedkernel(ds_name, remove_totters):
	"""Test marginalized kernel.
	"""
	from gklearn.kernels.marginalizedKernel import marginalizedkernel

	Gn, y = chooseDataset(ds_name)

	try:
		Kmatrix, run_time = marginalizedkernel(
			Gn,
			node_label='atom',
			edge_label='bond_type',
			p_quit=0.5,
			n_iteration=2,
			remove_totters=remove_totters,
			#                                               parallel=parallel,
			n_jobs=multiprocessing.cpu_count(),
			verbose=True
		)
	except Exception as exception:
		assert False, exception


@pytest.mark.parametrize(
	'compute_method,ds_name,sub_kernel',
	[
		#            ('sylvester', 'Alkane', None),
		#            ('conjugate', 'Alkane', None),
		#            ('conjugate', 'AIDS', None),
		#            ('fp', 'Alkane', None),
		#            ('fp', 'AIDS', None),
		('spectral', 'Alkane', 'exp'),
		('spectral', 'Alkane', 'geo'),
	]
)
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_randomwalkkernel(ds_name, compute_method, sub_kernel):
	"""Test random walk kernel kernel.
	"""
	from gklearn.kernels.randomWalkKernel import randomwalkkernel
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	import functools

	Gn, y = chooseDataset(ds_name)

	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = [
		{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}]
	try:
		Kmatrix, run_time, idx = randomwalkkernel(
			Gn,
			compute_method=compute_method,
			weight=1e-3,
			p=None,
			q=None,
			edge_weight=None,
			node_kernels=sub_kernels,
			edge_kernels=sub_kernels,
			node_label='atom',
			edge_label='bond_type',
			sub_kernel=sub_kernel,
			#                                                  parallel=parallel,
			n_jobs=multiprocessing.cpu_count(),
			verbose=True
		)
	except Exception as exception:
		assert False, exception


@pytest.mark.parametrize(
	'ds_name', ['Alkane', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint']
)
# @pytest.mark.parametrize('parallel', ['imap_unordered', None])
@pytest.mark.parametrize('parallel', ['imap_unordered'])
def test_spkernel(ds_name, parallel):
	"""Test shortest path kernel.
	"""
	from gklearn.kernels.spKernel import spkernel
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	import functools

	Gn, y = chooseDataset(ds_name)

	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {
		'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel
	}
	try:
		Kmatrix, run_time, idx = spkernel(
			Gn, node_label='atom',
			node_kernels=sub_kernels,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(), verbose=True
		)
	except Exception as exception:
		assert False, exception


# @pytest.mark.parametrize('ds_name', ['Alkane', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint'])
@pytest.mark.parametrize(
	'ds_name', ['Alkane', 'Acyclic', 'Letter-med', 'AIDS', 'Fingerprint',
	            'Fingerprint_edge', 'Cuneiform']
)
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_structuralspkernel(ds_name, parallel):
	"""Test structural shortest path kernel.
	"""
	from gklearn.kernels.structuralspKernel import structuralspkernel
	from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
	import functools

	Gn, y = chooseDataset(ds_name)

	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {
		'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel
	}
	try:
		Kmatrix, run_time = structuralspkernel(
			Gn, node_label='atom',
			edge_label='bond_type', node_kernels=sub_kernels,
			edge_kernels=sub_kernels,
			parallel=parallel, n_jobs=multiprocessing.cpu_count(),
			verbose=True
		)
	except Exception as exception:
		assert False, exception


@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
# @pytest.mark.parametrize('k_func', ['MinMax', 'tanimoto', None])
@pytest.mark.parametrize('k_func', ['MinMax', 'tanimoto'])
@pytest.mark.parametrize('compute_method', ['trie', 'naive'])
def test_untilhpathkernel(ds_name, parallel, k_func, compute_method):
	"""Test path kernel up to length $h$.
	"""
	from gklearn.kernels.untilHPathKernel import untilhpathkernel

	Gn, y = chooseDataset(ds_name)

	try:
		Kmatrix, run_time = untilhpathkernel(
			Gn, node_label='atom',
			edge_label='bond_type',
			depth=2, k_func=k_func, compute_method=compute_method,
			parallel=parallel,
			n_jobs=multiprocessing.cpu_count(), verbose=True
		)
	except Exception as exception:
		assert False, exception


@pytest.mark.parametrize('ds_name', ['Alkane', 'AIDS'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_treeletkernel(ds_name, parallel):
	"""Test treelet kernel.
	"""
	from gklearn.kernels.treeletKernel import treeletkernel
	from gklearn.utils.kernels import polynomialkernel
	import functools

	Gn, y = chooseDataset(ds_name)

	pkernel = functools.partial(polynomialkernel, d=2, c=1e5)
	try:
		Kmatrix, run_time = treeletkernel(
			Gn,
			sub_kernel=pkernel,
			node_label='atom',
			edge_label='bond_type',
			parallel=parallel,
			n_jobs=multiprocessing.cpu_count(),
			verbose=True
		)
	except Exception as exception:
		assert False, exception


@pytest.mark.parametrize('ds_name', ['Acyclic'])
# @pytest.mark.parametrize('base_kernel', ['subtree', 'sp', 'edge'])
@pytest.mark.parametrize('base_kernel', ['subtree'])
@pytest.mark.parametrize('parallel', ['imap_unordered', None])
def test_weisfeilerlehmankernel(ds_name, parallel, base_kernel):
	"""Test Weisfeiler-Lehman kernel.
	"""
	from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel

	Gn, y = chooseDataset(ds_name)

	try:
		Kmatrix, run_time = weisfeilerlehmankernel(
			Gn,
			node_label='atom',
			edge_label='bond_type',
			height=2,
			base_kernel=base_kernel,
			parallel=parallel,
			n_jobs=multiprocessing.cpu_count(),
			verbose=True
		)
	except Exception as exception:
		assert False, exception


if __name__ == "__main__":
	#    test_spkernel('Alkane', 'imap_unordered')
	test_structuralspkernel('Fingerprint_edge', 'imap_unordered')
