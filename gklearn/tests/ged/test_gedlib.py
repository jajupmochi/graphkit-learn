"""
test_gedlib

The test cases for the gedlib module.

@Author: linlin
@Date: 24.04.23
"""


def choose_dataset(ds_name):
	"""Choose dataset according to name.
	"""
	from gklearn.dataset import Dataset
	import os

	current_path = os.path.dirname(os.path.realpath(__file__)) + '/'
	root = current_path + '../../../datasets/'

	if ds_name == 'MUTAG':
		dataset = Dataset('MUTAG', root=root)
	# dataset.trim_dataset(edge_required=False)

	dataset.cut_graphs(range(0, 3))

	return dataset


def test_gedlib():
	"""**1. Get dataset.**"""
	graphs = choose_dataset('MUTAG').graphs

	"""**2. Compute graph edit distance.**"""
	try:
		nl_names = list(graphs[0].nodes[list(graphs[0].nodes)[0]].keys())
		el_names = list(graphs[0].edges[list(graphs[0].edges)[0]].keys())
		from gklearn.ged import GEDModel
		ged_model = GEDModel(
			ed_method='BIPARTITE',
			edit_cost_fun='CONSTANT',
			init_edit_cost_constants=[3, 3, 1, 3, 3, 1],
			optim_method='init',
			node_labels=nl_names, edge_labels=el_names,
			parallel=None,
			n_jobs=None,
			chunksize=None,
			copy_graphs=True,  # make sure it is a full deep copy. and faster!
			verbose=2
		)

		# Train model.
		dis_mat_train = ged_model.fit_transform(
			graphs, save_dm_train=False, repeats=1
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


if __name__ == '__main__':
	test_gedlib()
