"""Tests of GEDEnv.
"""


def test_GEDEnv():
	"""Test GEDEnv.
	"""
	"""**1.   Get dataset.**"""

	from gklearn.utils import Dataset

	# Predefined dataset name, use dataset "MUTAG".
	ds_name = 'MUTAG'

	# Initialize a Dataset.
	dataset = Dataset()
	# Load predefined dataset "MUTAG".
	dataset.load_predefined_dataset(ds_name)
	graph1 = dataset.graphs[0]
	graph2 = dataset.graphs[1]

	"""**2.  Compute graph edit distance.**"""

	try:
		from gklearn.ged.env import GEDEnv

		ged_env = GEDEnv()  # initailize GED environment.
		ged_env.set_edit_cost(
			'CONSTANT',  # GED cost type.
			edit_cost_constants=[3, 3, 1, 3, 3, 1]  # edit costs.
		)
		ged_env.add_nx_graph(graph1, '')  # add graph1
		ged_env.add_nx_graph(graph2, '')  # add graph2
		listID = ged_env.get_all_graph_ids()  # get list IDs of graphs
		ged_env.init(
			init_type='LAZY_WITHOUT_SHUFFLED_COPIES'
		)  # initialize GED environment.
		options = {
			'initialization_method': 'RANDOM',  # or 'NODE', etc.
			'threads': 1  # parallel threads.
		}
		ged_env.set_method(
			'BIPARTITE',  # GED method.
			options  # options for GED method.
		)
		ged_env.init_method()  # initialize GED method.

		ged_env.run_method(listID[0], listID[1])  # run.

		pi_forward = ged_env.get_forward_map(
			listID[0], listID[1]
		)  # forward map.
		pi_backward = ged_env.get_backward_map(
			listID[0], listID[1]
		)  # backward map.
		dis = ged_env.get_upper_bound(
			listID[0], listID[1]
		)  # GED bewteen two graphs.

		import networkx as nx
		assert len(pi_forward) == nx.number_of_nodes(graph1), len(
			pi_backward
		) == nx.number_of_nodes(graph2)

	except Exception as exception:
		assert False, exception


if __name__ == "__main__":
	test_GEDEnv()
