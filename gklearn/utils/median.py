"""
median

Methods to compute the median of a set of graphs.

@Author: linlin
@Date: 03.05.23
"""
import networkx as nx


def set_median_graph(graphs, dist_mat=None):
	"""Compute the set median of a set of graphs.

	Parameters
	----------
	graphs : list of networkx.Graph
		A set of graphs.
	dist_mat : numpy.matrix, optional
		The distance matrix of the graphs. If None, it will be computed using
		networkx.graph_edit_distance.

	Returns
	-------
	networkx.Graph
		The set median of the graphs.
	int
		The index of the set median in the input list.

	Authors
	-------
	Linlin Jia, Github Copilot (2023.05.03)
	"""
	# Compute the median of a set of graphs using Graph Edit Distance.
	if len(graphs) == 1:
		return graphs[0], 0

	if dist_mat is None:
		dist_mat = [[0 for _ in range(len(graphs))] for _ in range(len(graphs))]
		for i in range(len(graphs)):
			for j in range(i + 1, len(graphs)):
				dist_mat[i][j] = nx.graph_edit_distance(graphs[i], graphs[j])
				dist_mat[j][i] = dist_mat[i][j]
	else:
		if len(graphs) != len(dist_mat):
			raise ValueError(
				'The number of graphs and the distance matrix do not match.'
			)
	# print(dist_mat) # @TODO: remove
	dist_vec = [sum(v) for v in dist_mat]

	idx = dist_vec.index(min(dist_vec))
	set_median = graphs[idx]

	return set_median, idx
