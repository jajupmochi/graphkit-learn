"""
graph_generator

@Author: jajupmochi
@Date: May 22 2025
"""


class GraphGenerator:
	"""
	A class to generate random graphs for the Graph Edit Distance (GED) model with given
	constraints.

	Attributes:
		- num_graphs: Number of graphs to generate.
		- max_num_nodes: Maximum number of nodes in each graph.
		- min_num_nodes: Minimum number of nodes in each graph.
		- max_num_edges: Maximum number of edges in each graph.
		- min_num_edges: Minimum number of edges in each graph.
		- with_discrete_n_features: Whether to include discrete node features.
		- with_discrete_e_features: Whether to include discrete edge features.
		- with_continuous_n_features: Whether to include continuous node features.
		- with_continuous_e_features: Whether to include continuous edge features.
		- node_features: List of node feature names. Optional.
		- edge_features: List of edge feature names. Optional.
		- node_feature_values: Dictionary mapping node feature names to their possible values. Optional.
		- edge_feature_values: Dictionary mapping edge feature names to their possible values. Optional.
		- seed: Random seed for reproducibility. Default is None.
	"""


	def __init__(
			self,
			num_graphs: int,
			max_num_nodes: int,
			min_num_nodes: int,
			max_num_edges: int,
			min_num_edges: int,
			with_discrete_n_features: bool = False,
			with_discrete_e_features: bool = False,
			with_continuous_n_features: bool = False,
			with_continuous_e_features: bool = False,
			continuous_n_feature_dim: int = 10,
			continuous_e_feature_dim: int = 10,
			node_features: list = None,
			edge_features: list = None,
			node_feature_values: dict = None,
			edge_feature_values: dict = None,
			seed: int = None
	):
		self.num_graphs = num_graphs
		self.max_num_nodes = max_num_nodes
		self.min_num_nodes = min_num_nodes
		self.max_num_edges = max_num_edges
		self.min_num_edges = min_num_edges
		self.with_discrete_n_features = with_discrete_n_features
		self.with_discrete_e_features = with_discrete_e_features
		self.with_continuous_n_features = with_continuous_n_features
		self.with_continuous_e_features = with_continuous_e_features
		self.continuous_n_feature_dim = continuous_n_feature_dim
		self.continuous_e_feature_dim = continuous_e_feature_dim
		self.node_features = node_features if node_features else []
		self.edge_features = edge_features if edge_features else []
		self.node_feature_values = node_feature_values if node_feature_values else {}
		self.edge_feature_values = edge_feature_values if edge_feature_values else {}
		self.seed = seed
		if with_discrete_n_features and node_features is None:
			self.discrete_n_features = [str(i) for i in range(1, 100)]
		if with_discrete_e_features and edge_features is None:
			import string
			self.discrete_e_features = list(string.ascii_lowercase)


	def generate_graphs(self):
		"""
		Generates a list of random graphs based on the specified constraints.

		Returns:
			List of generated graphs.
		"""
		import numpy as np
		import networkx as nx
		import random

		rng = np.random.default_rng(self.seed)

		graphs = []

		for _ in range(self.num_graphs):
			num_nodes = rng.integers(self.min_num_nodes, self.max_num_nodes + 1)
			num_edges = rng.integers(self.min_num_edges, self.max_num_edges + 1)

			G = nx.Graph()
			G.add_nodes_from(range(num_nodes))

			if num_edges > 0:
				while G.number_of_edges() < num_edges:
					u = rng.integers(0, num_nodes)
					v = rng.integers(0, num_nodes)
					if u != v and not G.has_edge(u, v):
						G.add_edge(u, v)

			if self.with_discrete_n_features:
				if self.node_feature_values is None:
					for node in G.nodes():
						for feature in self.node_features:
							G.nodes[node][feature] = rng.choice(
								self.discrete_n_features
							)

				else:
					pass
			# for node in G.nodes():
			# 	for feature in self.node_features:
			# 		G.nodes[node][feature] = random.choice(
			# 			self.node_feature_values.get(feature, [0])
			# 		)

			if self.with_discrete_e_features:
				if self.edge_feature_values is None:
					for edge in G.edges():
						for feature in self.edge_features:
							G.edges[edge][feature] = rng.choice(
								self.discrete_e_features
							)
				else:
					pass
			# for edge in G.edges():
			# 	for feature in self.edge_features:
			# 		G.edges[edge][feature] = random.choice(
			# 			self.edge_feature_values.get(feature, [0])
			# 		)

			if self.with_continuous_n_features:
				if self.node_feature_values is None:
					for node in G.nodes():
						feature = rng.random(self.continuous_n_feature_dim)
						G.nodes[node]['feature'] = feature

				else:
					pass
			# for node in G.nodes():
			# 	for feature in self.node_features:
			# 		G.nodes[node][feature] = random.uniform(
			# 			self.node_feature_values.get(feature, (0, 1))[0],
			# 			self.node_feature_values.get(feature, (0, 1))[1]
			# 		)

			if self.with_continuous_e_features:
				if self.edge_feature_values is None:
					for edge in G.edges():
						feature = rng.random(self.continuous_e_feature_dim)
						G.edges[edge]['feature'] = feature

				else:
					pass
			# for edge in G.edges():
			# 	for feature in self.edge_features:
			# 		G.edges[edge][feature] = random.uniform(
			# 			self.edge_feature_values.get(feature, (0, 1))[0],
			# 			self.edge_feature_values.get(feature, (0, 1))[1]
			# 		)

			graphs.append(G)

		return graphs
