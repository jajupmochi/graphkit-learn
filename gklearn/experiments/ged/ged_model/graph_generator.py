"""
graph_generator

@Author: jajupmochi
@Date: May 22 2025
"""
import string


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
			node_feat_type: str | None = 'str',
			edge_feat_type: str | None = 'str',
			with_discrete_n_features: bool = False,
			with_discrete_e_features: bool = False,
			with_continuous_n_features: bool = False,
			with_continuous_e_features: bool = False,
			continuous_n_feature_key: str | None = 'feature',
			continuous_e_feature_key: str | None = 'feature',
			continuous_n_feature_dim: int | None = 10,
			continuous_e_feature_dim: int | None = 10,
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
		self.node_feat_type = node_feat_type
		self.edge_feat_type = edge_feat_type
		self.with_discrete_n_features = with_discrete_n_features
		self.with_discrete_e_features = with_discrete_e_features
		self.with_continuous_n_features = with_continuous_n_features
		self.with_continuous_e_features = with_continuous_e_features
		self.continuous_n_feature_key = continuous_n_feature_key
		self.continuous_e_feature_key = continuous_e_feature_key
		self.continuous_n_feature_dim = continuous_n_feature_dim
		self.continuous_e_feature_dim = continuous_e_feature_dim
		self.node_features = node_features if node_features else []
		self.edge_features = edge_features if edge_features else []
		self.node_feature_values = node_feature_values if node_feature_values else {}
		self.edge_feature_values = edge_feature_values if edge_feature_values else {}
		self.seed = seed
		if with_discrete_n_features and node_features is None:
			self.discrete_n_features = self.generate_discrete_feats()
		if with_discrete_e_features and edge_features is None:
			self.discrete_e_features = self.generate_discrete_feats()


	def generate_discrete_feats(self):

		if self.node_feat_type == 'str':
			discrete_features = self.generate_symbolic_feats()
		elif self.node_feat_type == 'int':
			discrete_features = [i for i in range(1, 100)]
		elif self.node_feat_type == 'float':
			discrete_features = None
		else:
			raise ValueError(
				"node_feat_type must be 'str' or 'int'."
			)
		return discrete_features


	def generate_symbolic_feats(self):
		flist = list(string.ascii_lowercase)
		# Combine some letters to make them more realistic:
		count = 0
		for i in string.ascii_lowercase:
			for j in string.ascii_lowercase:
				if i != j:
					flist.append(i + j)
					count += 1
					if count >= 26:
						break
		return flist


	def generate_graphs(self):
		"""
		Generates a list of random graphs based on the specified constraints.

		Returns:
			List of generated graphs.
		"""
		import numpy as np
		import networkx as nx

		rng = np.random.default_rng(self.seed)

		graphs = []

		for _ in range(self.num_graphs):
			num_nodes = rng.integers(self.min_num_nodes, self.max_num_nodes + 1)
			num_edges = rng.integers(self.min_num_edges, self.max_num_edges + 1)

			G = nx.Graph()
			G.add_nodes_from(range(num_nodes))

			if num_edges > 0:
				edge_pairs = num_nodes * (num_nodes - 1) // 2
				if num_edges > edge_pairs:
					num_edges = edge_pairs
				# Generate random edges:
				edges = rng.choice(
					edge_pairs, num_edges, replace=False
				)
				for edge in edges:
					u = edge // (num_nodes - 1)
					v = edge % (num_nodes - 1)
					if v >= u:
						v += 1
					G.add_edge(u, v)

			# Add discrete node features:
			if self.with_discrete_n_features:
				if self.node_feature_values == {}:
					for node in G.nodes():
						G.nodes[node]['feature'] = rng.choice(
							self.discrete_n_features
						).item()

				else:
					pass
			# for node in G.nodes():
			# 	for feature in self.node_features:
			# 		G.nodes[node][feature] = random.choice(
			# 			self.node_feature_values.get(feature, [0])
			# 		)

			if self.with_discrete_e_features:
				if self.edge_feature_values == {}:
					for edge in G.edges():
						G.edges[edge]['feature'] = rng.choice(
							self.discrete_e_features
						).item()
				else:
					pass
			# for edge in G.edges():
			# 	for feature in self.edge_features:
			# 		G.edges[edge][feature] = random.choice(
			# 			self.edge_feature_values.get(feature, [0])
			# 		)

			if self.with_continuous_n_features:
				if self.node_feature_values == {}:
					for node in G.nodes():
						feature = rng.random(self.continuous_n_feature_dim)
						G.nodes[node][self.continuous_n_feature_key] = feature

				else:
					pass
			# for node in G.nodes():
			# 	for feature in self.node_features:
			# 		G.nodes[node][feature] = random.uniform(
			# 			self.node_feature_values.get(feature, (0, 1))[0],
			# 			self.node_feature_values.get(feature, (0, 1))[1]
			# 		)

			if self.with_continuous_e_features:
				if self.edge_feature_values == {}:
					for edge in G.edges():
						feature = rng.random(self.continuous_e_feature_dim)
						G.edges[edge][self.continuous_e_feature_key] = feature

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
