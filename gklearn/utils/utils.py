import networkx as nx
import numpy as np
from copy import deepcopy
from enum import Enum, unique
#from itertools import product

# from tqdm import tqdm


#%%


def getSPLengths(G1):
	sp = nx.shortest_path(G1)
	distances = np.zeros((G1.number_of_nodes(), G1.number_of_nodes()))
	for i in sp.keys():
		for j in sp[i].keys():
			distances[i, j] = len(sp[i][j]) - 1
	return distances


def get_sp_graph(G, edge_weight=None):
	"""Transform graph G to its corresponding shortest-paths graph.

	Parameters
	----------
	G : NetworkX graph
		The graph to be transformed.
	edge_weight : string
		edge attribute corresponding to the edge weight.

	Return
	------
	S : NetworkX graph
		The shortest-paths graph corresponding to G.

	Notes
	------
	For an input graph G, its corresponding shortest-paths graph S contains the same set of nodes as G, while there exists an edge between all nodes in S which are connected by a walk in G. Every edge in S between two nodes is labeled by the shortest distance between these two nodes.

	References
	----------
	.. [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
	"""
	return floydTransformation(G, edge_weight=edge_weight)


def getSPGraph(G, edge_weight=None):
	"""Transform graph G to its corresponding shortest-paths graph.

	Parameters
	----------
	G : NetworkX graph
		The graph to be transformed.
	edge_weight : string
		edge attribute corresponding to the edge weight.

	Return
	------
	S : NetworkX graph
		The shortest-paths graph corresponding to G.

	Notes
	------
	For an input graph G, its corresponding shortest-paths graph S contains the same set of nodes as G, while there exists an edge between all nodes in S which are connected by a walk in G. Every edge in S between two nodes is labeled by the shortest distance between these two nodes.

	References
	----------
	.. [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
	"""
	# Raise deprecated warning:
	import warnings
	warnings.warn("getSPGraph is deprecated, use get_sp_graph instead.", DeprecationWarning)
	return floydTransformation(G, edge_weight=edge_weight)


def floydTransformation(G, edge_weight=None):
	"""Transform graph G to its corresponding shortest-paths graph using Floyd-transformation.

	Parameters
	----------
	G : NetworkX graph
		The graph to be tramsformed.
	edge_weight : string
		edge attribute corresponding to the edge weight. The default edge weight is bond_type.

	Return
	------
	S : NetworkX graph
		The shortest-paths graph corresponding to G.

	References
	----------
	.. [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
	"""
	spMatrix = nx.floyd_warshall_numpy(G, weight=edge_weight)
	S = nx.Graph()
	S.add_nodes_from(G.nodes(data=True))
	ns = list(G.nodes())
	for i in range(0, G.number_of_nodes()):
		for j in range(i + 1, G.number_of_nodes()):
			if spMatrix[i, j] != np.inf:
				S.add_edge(ns[i], ns[j], cost=spMatrix[i, j])
	return S


def get_shortest_paths(G, weight, directed):
	"""Get all shortest paths of a graph.

	Parameters
	----------
	G : NetworkX graphs
		The graphs whose paths are calculated.
	weight : string/None
		edge attribute used as weight to calculate the shortest path.
	directed: boolean
		Whether graph is directed.

	Return
	------
	sp : list of list
		List of shortest paths of the graph, where each path is represented by a list of nodes.
	"""
	from itertools import combinations
	sp = []
	for n1, n2 in combinations(G.nodes(), 2):
		try:
			spltemp = list(nx.all_shortest_paths(G, n1, n2, weight=weight))
		except nx.NetworkXNoPath:  # nodes not connected
			pass
		else:
			sp += spltemp
			# each edge walk is counted twice, starting from both its extreme nodes.
			if not directed:
				sp += [sptemp[::-1] for sptemp in spltemp]

	# add single nodes as length 0 paths.
	sp += [[n] for n in G.nodes()]
	return sp


def untotterTransformation(G, node_label, edge_label):
	"""Transform graph G according to Mahé et al.'s work to filter out tottering patterns of marginalized kernel and tree pattern kernel.

	Parameters
	----------
	G : NetworkX graph
		The graph to be tramsformed.
	node_label : string
		node attribute used as label. The default node label is 'atom'.
	edge_label : string
		edge attribute used as label. The default edge label is 'bond_type'.

	Return
	------
	gt : NetworkX graph
		The transformed graph corresponding to G.

	References
	----------
	.. [1] Pierre Mahé, Nobuhisa Ueda, Tatsuya Akutsu, Jean-Luc Perret, and Jean-Philippe Vert. Extensions of marginalized graph kernels. In Proceedings of the twenty-first international conference on Machine learning, page 70. ACM, 2004.
	"""
	# arrange all graphs in a list
	G = G.to_directed()
	gt = nx.Graph()
	gt.graph = G.graph
	gt.add_nodes_from(G.nodes(data=True))
	for edge in G.edges():
		gt.add_node(edge)
		gt.nodes[edge].update({node_label: G.nodes[edge[1]][node_label]})
		gt.add_edge(edge[0], edge)
		gt.edges[edge[0], edge].update({
			edge_label:
			G[edge[0]][edge[1]][edge_label]
		})
		for neighbor in G[edge[1]]:
			if neighbor != edge[0]:
				gt.add_edge(edge, (edge[1], neighbor))
				gt.edges[edge, (edge[1], neighbor)].update({
					edge_label:
					G[edge[1]][neighbor][edge_label]
				})
	# nx.draw_networkx(gt)
	# plt.show()

	# relabel nodes using consecutive integers for convenience of kernel calculation.
	gt = nx.convert_node_labels_to_integers(
		gt, first_label=0, label_attribute='label_orignal')
	return gt


def direct_product(G1, G2, node_label, edge_label):
	"""Return the direct/tensor product of directed graphs G1 and G2.

	Parameters
	----------
	G1, G2 : NetworkX graph
		The original graphs.
	node_label : string
		node attribute used as label. The default node label is 'atom'.
	edge_label : string
		edge attribute used as label. The default edge label is 'bond_type'.

	Return
	------
	gt : NetworkX graph
		The direct product graph of G1 and G2.

	Notes
	-----
	This method differs from networkx.tensor_product in that this method only adds nodes and edges in G1 and G2 that have the same labels to the direct product graph.

	References
	----------
	.. [1] Thomas Gärtner, Peter Flach, and Stefan Wrobel. On graph kernels: Hardness results and efficient alternatives. Learning Theory and Kernel Machines, pages 129–143, 2003.
	"""
	# arrange all graphs in a list
	from itertools import product
	# G = G.to_directed()
	gt = nx.DiGraph()
	# add nodes
	for u, v in product(G1, G2):
		if G1.nodes[u][node_label] == G2.nodes[v][node_label]:
			gt.add_node((u, v))
			gt.nodes[(u, v)].update({node_label: G1.nodes[u][node_label]})
	# add edges, faster for sparse graphs (no so many edges), which is the most case for now.
	for (u1, v1), (u2, v2) in product(G1.edges, G2.edges):
		if (u1, u2) in gt and (
				v1, v2
		) in gt and G1.edges[u1, v1][edge_label] == G2.edges[u2,
															 v2][edge_label]:
			gt.add_edge((u1, u2), (v1, v2))
			gt.edges[(u1, u2), (v1, v2)].update({
				edge_label:
				G1.edges[u1, v1][edge_label]
			})

	# # add edges, faster for dense graphs (a lot of edges, complete graph would be super).
	# for u, v in product(gt, gt):
	#	 if (u[0], v[0]) in G1.edges and (
	#			 u[1], v[1]
	#	 ) in G2.edges and G1.edges[u[0],
	#								v[0]][edge_label] == G2.edges[u[1],
	#															  v[1]][edge_label]:
	#		 gt.add_edge((u[0], u[1]), (v[0], v[1]))
	#		 gt.edges[(u[0], u[1]), (v[0], v[1])].update({
	#			 edge_label:
	#			 G1.edges[u[0], v[0]][edge_label]
	#		 })

	# relabel nodes using consecutive integers for convenience of kernel calculation.
	# gt = nx.convert_node_labels_to_integers(
	#	 gt, first_label=0, label_attribute='label_orignal')
	return gt


def direct_product_graph(G1, G2, node_labels, edge_labels):
	"""Return the direct/tensor product of directed graphs G1 and G2.

	Parameters
	----------
	G1, G2 : NetworkX graph
		The original graphs.
	node_labels : list
		A list of node attributes used as labels.
	edge_labels : list
		A list of edge attributes used as labels.

	Return
	------
	gt : NetworkX graph
		The direct product graph of G1 and G2.

	Notes
	-----
	This method differs from networkx.tensor_product in that this method only adds nodes and edges in G1 and G2 that have the same labels to the direct product graph.

	References
	----------
	.. [1] Thomas Gärtner, Peter Flach, and Stefan Wrobel. On graph kernels: Hardness results and efficient alternatives. Learning Theory and Kernel Machines, pages 129–143, 2003.
	"""
	# arrange all graphs in a list
	from itertools import product
	# G = G.to_directed()
	gt = nx.DiGraph()
	# add nodes
	for u, v in product(G1, G2):
		label1 = tuple(G1.nodes[u][nl] for nl in node_labels)
		label2 = tuple(G2.nodes[v][nl] for nl in node_labels)
		if label1 == label2:
			gt.add_node((u, v), node_label=label1)

	# add edges, faster for sparse graphs (no so many edges), which is the most case for now.
	for (u1, v1), (u2, v2) in product(G1.edges, G2.edges):
		if (u1, u2) in gt and (v1, v2) in gt:
			label1 = tuple(G1.edges[u1, v1][el] for el in edge_labels)
			label2 = tuple(G2.edges[u2, v2][el] for el in edge_labels)
			if label1 == label2:
				gt.add_edge((u1, u2), (v1, v2), edge_label=label1)


	# # add edges, faster for dense graphs (a lot of edges, complete graph would be super).
	# for u, v in product(gt, gt):
	#	 if (u[0], v[0]) in G1.edges and (
	#			 u[1], v[1]
	#	 ) in G2.edges and G1.edges[u[0],
	#								v[0]][edge_label] == G2.edges[u[1],
	#															  v[1]][edge_label]:
	#		 gt.add_edge((u[0], u[1]), (v[0], v[1]))
	#		 gt.edges[(u[0], u[1]), (v[0], v[1])].update({
	#			 edge_label:
	#			 G1.edges[u[0], v[0]][edge_label]
	#		 })

	# relabel nodes using consecutive integers for convenience of kernel calculation.
	# gt = nx.convert_node_labels_to_integers(
	#	 gt, first_label=0, label_attribute='label_orignal')
	return gt


def find_paths(G, source_node, length):
	"""Find all paths with a certain length those start from a source node.
	A recursive depth first search is applied.

	Parameters
	----------
	G : NetworkX graphs
		The graph in which paths are searched.
	source_node : integer
		The number of the node from where all paths start.
	length : integer
		The length of paths.

	Return
	------
	path : list of list
		List of paths retrieved, where each path is represented by a list of nodes.
	"""
	if length == 0:
		return [[source_node]]
	path = [[source_node] + path for neighbor in G[source_node] \
		for path in find_paths(G, neighbor, length - 1) if source_node not in path]
	return path


def find_all_paths(G, length, is_directed):
	"""Find all paths with a certain length in a graph. A recursive depth first
	search is applied.

	Parameters
	----------
	G : NetworkX graphs
		The graph in which paths are searched.
	length : integer
		The length of paths.

	Return
	------
	path : list of list
		List of paths retrieved, where each path is represented by a list of nodes.
	"""
	all_paths = []
	for node in G:
		all_paths.extend(find_paths(G, node, length))

	if not is_directed:
		# For each path, two presentations are retrieved from its two extremities.
		# Remove one of them.
		all_paths_r = [path[::-1] for path in all_paths]
		for idx, path in enumerate(all_paths[:-1]):
			for path2 in all_paths_r[idx+1::]:
				if path == path2:
					all_paths[idx] = []
					break
		all_paths = list(filter(lambda a: a != [], all_paths))

	return all_paths


# @todo: use it in ShortestPath.
def compute_vertex_kernels(g1, g2, node_kernels, node_labels=[], node_attrs=[]):
	"""Compute kernels between each pair of vertices in two graphs.

	Parameters
	----------
	g1, g2 : NetworkX graph
		The kernels bewteen pairs of vertices in these two graphs are computed.
	node_kernels : dict
		A dictionary of kernel functions for nodes, including 3 items: 'symb'
		for symbolic node labels, 'nsymb' for non-symbolic node labels, 'mix'
		for both labels. The first 2 functions take two node labels as
		parameters, and the 'mix' function takes 4 parameters, a symbolic and a
		non-symbolic label for each the two nodes. Each label is in form of 2-D
		dimension array (n_samples, n_features). Each function returns a number
		as the kernel value. Ignored when nodes are unlabeled. This argument
		is designated to conjugate gradient method and fixed-point iterations.
	node_labels : list, optional
		The list of the name strings of the node labels. The default is [].
	node_attrs : list, optional
		The list of the name strings of the node attributes. The default is [].

	Returns
	-------
	vk_dict : dict
		Vertex kernels keyed by vertices.

	Notes
	-----
	This function is used by ``gklearn.kernels.FixedPoint'' and
	``gklearn.kernels.StructuralSP''. The method is borrowed from FCSP [1].

	References
	----------
	.. [1] Lifan Xu, Wei Wang, M Alvarez, John Cavazos, and Dongping Zhang.
	Parallelization of shortest path graph kernels on multi-core cpus and gpus.
	Proceedings of the Programmability Issues for Heterogeneous Multicores
	(MultiProg), Vienna, Austria, 2014.
	"""
	vk_dict = {}  # shortest path matrices dict
	if len(node_labels) > 0:
		# node symb and non-synb labeled
		if len(node_attrs) > 0:
			kn = node_kernels['mix']
			for n1 in g1.nodes(data=True):
				for n2 in g2.nodes(data=True):
					n1_labels = [n1[1][nl] for nl in node_labels]
					n2_labels = [n2[1][nl] for nl in node_labels]
					# @TODO: reformat attrs during data processing a priori to save time.
					n1_attrs = np.array([n1[1][na] for na in node_attrs]).astype(float)
					n2_attrs = np.array([n2[1][na] for na in node_attrs]).astype(float)
					vk_dict[(n1[0], n2[0])] = kn(n1_labels, n2_labels, n1_attrs, n2_attrs)
		# node symb labeled
		else:
			kn = node_kernels['symb']
			for n1 in g1.nodes(data=True):
				for n2 in g2.nodes(data=True):
					n1_labels = [n1[1][nl] for nl in node_labels]
					n2_labels = [n2[1][nl] for nl in node_labels]
					vk_dict[(n1[0], n2[0])] = kn(n1_labels, n2_labels)
	else:
		# node non-synb labeled
		if len(node_attrs) > 0:
			kn = node_kernels['nsymb']
			for n1 in g1.nodes(data=True):
				for n2 in g2.nodes(data=True):
					n1_attrs = np.array([n1[1][na] for na in node_attrs]).astype(float)
					n2_attrs = np.array([n2[1][na] for na in node_attrs]).astype(float)
					vk_dict[(n1[0], n2[0])] = kn(n1_attrs, n2_attrs)
		# node unlabeled
		else:
			pass  # @todo: add edge weights.
# 			for e1 in g1.edges(data=True):
# 				for e2 in g2.edges(data=True):
# 					if e1[2]['cost'] == e2[2]['cost']:
# 						kernel += 1
# 			return kernel

	return vk_dict


#%%


def get_graph_kernel_by_name(name, node_labels=None, edge_labels=None, node_attrs=None, edge_attrs=None, ds_infos=None, kernel_options={}, **kwargs):
	if len(kwargs) != 0:
		kernel_options = kwargs

	if name == 'CommonWalk' or name == 'common walk':
		from gklearn.kernels import CommonWalk
		graph_kernel = CommonWalk(node_labels=node_labels,
								 edge_labels=edge_labels,
								 ds_infos=ds_infos,
								 **kernel_options)

	elif name == 'Marginalized' or name == 'marginalized':
		from gklearn.kernels import Marginalized
		graph_kernel = Marginalized(node_labels=node_labels,
								 edge_labels=edge_labels,
								 ds_infos=ds_infos,
								 **kernel_options)

	elif name == 'SylvesterEquation' or name == 'sylvester equation':
		from gklearn.kernels import SylvesterEquation
		graph_kernel = SylvesterEquation(
								  ds_infos=ds_infos,
								  **kernel_options)

	elif name == 'FixedPoint' or name == 'fixed point':
		from gklearn.kernels import FixedPoint
		graph_kernel = FixedPoint(node_labels=node_labels,
								  edge_labels=edge_labels,
								  node_attrs=node_attrs,
								  edge_attrs=edge_attrs,
								  ds_infos=ds_infos,
								  **kernel_options)

	elif name == 'ConjugateGradient' or name == 'conjugate gradient':
		from gklearn.kernels import ConjugateGradient
		graph_kernel = ConjugateGradient(node_labels=node_labels,
								  edge_labels=edge_labels,
								  node_attrs=node_attrs,
								  edge_attrs=edge_attrs,
								  ds_infos=ds_infos,
								  **kernel_options)

	elif name == 'SpectralDecomposition' or name == 'spectral decomposition':
		from gklearn.kernels import SpectralDecomposition
		graph_kernel = SpectralDecomposition(node_labels=node_labels,
								  edge_labels=edge_labels,
								  node_attrs=node_attrs,
								  edge_attrs=edge_attrs,
								  ds_infos=ds_infos,
								  **kernel_options)

	elif name == 'ShortestPath' or name == 'shortest path':
		from gklearn.kernels import ShortestPath
		graph_kernel = ShortestPath(node_labels=node_labels,
								 node_attrs=node_attrs,
								 ds_infos=ds_infos,
								 **kernel_options)

	elif name == 'StructuralSP' or name == 'structural shortest path':
		from gklearn.kernels import StructuralSP
		graph_kernel = StructuralSP(node_labels=node_labels,
								  edge_labels=edge_labels,
								  node_attrs=node_attrs,
								  edge_attrs=edge_attrs,
								  ds_infos=ds_infos,
								  **kernel_options)

	elif name == 'PathUpToH' or name == 'path up to length h':
		from gklearn.kernels import PathUpToH
		graph_kernel = PathUpToH(node_labels=node_labels,
							  edge_labels=edge_labels,
							  ds_infos=ds_infos,
							  **kernel_options)

	elif name == 'Treelet' or name == 'treelet':
		from gklearn.kernels import Treelet
		graph_kernel = Treelet(node_labels=node_labels,
							  edge_labels=edge_labels,
							  ds_infos=ds_infos,
							  **kernel_options)

	elif name == 'WLSubtree' or name == 'weisfeiler-lehman subtree':
		from gklearn.kernels import WLSubtree
		graph_kernel = WLSubtree(node_labels=node_labels,
							  edge_labels=edge_labels,
							  ds_infos=ds_infos,
							  **kernel_options)

	elif name == 'WeisfeilerLehman' or name == 'weisfeiler-lehman':
		from gklearn.kernels import WeisfeilerLehman
		graph_kernel = WeisfeilerLehman(node_labels=node_labels,
							  edge_labels=edge_labels,
							  ds_infos=ds_infos,
							  **kernel_options)
	else:
		raise Exception('The graph kernel given is not defined. Possible choices include: "StructuralSP", "ShortestPath", "PathUpToH", "Treelet", "WLSubtree", "WeisfeilerLehman".')

	return graph_kernel


def compute_gram_matrices_by_class(ds_name, kernel_options, save_results=True, dir_save='', irrelevant_labels=None, edge_required=False):
	import os
	from gklearn.utils import Dataset, split_dataset_by_target

	# 1. get dataset.
	print('1. getting dataset...')
	dataset_all = Dataset()
	dataset_all.load_predefined_dataset(ds_name)
	dataset_all.trim_dataset(edge_required=edge_required)
	if not irrelevant_labels is None:
		dataset_all.remove_labels(**irrelevant_labels)
# 	dataset_all.cut_graphs(range(0, 10))
	datasets = split_dataset_by_target(dataset_all)

	gram_matrix_unnorm_list = []
	run_time_list = []

	print('start generating preimage for each class of target...')
	for idx, dataset in enumerate(datasets):
		target = dataset.targets[0]
		print('\ntarget =', target, '\n')

		# 2. initialize graph kernel.
		print('2. initializing graph kernel and setting parameters...')
		graph_kernel = get_graph_kernel_by_name(kernel_options['name'],
										  node_labels=dataset.node_labels,
										  edge_labels=dataset.edge_labels,
										  node_attrs=dataset.node_attrs,
										  edge_attrs=dataset.edge_attrs,
										  ds_infos=dataset.get_dataset_infos(keys=['directed']),
										  kernel_options=kernel_options)

		# 3. compute gram matrix.
		print('3. computing gram matrix...')
		gram_matrix, run_time = graph_kernel.compute(dataset.graphs, **kernel_options)
		gram_matrix_unnorm = graph_kernel.gram_matrix_unnorm

		gram_matrix_unnorm_list.append(gram_matrix_unnorm)
		run_time_list.append(run_time)

	# 4. save results.
	print()
	print('4. saving results...')
	if save_results:
		os.makedirs(dir_save, exist_ok=True)
		np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm_list=gram_matrix_unnorm_list, run_time_list=run_time_list)

	print('\ncomplete.')


def normalize_gram_matrix(gram_matrix):
	diag = gram_matrix.diagonal().copy()
	old_settings = np.seterr(invalid='raise') # Catch FloatingPointError: invalid value encountered in sqrt.
	for i in range(len(gram_matrix)):
		for j in range(i, len(gram_matrix)):
			try:
				gram_matrix[i][j] /= np.sqrt(diag[i] * diag[j])
			except:
# 				rollback()
				np.seterr(**old_settings)
				raise
			else:
				gram_matrix[j][i] = gram_matrix[i][j]
	np.seterr(**old_settings)
	return gram_matrix


def compute_distance_matrix(gram_matrix):
	dis_mat = np.empty((len(gram_matrix), len(gram_matrix)))
	for i in range(len(gram_matrix)):
		for j in range(i, len(gram_matrix)):
			dis = gram_matrix[i, i] + gram_matrix[j, j] - 2 * gram_matrix[i, j]
			if dis < 0:
				if dis > -1e-10:
					dis = 0
				else:
					raise ValueError('The distance is negative.')
			dis_mat[i, j] = np.sqrt(dis)
			dis_mat[j, i] = dis_mat[i, j]
	dis_max = np.max(np.max(dis_mat))
	dis_min = np.min(np.min(dis_mat[dis_mat != 0]))
	dis_mean = np.mean(np.mean(dis_mat))
	return dis_mat, dis_max, dis_min, dis_mean


#%%


def graph_deepcopy(G):
	"""Deep copy a graph, including deep copy of all nodes, edges and
	attributes of the graph, nodes and edges.

	Note
	----
	- It is the same as the NetworkX function graph.copy(), as far as I know.

	- This function only supports Networkx.Graph and Networkx.DiGraph.
	"""
	# add graph attributes.
	labels = {}
	for k, v in G.graph.items():
		labels[k] = deepcopy(v)
	if G.is_directed():
		G_copy = nx.DiGraph(**labels)
	else:
		G_copy = nx.Graph(**labels)

	# add nodes
	for nd, attrs in G.nodes(data=True):
		labels = {}
		for k, v in attrs.items():
			labels[k] = deepcopy(v)
		G_copy.add_node(nd, **labels)

	# add edges.
	for nd1, nd2, attrs in G.edges(data=True):
		labels = {}
		for k, v in attrs.items():
			labels[k] = deepcopy(v)
		G_copy.add_edge(nd1, nd2, **labels)

	return G_copy


def graph_isIdentical(G1, G2):
	"""Check if two graphs are identical, including: same nodes, edges, node
	labels/attributes, edge labels/attributes.

	Notes
	-----
	1. The type of graphs has to be the same.

	2. Global/Graph attributes are neglected as they may contain names for graphs.
	"""
	# check nodes.
	nlist1 = [n for n in G1.nodes(data=True)]
	nlist2 = [n for n in G2.nodes(data=True)]
	if not nlist1 == nlist2:
		return False
	# check edges.
	elist1 = [n for n in G1.edges(data=True)]
	elist2 = [n for n in G2.edges(data=True)]
	if not elist1 == elist2:
		return False
	# check graph attributes.

	return True


def get_node_labels(Gn, node_label):
	"""Get node labels of dataset Gn.
	"""
	nl = set()
	for G in Gn:
		nl = nl | set(nx.get_node_attributes(G, node_label).values())
	return nl


def get_edge_labels(Gn, edge_label):
	"""Get edge labels of dataset Gn.
	"""
	el = set()
	for G in Gn:
		el = el | set(nx.get_edge_attributes(G, edge_label).values())
	return el


def get_mlti_dim_node_attrs(G, attr_names):
	attributes = []
	for nd, attrs in G.nodes(data=True):
		attributes.append(tuple(attrs[aname] for aname in attr_names))
	return attributes


def get_mlti_dim_edge_attrs(G, attr_names):
	attributes = []
	for ed, attrs in G.edges(data=True):
		attributes.append(tuple(attrs[aname] for aname in attr_names))
	return attributes


def nx_permute_nodes(G, random_state=None):
	"""Permute node indices in a NetworkX graph.

	Parameters
	----------
	G : TYPE
		DESCRIPTION.
	random_state : TYPE, optional
		DESCRIPTION. The default is None.

	Returns
	-------
	G_new : TYPE
		DESCRIPTION.

	Notes
	-----
	- This function only supports Networkx.Graph and Networkx.DiGraph.
	"""
	# @todo: relabel node with integers? (in case something went wrong...)
	# Add graph attributes.
	labels = {}
	for k, v in G.graph.items():
		labels[k] = deepcopy(v)
	if G.is_directed():
		G_new = nx.DiGraph(**labels)
	else:
		G_new = nx.Graph(**labels)

	# Create a random mapping old node indices <-> new indices.
	nb_nodes = nx.number_of_nodes(G)
	indices_orig = range(nb_nodes)
	idx_mapping = np.random.RandomState(seed=random_state).permutation(indices_orig)

	# Add nodes.
	nodes_orig = list(G.nodes)
	for i_orig in range(nb_nodes):
		i_new = idx_mapping[i_orig]
		labels = {}
		for k, v in G.nodes[nodes_orig[i_new]].items():
			labels[k] = deepcopy(v)
		G_new.add_node(nodes_orig[i_new], **labels)

	# Add edges.
	for nd1, nd2, attrs in G.edges(data=True):
		labels = {}
		for k, v in attrs.items():
			labels[k] = deepcopy(v)
		G_new.add_edge(nd1, nd2, **labels)


# 	# create a random mapping old label -> new label
# 	node_mapping = dict(zip(G.nodes(), np.random.RandomState(seed=random_state).permutation(G.nodes())))
# 	# build a new graph
# 	G_new = nx.relabel_nodes(G, node_mapping)

	return G_new


#%%


def dummy_node():
	"""
	/*!
	 * @brief Returns a dummy node.
	 * @return ID of dummy node.
	 */
	"""
	return np.inf # @todo: in GEDLIB, this is the max - 1 rather than max, I don't know why.


def undefined_node():
	"""
	/*!
	 * @brief Returns an undefined node.
	 * @return ID of undefined node.
	 */

	"""
	return np.inf


def dummy_edge():
	"""
	/*!
	 * @brief Returns a dummy edge.
	 * @return ID of dummy edge.
	 */

	"""
	return np.inf


@unique
class SpecialLabel(Enum):
	"""can be used to define special labels.
	"""
	DUMMY = 1 # The dummy label.
	# DUMMY = auto # enum.auto does not exist in Python 3.5.


#%%


def check_json_serializable(
		obj,
		deep: bool = False,
) -> bool:
	"""Check if an object is JSON serializable.

	Parameters
	----------
	obj : object
		The object to be checked.

	deep : bool, optional
		Whether to check the object recursively when `obj` is iterable.
		The default is False.

	Returns
	-------
	bool
		True if the object is JSON serializable, False otherwise.
	"""
	import json
	try:
		json.dumps(obj)
	except TypeError:
		return False
	else:
		if deep and hasattr(obj, '__iter__'):
			for item in obj:
				if not check_json_serializable(item, deep=True):
					return False
		return True


def is_basic_python_type(
		obj,
		type_list: list = None,
		deep: bool = False,
) -> bool:
	"""Check if an object is a basic type in Python.

	Parameters
	----------
	obj : object
		The object to be checked.

	type_list : list, optional
		The list of basic types in Python. The default is None, which means
		the default basic types are used. The default basic types include
		`int`, `float`, `complex`, `str`, `bool`, `NoneType`, `list`,
		`tuple`, `dict`, `set`, `frozenset`, `range`, `slice`.

	deep : bool, optional
		Whether to check the object recursively when `obj` is iterable.
		The default is False.

	Returns
	-------
	bool
		True if the object is a basic type in Python, False otherwise.
	"""
	if type_list is None:
		type_list = [
			int, float, complex, str, bool, type(None), list, tuple, dict,
			set, frozenset, range, slice
		]
	if not hasattr(obj, '__iter__') or isinstance(obj, (str, bytes)):
		return type(obj) in type_list
	else:
		if deep:
			for item in obj:
				if not is_basic_python_type(item, type_list=type_list, deep=True):
					return False
			return True
		else:
			return False
