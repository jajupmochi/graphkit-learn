""" Utilities function to manage graph files
"""
# import warnings
# warnings.simplefilter('always', DeprecationWarning)
# warnings.warn('The functions in the module "gklearn.utils.graph_files" will be deprecated and removed since version 0.4.0. Use the corresponding functions in the module "gklearn.dataset" instead.', DeprecationWarning)

from os.path import dirname, splitext


def load_dataset(filename, filename_targets=None, gformat=None, **kwargs):
	"""Read graph data from filename and load them as NetworkX graphs.

	Parameters
	----------
	filename : string
		The name of the file from where the dataset is read.
	filename_y : string
		The name of file of the targets corresponding to graphs.
	extra_params : dict
		Extra parameters only designated to '.mat' format.

	Return
	------
	data : List of NetworkX graph.

	y : List

	Targets corresponding to graphs.

	Notes
	-----
	This function supports following graph dataset formats:

	'ds': load data from .ds file. See comments of function loadFromDS for a example.

	'cxl': load data from Graph eXchange Language file (.cxl file). See
	`here <http://www.gupro.de/GXL/Introduction/background.html>`__ for detail.

	'sdf': load data from structured data file (.sdf file). See
	`here <http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx>`__
	for details.

	'mat': Load graph data from a MATLAB (up to version 7.1) .mat file. See
	README in `downloadable file <http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/WL/>`__
	for details.

	'txt': Load graph data from a special .txt file. See
	`here <https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets>`__
	for details. Note here filename is the name of either .txt file in
	the dataset directory.
	"""
	import warnings
	warnings.simplefilter('always', DeprecationWarning)
	warnings.warn('The function "gklearn.utils.load_dataset" will be deprecated and removed since version 0.4.0. Use the class "gklearn.dataset.DataLoader" instead.', DeprecationWarning)

	extension = splitext(filename)[1][1:]
	if extension == "ds":
		data, y, label_names = load_from_ds(filename, filename_targets)
	elif extension == "cxl":
		dir_dataset = kwargs.get('dirname_dataset', None)
		data, y, label_names = load_from_xml(filename, dir_dataset)
	elif extension == 'xml':
		dir_dataset = kwargs.get('dirname_dataset', None)
		data, y, label_names = load_from_xml(filename, dir_dataset)
	elif extension == "mat":
		order = kwargs.get('order')
		data, y, label_names = load_mat(filename, order)
	elif extension == 'txt':
		data, y, label_names = load_tud(filename)

	return data, y, label_names


def save_dataset(Gn, y, gformat='gxl', group=None, filename='gfile', **kwargs):
	"""Save list of graphs.
	"""
	import warnings
	warnings.simplefilter('always', DeprecationWarning)
	warnings.warn('The function "gklearn.utils.save_dataset" will be deprecated and removed since version 0.4.0. Use the class "gklearn.dataset.DataSaver" instead.', DeprecationWarning)

	import os
	dirname_ds = os.path.dirname(filename)
	if dirname_ds != '':
		dirname_ds += '/'
		os.makedirs(dirname_ds, exist_ok=True)

	if 'graph_dir' in kwargs:
		graph_dir = kwargs['graph_dir'] + '/'
		os.makedirs(graph_dir, exist_ok=True)
		del kwargs['graph_dir']
	else:
		graph_dir = dirname_ds

	if group == 'xml' and gformat == 'gxl':
		with open(filename + '.xml', 'w') as fgroup:
			fgroup.write("<?xml version=\"1.0\"?>")
			fgroup.write("\n<!DOCTYPE GraphCollection SYSTEM \"http://www.inf.unibz.it/~blumenthal/dtd/GraphCollection.dtd\">")
			fgroup.write("\n<GraphCollection>")
			for idx, g in enumerate(Gn):
				fname_tmp = "graph" + str(idx) + ".gxl"
				save_gxl(g, graph_dir + fname_tmp, **kwargs)
				fgroup.write("\n\t<graph file=\"" + fname_tmp + "\" class=\"" + str(y[idx]) + "\"/>")
			fgroup.write("\n</GraphCollection>")
			fgroup.close()


def load_ct(filename): # @todo: this function is only tested on CTFile V2000; header not considered; only simple cases (atoms and bonds are considered.)
	"""load data from a Chemical Table (.ct) file.

	Notes
	------
	a typical example of data in .ct is like this:

	3 2  <- number of nodes and edges

	0.0000	0.0000	0.0000 C <- each line describes a node (x,y,z + label)

	0.0000	0.0000	0.0000 C

	0.0000	0.0000	0.0000 O

	1  3  1  1 <- each line describes an edge : to, from, bond type, bond stereo

	2  3  1  1

	Check `CTFile Formats file <https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=10&ved=2ahUKEwivhaSdjsTlAhVhx4UKHczHA8gQFjAJegQIARAC&url=https%3A%2F%2Fwww.daylight.com%2Fmeetings%2Fmug05%2FKappler%2Fctfile.pdf&usg=AOvVaw1cDNrrmMClkFPqodlF2inS>`__
	for detailed format discription.
	"""
	import networkx as nx
	from os.path import basename
	g = nx.Graph()
	with open(filename) as f:
		content = f.read().splitlines()
		g = nx.Graph(name=str(content[0]), filename=basename(filename))  # set name of the graph

		# read the counts line.
		tmp = content[1].split(' ')
		tmp = [x for x in tmp if x != '']
		nb_atoms = int(tmp[0].strip())  # number of atoms
		nb_bonds = int(tmp[1].strip())  # number of bonds
		count_line_tags = ['number_of_atoms', 'number_of_bonds', 'number_of_atom_lists', '', 'chiral_flag', 'number_of_stext_entries', '', '', '', '', 'number_of_properties', 'CT_version']
		i = 0
		while i < len(tmp):
			if count_line_tags[i] != '': # if not obsoleted
				g.graph[count_line_tags[i]] = tmp[i].strip()
			i += 1

		# read the atom block.
		atom_tags = ['x', 'y', 'z', 'atom_symbol', 'mass_difference', 'charge', 'atom_stereo_parity', 'hydrogen_count_plus_1', 'stereo_care_box', 'valence', 'h0_designator', '', '', 'atom_atom_mapping_number', 'inversion_retention_flag', 'exact_change_flag']
		for i in range(0, nb_atoms):
			tmp = content[i + 2].split(' ')
			tmp = [x for x in tmp if x != '']
			g.add_node(i)
			j = 0
			while j < len(tmp):
				if atom_tags[j] != '':
					g.nodes[i][atom_tags[j]] = tmp[j].strip()
				j += 1

		# read the bond block.
		bond_tags = ['first_atom_number', 'second_atom_number', 'bond_type', 'bond_stereo', '', 'bond_topology', 'reacting_center_status']
		for i in range(0, nb_bonds):
			tmp = content[i + g.number_of_nodes() + 2].split(' ')
			tmp = [x for x in tmp if x != '']
			n1, n2 = int(tmp[0].strip()) - 1, int(tmp[1].strip()) - 1
			g.add_edge(n1, n2)
			j = 2
			while j < len(tmp):
				if bond_tags[j] != '':
					g.edges[(n1, n2)][bond_tags[j]] = tmp[j].strip()
				j += 1

	# get label names.
	label_names = {'node_labels': [], 'edge_labels': [], 'node_attrs': [], 'edge_attrs': []}
	atom_symbolic = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, None, None, 1, 1, 1]
	for nd in g.nodes():
		for key in g.nodes[nd]:
			if atom_symbolic[atom_tags.index(key)] == 1:
				label_names['node_labels'].append(key)
			else:
				label_names['node_attrs'].append(key)
		break
	bond_symbolic = [None, None, 1, 1, None, 1, 1]
	for ed in g.edges():
		for key in g.edges[ed]:
			if bond_symbolic[bond_tags.index(key)] == 1:
				label_names['edge_labels'].append(key)
			else:
				label_names['edge_attrs'].append(key)
		break

	return g, label_names


def load_gxl(filename): # @todo: directed graphs.
	from os.path import basename
	import networkx as nx
	import xml.etree.ElementTree as ET

	tree = ET.parse(filename)
	root = tree.getroot()
	index = 0
	g = nx.Graph(filename=basename(filename), name=root[0].attrib['id'])
	dic = {}  # used to retrieve incident nodes of edges
	for node in root.iter('node'):
		dic[node.attrib['id']] = index
		labels = {}
		for attr in node.iter('attr'):
			labels[attr.attrib['name']] = attr[0].text
		g.add_node(index, **labels)
		index += 1

	for edge in root.iter('edge'):
		labels = {}
		for attr in edge.iter('attr'):
			labels[attr.attrib['name']] = attr[0].text
		g.add_edge(dic[edge.attrib['from']], dic[edge.attrib['to']], **labels)

	# get label names.
	label_names = {'node_labels': [], 'edge_labels': [], 'node_attrs': [], 'edge_attrs': []}
	for node in root.iter('node'):
		for attr in node.iter('attr'):
			if attr[0].tag == 'int': # @todo: this maybe wrong, and slow.
				label_names['node_labels'].append(attr.attrib['name'])
			else:
				label_names['node_attrs'].append(attr.attrib['name'])
		break
	for edge in root.iter('edge'):
		for attr in edge.iter('attr'):
			if attr[0].tag == 'int': # @todo: this maybe wrong, and slow.
				label_names['edge_labels'].append(attr.attrib['name'])
			else:
				label_names['edge_attrs'].append(attr.attrib['name'])
		break

	return g, label_names


def save_gxl(graph, filename, method='default', node_labels=[], edge_labels=[], node_attrs=[], edge_attrs=[]):
	if method == 'default':
		gxl_file = open(filename, 'w')
		gxl_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
		gxl_file.write("<!DOCTYPE gxl SYSTEM \"http://www.gupro.de/GXL/gxl-1.0.dtd\">\n")
		gxl_file.write("<gxl xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n")
		if 'name' in graph.graph:
			name = str(graph.graph['name'])
		else:
			name = 'dummy'
		gxl_file.write("<graph id=\"" + name + "\" edgeids=\"false\" edgemode=\"undirected\">\n")
		for v, attrs in graph.nodes(data=True):
			gxl_file.write("<node id=\"_" + str(v) + "\">")
			for l_name in node_labels:
				gxl_file.write("<attr name=\"" + l_name + "\"><int>" +
							   str(attrs[l_name]) + "</int></attr>")
			for a_name in node_attrs:
				gxl_file.write("<attr name=\"" + a_name + "\"><float>" +
							   str(attrs[a_name]) + "</float></attr>")
			gxl_file.write("</node>\n")
		for v1, v2, attrs in graph.edges(data=True):
			gxl_file.write("<edge from=\"_" + str(v1) + "\" to=\"_" + str(v2) + "\">")
			for l_name in edge_labels:
				gxl_file.write("<attr name=\"" + l_name + "\"><int>" +
							   str(attrs[l_name]) + "</int></attr>")
			for a_name in edge_attrs:
				gxl_file.write("<attr name=\"" + a_name + "\"><float>" +
							   str(attrs[a_name]) + "</float></attr>")
			gxl_file.write("</edge>\n")
		gxl_file.write("</graph>\n")
		gxl_file.write("</gxl>")
		gxl_file.close()
	elif method == 'benoit':
		import xml.etree.ElementTree as ET
		root_node = ET.Element('gxl')
		attr = dict()
		attr['id'] = str(graph.graph['name'])
		attr['edgeids'] = 'true'
		attr['edgemode'] = 'undirected'
		graph_node = ET.SubElement(root_node, 'graph', attrib=attr)

		for v in graph:
			current_node = ET.SubElement(graph_node, 'node', attrib={'id': str(v)})
			for attr in graph.nodes[v].keys():
				cur_attr = ET.SubElement(
					current_node, 'attr', attrib={'name': attr})
				cur_value = ET.SubElement(cur_attr,
										  graph.nodes[v][attr].__class__.__name__)
				cur_value.text = graph.nodes[v][attr]

		for v1 in graph:
			for v2 in graph[v1]:
				if (v1 < v2):  # Non oriented graphs
					cur_edge = ET.SubElement(
						graph_node,
						'edge',
						attrib={
							'from': str(v1),
							'to': str(v2)
						})
					for attr in graph[v1][v2].keys():
						cur_attr = ET.SubElement(
							cur_edge, 'attr', attrib={'name': attr})
						cur_value = ET.SubElement(
							cur_attr, graph[v1][v2][attr].__class__.__name__)
						cur_value.text = str(graph[v1][v2][attr])

		tree = ET.ElementTree(root_node)
		tree.write(filename)
	elif method == 'gedlib':
		# reference: https://github.com/dbblumenthal/gedlib/blob/master/data/generate_molecules.py#L22
#		pass
		gxl_file = open(filename, 'w')
		gxl_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
		gxl_file.write("<!DOCTYPE gxl SYSTEM \"http://www.gupro.de/GXL/gxl-1.0.dtd\">\n")
		gxl_file.write("<gxl xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n")
		gxl_file.write("<graph id=\"" + str(graph.graph['name']) + "\" edgeids=\"true\" edgemode=\"undirected\">\n")
		for v, attrs in graph.nodes(data=True):
			gxl_file.write("<node id=\"_" + str(v) + "\">")
			gxl_file.write("<attr name=\"" + "chem" + "\"><int>" + str(attrs['chem']) + "</int></attr>")
			gxl_file.write("</node>\n")
		for v1, v2, attrs in graph.edges(data=True):
			gxl_file.write("<edge from=\"_" + str(v1) + "\" to=\"_" + str(v2) + "\">")
			gxl_file.write("<attr name=\"valence\"><int>" + str(attrs['valence']) + "</int></attr>")
#			gxl_file.write("<attr name=\"valence\"><int>" + "1" + "</int></attr>")
			gxl_file.write("</edge>\n")
		gxl_file.write("</graph>\n")
		gxl_file.write("</gxl>")
		gxl_file.close()
	elif method == 'gedlib-letter':
		# reference: https://github.com/dbblumenthal/gedlib/blob/master/data/generate_molecules.py#L22
		# and https://github.com/dbblumenthal/gedlib/blob/master/data/datasets/Letter/HIGH/AP1_0000.gxl
		gxl_file = open(filename, 'w')
		gxl_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
		gxl_file.write("<!DOCTYPE gxl SYSTEM \"http://www.gupro.de/GXL/gxl-1.0.dtd\">\n")
		gxl_file.write("<gxl xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n")
		gxl_file.write("<graph id=\"" + str(graph.graph['name']) + "\" edgeids=\"false\" edgemode=\"undirected\">\n")
		for v, attrs in graph.nodes(data=True):
			gxl_file.write("<node id=\"_" + str(v) + "\">")
			gxl_file.write("<attr name=\"x\"><float>" + str(attrs['attributes'][0]) + "</float></attr>")
			gxl_file.write("<attr name=\"y\"><float>" + str(attrs['attributes'][1]) + "</float></attr>")
			gxl_file.write("</node>\n")
		for v1, v2, attrs in graph.edges(data=True):
			gxl_file.write("<edge from=\"_" + str(v1) + "\" to=\"_" + str(v2) + "\"/>\n")
		gxl_file.write("</graph>\n")
		gxl_file.write("</gxl>")
		gxl_file.close()


# def loadSDF(filename):
# 	"""load data from structured data file (.sdf file).

# 	Notes
# 	------
# 	A SDF file contains a group of molecules, represented in the similar way as in MOL format.
# 	Check `here <http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx>`__ for detailed structure.
# 	"""
# 	import networkx as nx
# 	from os.path import basename
# 	from tqdm import tqdm
# 	import sys
# 	data = []
# 	with open(filename) as f:
# 		content = f.read().splitlines()
# 		index = 0
# 		pbar = tqdm(total=len(content) + 1, desc='load SDF', file=sys.stdout)
# 		while index < len(content):
# 			index_old = index

# 			g = nx.Graph(name=content[index].strip())  # set name of the graph

# 			tmp = content[index + 3]
# 			nb_nodes = int(tmp[:3])  # number of the nodes
# 			nb_edges = int(tmp[3:6])  # number of the edges

# 			for i in range(0, nb_nodes):
# 				tmp = content[i + index + 4]
# 				g.add_node(i, atom=tmp[31:34].strip())

# 			for i in range(0, nb_edges):
# 				tmp = content[i + index + g.number_of_nodes() + 4]
# 				tmp = [tmp[i:i + 3] for i in range(0, len(tmp), 3)]
# 				g.add_edge(
# 					int(tmp[0]) - 1, int(tmp[1]) - 1, bond_type=tmp[2].strip())

# 			data.append(g)

# 			index += 4 + g.number_of_nodes() + g.number_of_edges()
# 			while content[index].strip() != '$$$$':  # seperator
# 				index += 1
# 			index += 1

# 			pbar.update(index - index_old)
# 		pbar.update(1)
# 		pbar.close()

# 	return data


def load_mat(filename, order): # @todo: need to be updated (auto order) or deprecated.
	"""Load graph data from a MATLAB (up to version 7.1) .mat file.

	Notes
	------
	A MAT file contains a struct array containing graphs, and a column vector lx containing a class label for each graph.
	Check README in `downloadable file <http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/WL/>`__ for detailed structure.
	"""
	from scipy.io import loadmat
	import numpy as np
	import networkx as nx
	data = []
	content = loadmat(filename)
	# print(content)
	# print('----')
	for key, value in content.items():
		if key[0] == 'l':  # class label
			y = np.transpose(value)[0].tolist()
			# print(y)
		elif key[0] != '_':
			# print(value[0][0][0])
			# print()
			# print(value[0][0][1])
			# print()
			# print(value[0][0][2])
			# print()
			# if len(value[0][0]) > 3:
			#	 print(value[0][0][3])
			# print('----')
			# if adjacency matrix is not compressed / edge label exists
			if order[1] == 0:
				for i, item in enumerate(value[0]):
					# print(item)
					# print('------')
					g = nx.Graph(name=i)  # set name of the graph
					nl = np.transpose(item[order[3]][0][0][0])  # node label
					# print(item[order[3]])
					# print()
					for index, label in enumerate(nl[0]):
						g.add_node(index, label_1=str(label))
					el = item[order[4]][0][0][0]  # edge label
					for edge in el:
						g.add_edge(edge[0] - 1, edge[1] - 1, label_1=str(edge[2]))
					data.append(g)
			else:
# 				from scipy.sparse import csc_matrix
				for i, item in enumerate(value[0]):
					# print(item)
					# print('------')
					g = nx.Graph(name=i)  # set name of the graph
					nl = np.transpose(item[order[3]][0][0][0])  # node label
					# print(nl)
					# print()
					for index, label in enumerate(nl[0]):
						g.add_node(index, label_1=str(label))
					sam = item[order[0]]  # sparse adjacency matrix
					index_no0 = sam.nonzero()
					for col, row in zip(index_no0[0], index_no0[1]):
						# print(col)
						# print(row)
						g.add_edge(col, row)
					data.append(g)
					# print(g.edges(data=True))

	label_names = {'node_labels': ['label_1'], 'edge_labels': [], 'node_attrs': [], 'edge_attrs': []}
	if order[1] == 0:
		label_names['edge_labels'].append('label_1')

	return data, y, label_names


def load_tud(filename):
	"""Load graph data from TUD dataset files.

	Notes
	------
	The graph data is loaded from separate files.
	Check README in `downloadable file <http://tiny.cc/PK_MLJ_data>`__, 2018 for detailed structure.
	"""
	import networkx as nx
	from os import listdir
	from os.path import dirname, basename


	def get_infos_from_readme(frm): # @todo: add README (cuniform), maybe node/edge label maps.
		"""Get information from DS_label_readme.txt file.
		"""

		def get_label_names_from_line(line):
			"""Get names of labels/attributes from a line.
			"""
			str_names = line.split('[')[1].split(']')[0]
			names = str_names.split(',')
			names = [attr.strip() for attr in names]
			return names


		def get_class_label_map(label_map_strings):
			label_map = {}
			for string in label_map_strings:
				integer, label = string.split('\t')
				label_map[int(integer.strip())] = label.strip()
			return label_map


		label_names = {'node_labels': [], 'node_attrs': [],
					   'edge_labels': [], 'edge_attrs': []}
		class_label_map = None
		class_label_map_strings = []
		with open(frm) as rm:
			content_rm = rm.read().splitlines()
		i = 0
		while i < len(content_rm):
			line = content_rm[i].strip()
			# get node/edge labels and attributes.
			if line.startswith('Node labels:'):
				label_names['node_labels'] = get_label_names_from_line(line)
			elif line.startswith('Node attributes:'):
				label_names['node_attrs'] = get_label_names_from_line(line)
			elif line.startswith('Edge labels:'):
				label_names['edge_labels'] = get_label_names_from_line(line)
			elif line.startswith('Edge attributes:'):
				label_names['edge_attrs'] = get_label_names_from_line(line)
			# get class label map.
			elif line.startswith('Class labels were converted to integer values using this map:'):
				i += 2
				line = content_rm[i].strip()
				while line != '' and i < len(content_rm):
					class_label_map_strings.append(line)
					i += 1
					line = content_rm[i].strip()
				class_label_map = get_class_label_map(class_label_map_strings)
			i += 1

		return label_names, class_label_map


	# get dataset name.
	dirname_dataset = dirname(filename)
	filename = basename(filename)
	fn_split = filename.split('_A')
	ds_name = fn_split[0].strip()

	# load data file names
	for name in listdir(dirname_dataset):
		if ds_name + '_A' in name:
			fam = dirname_dataset + '/' + name
		elif ds_name + '_graph_indicator' in name:
			fgi = dirname_dataset + '/' + name
		elif ds_name + '_graph_labels' in name:
			fgl = dirname_dataset + '/' + name
		elif ds_name + '_node_labels' in name:
			fnl = dirname_dataset + '/' + name
		elif ds_name + '_edge_labels' in name:
			fel = dirname_dataset + '/' + name
		elif ds_name + '_edge_attributes' in name:
			fea = dirname_dataset + '/' + name
		elif ds_name + '_node_attributes' in name:
			fna = dirname_dataset + '/' + name
		elif ds_name + '_graph_attributes' in name:
			fga = dirname_dataset + '/' + name
		elif ds_name + '_label_readme' in name:
			frm = dirname_dataset + '/' + name
		# this is supposed to be the node attrs, make sure to put this as the last 'elif'
		elif ds_name + '_attributes' in name:
			fna = dirname_dataset + '/' + name

	# get labels and attributes names.
	if 'frm' in locals():
		label_names, class_label_map = get_infos_from_readme(frm)
	else:
		label_names = {'node_labels': [], 'node_attrs': [],
					   'edge_labels': [], 'edge_attrs': []}
		class_label_map = None

	with open(fgi) as gi:
		content_gi = gi.read().splitlines()  # graph indicator
	with open(fam) as am:
		content_am = am.read().splitlines()  # adjacency matrix

	# load targets.
	if 'fgl' in locals():
		with open(fgl) as gl:
			content_targets = gl.read().splitlines()  # targets (classification)
		targets = [float(i) for i in content_targets]
	elif 'fga' in locals():
		with open(fga) as ga:
			content_targets = ga.read().splitlines()  # targets (regression)
		targets = [int(i) for i in content_targets]
	else:
		raise Exception('Can not find targets file. Please make sure there is a "', ds_name, '_graph_labels.txt" or "', ds_name, '_graph_attributes.txt"', 'file in your dataset folder.')
	if class_label_map is not None:
		targets = [class_label_map[t] for t in targets]

	# create graphs and add nodes
	data = [nx.Graph(name=str(i)) for i in range(0, len(content_targets))]
	if 'fnl' in locals():
		with open(fnl) as nl:
			content_nl = nl.read().splitlines()  # node labels
		for idx, line in enumerate(content_gi):
			# transfer to int first in case of unexpected blanks
			data[int(line) - 1].add_node(idx)
			labels = [l.strip() for l in content_nl[idx].split(',')]
			if label_names['node_labels'] == []: # @todo: need fix bug.
				for i, label in enumerate(labels):
					l_name = 'label_' + str(i)
					data[int(line) - 1].nodes[idx][l_name] = label
					label_names['node_labels'].append(l_name)
			else:
				for i, l_name in enumerate(label_names['node_labels']):
					data[int(line) - 1].nodes[idx][l_name] = labels[i]
	else:
		for i, line in enumerate(content_gi):
			data[int(line) - 1].add_node(i)

	# add edges
	for line in content_am:
		tmp = line.split(',')
		n1 = int(tmp[0]) - 1
		n2 = int(tmp[1]) - 1
		# ignore edge weight here.
		g = int(content_gi[n1]) - 1
		data[g].add_edge(n1, n2)

	# add edge labels
	if 'fel' in locals():
		with open(fel) as el:
			content_el = el.read().splitlines()
		for idx, line in enumerate(content_el):
			labels = [l.strip() for l in line.split(',')]
			n = [int(i) - 1 for i in content_am[idx].split(',')]
			g = int(content_gi[n[0]]) - 1
			if label_names['edge_labels'] == []:
				for i, label in enumerate(labels):
					l_name = 'label_' + str(i)
					data[g].edges[n[0], n[1]][l_name] = label
					label_names['edge_labels'].append(l_name)
			else:
				for i, l_name in enumerate(label_names['edge_labels']):
					data[g].edges[n[0], n[1]][l_name] = labels[i]

	# add node attributes
	if 'fna' in locals():
		with open(fna) as na:
			content_na = na.read().splitlines()
		for idx, line in enumerate(content_na):
			attrs = [a.strip() for a in line.split(',')]
			g = int(content_gi[idx]) - 1
			if label_names['node_attrs'] == []:
				for i, attr in enumerate(attrs):
					a_name = 'attr_' + str(i)
					data[g].nodes[idx][a_name] = attr
					label_names['node_attrs'].append(a_name)
			else:
				for i, a_name in enumerate(label_names['node_attrs']):
					data[g].nodes[idx][a_name] = attrs[i]

	# add edge attributes
	if 'fea' in locals():
		with open(fea) as ea:
			content_ea = ea.read().splitlines()
		for idx, line in enumerate(content_ea):
			attrs = [a.strip() for a in line.split(',')]
			n = [int(i) - 1 for i in content_am[idx].split(',')]
			g = int(content_gi[n[0]]) - 1
			if label_names['edge_attrs'] == []:
				for i, attr in enumerate(attrs):
					a_name = 'attr_' + str(i)
					data[g].edges[n[0], n[1]][a_name] = attr
					label_names['edge_attrs'].append(a_name)
			else:
				for i, a_name in enumerate(label_names['edge_attrs']):
					data[g].edges[n[0], n[1]][a_name] = attrs[i]

	return data, targets, label_names


def load_from_ds(filename, filename_targets):
	"""Load data from .ds file.

	Possible graph formats include:

	'.ct': see function load_ct for detail.

	'.gxl': see dunction load_gxl for detail.

	Note these graph formats are checked automatically by the extensions of
	graph files.
	"""
	dirname_dataset = dirname(filename)
	data = []
	y = []
	label_names = {'node_labels': [], 'edge_labels': [], 'node_attrs': [], 'edge_attrs': []}
	with open(filename) as fn:
		content = fn.read().splitlines()
	extension = splitext(content[0].split(' ')[0])[1][1:]
	if extension == 'ct':
		load_file_fun = load_ct
	elif extension == 'gxl' or extension == 'sdf': # @todo: .sdf not tested yet.
		load_file_fun = load_gxl

	if filename_targets is None or filename_targets == '':
		for i in range(0, len(content)):
			tmp = content[i].split(' ')
			# remove the '#'s in file names
			g, l_names = load_file_fun(dirname_dataset + '/' + tmp[0].replace('#', '', 1))
			data.append(g)
			_append_label_names(label_names, l_names)
			y.append(float(tmp[1]))
	else:  # targets in a seperate file
		for i in range(0, len(content)):
			tmp = content[i]
			# remove the '#'s in file names
			g, l_names = load_file_fun(dirname_dataset + '/' + tmp.replace('#', '', 1))
			data.append(g)
			_append_label_names(label_names, l_names)

		with open(filename_targets) as fnt:
			content_y = fnt.read().splitlines()
		# assume entries in filename and filename_targets have the same order.
		for item in content_y:
			tmp = item.split(' ')
			# assume the 3rd entry in a line is y (for Alkane dataset)
			y.append(float(tmp[2]))

	return data, y, label_names


# def load_from_cxl(filename):
# 	import xml.etree.ElementTree as ET
#
# 	dirname_dataset = dirname(filename)
# 	tree = ET.parse(filename)
# 	root = tree.getroot()
# 	data = []
# 	y = []
# 	for graph in root.iter('graph'):
# 		mol_filename = graph.attrib['file']
# 		mol_class = graph.attrib['class']
# 		data.append(load_gxl(dirname_dataset + '/' + mol_filename))
# 		y.append(mol_class)


def load_from_xml(filename, dir_dataset=None):
	import xml.etree.ElementTree as ET

	if dir_dataset is not None:
		dir_dataset = dir_dataset
	else:
		dir_dataset = dirname(filename)
	tree = ET.parse(filename)
	root = tree.getroot()
	data = []
	y = []
	label_names = {'node_labels': [], 'edge_labels': [], 'node_attrs': [], 'edge_attrs': []}
	for graph in root.iter('graph'):
		mol_filename = graph.attrib['file']
		mol_class = graph.attrib['class']
		g, l_names = load_gxl(dir_dataset + '/' + mol_filename)
		data.append(g)
		_append_label_names(label_names, l_names)
		y.append(mol_class)

	return data, y, label_names


def _append_label_names(label_names, new_names):
	for key, val in label_names.items():
		label_names[key] += [name for name in new_names[key] if name not in val]


if __name__ == '__main__':
#	### Load dataset from .ds file.
#	# .ct files.
#	ds = {'name': 'Alkane', 'dataset': '../../datasets/Alkane/dataset.ds',
#		'dataset_y': '../../datasets/Alkane/dataset_boiling_point_names.txt'}
#	Gn, y = loadDataset(ds['dataset'], filename_y=ds['dataset_y'])
# 	ds_file = '../../datasets/Acyclic/dataset_bps.ds'  # node symb
# 	Gn, targets, label_names = load_dataset(ds_file)
# 	ds_file = '../../datasets/MAO/dataset.ds' # node/edge symb
# 	Gn, targets, label_names = load_dataset(ds_file)
##	ds = {'name': 'PAH', 'dataset': '../../datasets/PAH/dataset.ds'} # unlabeled
##	Gn, y = loadDataset(ds['dataset'])
# 	print(Gn[1].graph)
# 	print(Gn[1].nodes(data=True))
# 	print(Gn[1].edges(data=True))
# 	print(targets[1])

# 	# .gxl file.
# 	ds_file = '../../datasets/monoterpenoides/dataset_10+.ds'  # node/edge symb
# 	Gn, y, label_names = load_dataset(ds_file)
# 	print(Gn[1].graph)
# 	print(Gn[1].nodes(data=True))
# 	print(Gn[1].edges(data=True))
# 	print(y[1])

	# .mat file.
	ds_file = '../../datasets/MUTAG_mat/MUTAG.mat'
	order = [0, 0, 3, 1, 2]
	Gn, targets, label_names = load_dataset(ds_file, order=order)
	print(Gn[1].graph)
	print(Gn[1].nodes(data=True))
	print(Gn[1].edges(data=True))
	print(targets[1])

#	### Convert graph from one format to another.
#	# .gxl file.
#	import networkx as nx
#	ds = {'name': 'monoterpenoides',
#		  'dataset': '../../datasets/monoterpenoides/dataset_10+.ds'}  # node/edge symb
#	Gn, y = loadDataset(ds['dataset'])
#	y = [int(i) for i in y]
#	print(Gn[1].nodes(data=True))
#	print(Gn[1].edges(data=True))
#	print(y[1])
#	# Convert a graph to the proper NetworkX format that can be recognized by library gedlib.
#	Gn_new = []
#	for G in Gn:
#		G_new = nx.Graph()
#		for nd, attrs in G.nodes(data=True):
#			G_new.add_node(str(nd), chem=attrs['atom'])
#		for nd1, nd2, attrs in G.edges(data=True):
#			G_new.add_edge(str(nd1), str(nd2), valence=attrs['bond_type'])
##			G_new.add_edge(str(nd1), str(nd2))
#		Gn_new.append(G_new)
#	print(Gn_new[1].nodes(data=True))
#	print(Gn_new[1].edges(data=True))
#	print(Gn_new[1])
#	filename = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/generated_datsets/monoterpenoides/gxl/monoterpenoides'
#	xparams = {'method': 'gedlib'}
#	saveDataset(Gn, y, gformat='gxl', group='xml', filename=filename, xparams=xparams)

	# save dataset.
#	ds = {'name': 'MUTAG', 'dataset': '../../datasets/MUTAG/MUTAG.mat',
#		  'extra_params': {'am_sp_al_nl_el': [0, 0, 3, 1, 2]}}  # node/edge symb
#	Gn, y = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
#	saveDataset(Gn, y, group='xml', filename='temp/temp')

	# test - new way to add labels and attributes.
#	dataset = '../../datasets/SYNTHETICnew/SYNTHETICnew_A.txt'
# 	filename = '../../datasets/Fingerprint/Fingerprint_A.txt'
#	dataset = '../../datasets/Letter-med/Letter-med_A.txt'
#	dataset = '../../datasets/AIDS/AIDS_A.txt'
#	dataset = '../../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'
# 	Gn, targets, label_names = load_dataset(filename)
	pass
