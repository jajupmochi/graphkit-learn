""" Utilities function to manage graph files
"""

def loadCT(filename):
    """load data from .ct file.
nn
    Notes
    ------
    a typical example of data in .ct is like this:

     3 2  <- number of nodes and edges
        0.0000    0.0000    0.0000 C <- each line describes a node (x,y,z + label)
        0.0000    0.0000    0.0000 C
        0.0000    0.0000    0.0000 O
      1  3  1  1 <- each line describes an edge : to, from,?, label
      2  3  1  1
    """
    import networkx as nx
    from os.path import basename
    g = nx.Graph()
    with open(filename) as f:
        content = f.read().splitlines()
        g = nx.Graph(name=str(content[0]), filename=basename(filename)) # set name of the graph
        tmp = content[1].split(" ")
        if tmp[0] == '':
            nb_nodes = int(tmp[1]) # number of the nodes
            nb_edges = int(tmp[2]) # number of the edges
        else:
            nb_nodes = int(tmp[0])
            nb_edges = int(tmp[1])
            # patch for compatibility : label will be removed later
        for i in range(0, nb_nodes):
            tmp = content[i + 2].split(" ")
            tmp = [x for x in tmp if x != '']
            g.add_node(i, atom=tmp[3], label=tmp[3])
        for i in range(0, nb_edges):
            tmp = content[i + g.number_of_nodes() + 2].split(" ")
            tmp = [x for x in tmp if x != '']
            g.add_edge(int(tmp[0]) - 1, int(tmp[1]) - 1,
                         bond_type=tmp[3].strip(), label=tmp[3].strip())

#         for i in range(0, nb_edges):
#             tmp = content[i + g.number_of_nodes() + 2]
#             tmp = [tmp[i:i+3] for i in range(0, len(tmp), 3)]
#             g.add_edge(int(tmp[0]) - 1, int(tmp[1]) - 1,
#                        bond_type=tmp[3].strip(), label=tmp[3].strip())
    return g


def loadGXL(filename):
    from os.path import basename
    import networkx as nx
    import xml.etree.ElementTree as ET

    tree = ET.parse(filename)
    root = tree.getroot()
    index = 0
    g = nx.Graph(filename=basename(filename), name=root[0].attrib['id'])
    dic = {} #used to retrieve incident nodes of edges
    for node in root.iter('node'):
        dic[node.attrib['id']] = index
        labels = {}
        for attr in node.iter('attr'):
            labels[attr.attrib['name']] = attr[0].text
        if 'chem' in labels:
            labels['label'] = labels['chem']
        g.add_node(index, **labels)
        index += 1

    for edge in root.iter('edge'):
        labels = {}
        for attr in edge.iter('attr'):
            labels[attr.attrib['name']] = attr[0].text
        if 'valence' in labels:
           labels['label'] = labels['valence']
        g.add_edge(dic[edge.attrib['from']], dic[edge.attrib['to']], **labels)
    return g

def saveGXL(graph, filename):
    import xml.etree.ElementTree as ET
    root_node = ET.Element('gxl')
    attr = dict()
    attr['id'] = graph.graph['name']
    attr['edgeids'] = 'true'
    attr['edgemode'] = 'undirected'
    graph_node = ET.SubElement(root_node, 'graph', attrib=attr)

    for v in graph:
        current_node = ET.SubElement(graph_node, 'node', attrib={'id' : str(v)})
        for attr in graph.nodes[v].keys():
            cur_attr = ET.SubElement(current_node, 'attr', attrib={'name' : attr})
            cur_value = ET.SubElement(cur_attr,graph.nodes[v][attr].__class__.__name__)
            cur_value.text = graph.nodes[v][attr]

    for v1 in graph:
        for v2 in graph[v1]:
            if(v1 < v2): #Non oriented graphs
                cur_edge = ET.SubElement(graph_node, 'edge', attrib={'from' : str(v1),
                                                                     'to' : str(v2)})
                for attr in graph[v1][v2].keys():
                    cur_attr = ET.SubElement(cur_edge, 'attr', attrib={'name' : attr})
                    cur_value = ET.SubElement(cur_attr, graph[v1][v2][attr].__class__.__name__)
                    cur_value.text = str(graph[v1][v2][attr])

    tree = ET.ElementTree(root_node)
    tree.write(filename)


def loadSDF(filename):
    """load data from structured data file (.sdf file).

    Notes
    ------
    A SDF file contains a group of molecules, represented in the similar way as in MOL format.
    see http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx, 2018 for detailed structure.
    """
    import networkx as nx
    from os.path import basename
    from tqdm import tqdm
    import sys
    data = []
    with open(filename) as f:
        content = f.read().splitlines()
        index = 0
        pbar = tqdm(total = len(content) + 1, desc = 'load SDF', file=sys.stdout)
        while index < len(content):
            index_old = index

            g = nx.Graph(name=content[index].strip()) # set name of the graph

            tmp = content[index + 3]
            nb_nodes = int(tmp[:3]) # number of the nodes
            nb_edges = int(tmp[3:6]) # number of the edges

            for i in range(0, nb_nodes):
                tmp = content[i + index + 4]
                g.add_node(i, atom=tmp[31:34].strip())

            for i in range(0, nb_edges):
                tmp = content[i + index + g.number_of_nodes() + 4]
                tmp = [tmp[i:i+3] for i in range(0, len(tmp), 3)]
                g.add_edge(int(tmp[0]) - 1, int(tmp[1]) - 1, bond_type=tmp[2].strip())

            data.append(g)

            index += 4 + g.number_of_nodes() + g.number_of_edges()
            while content[index].strip() != '$$$$': # seperator
                index += 1
            index += 1

            pbar.update(index - index_old)
        pbar.update(1)
        pbar.close()

    return data



def loadDataset(filename, filename_y = ''):
    """load file list of the dataset.
    """
    from os.path import dirname, splitext

    dirname_dataset = dirname(filename)
    extension = splitext(filename)[1][1:]
    data = []
    y = []
    if extension == "ds":
        content = open(filename).read().splitlines()
        for i in range(0, len(content)):
            tmp = content[i].split(' ')
            data.append(loadCT(dirname_dataset + '/' + tmp[0].replace('#', '', 1))) # remove the '#'s in file names
            y.append(float(tmp[1]))
    elif(extension == "cxl"):
        import xml.etree.ElementTree as ET

        tree = ET.parse(filename)
        root = tree.getroot()
        data = []
        y = []
        for graph in root.iter('print'):
            mol_filename = graph.attrib['file']
            mol_class = graph.attrib['class']
            data.append(loadGXL(dirname_dataset + '/' + mol_filename))
            y.append(mol_class)
    elif extension == "sdf":
        import numpy as np
        from tqdm import tqdm
        import sys

        data = loadSDF(filename)

        y_raw = open(filename_y).read().splitlines()
        y_raw.pop(0)
        tmp0 = []
        tmp1 = []
        for i in range(0, len(y_raw)):
            tmp = y_raw[i].split(',')
            tmp0.append(tmp[0])
            tmp1.append(tmp[1].strip())

        y = []
        for i in tqdm(range(0, len(data)), desc = 'ajust data', file=sys.stdout):
            try:
                y.append(tmp1[tmp0.index(data[i].name)].strip())
            except ValueError: # if data[i].name not in tmp0
                data[i] = []
        data = list(filter(lambda a: a != [], data))

    return data, y
