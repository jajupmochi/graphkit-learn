""" Utilities function to manage graph files
"""

def loadCT(filename):
    import networkx as nx
    """load data from .ct file.

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
    from os.path import basename
    content = open(filename).read().splitlines()
    g = nx.Graph(name=str(content[0]), filename=basename(filename)) # set name of the graph
    tmp = content[1].split(" ")
    if tmp[0] == '':
        nb_nodes = int(tmp[1]) # number of the nodes
        nb_edges = int(tmp[2]) # number of the edges
    else:
        nb_nodes = int(tmp[0])
        nb_edges = int(tmp[1])

    for i in range(0, nb_nodes):
        tmp = content[i + 2].split(" ")
        tmp = [x for x in tmp if x != '']
        g.add_node(i, label=tmp[3])

    for i in range(0, nb_edges):
        tmp = content[i + g.number_of_nodes() + 2]
        tmp = [tmp[i:i+3] for i in range(0, len(tmp), 3)]
        g.add_edge(int(tmp[0]) - 1, int(tmp[1]) - 1, label=int(tmp[3]))
    return g


def loadGXL(filename):
    from os.path import basename
    import networkx as nx
    import xml.etree.ElementTree as ET

    tree = ET.parse(filename)
    root = tree.getroot()
    index = 0
    g = nx.Graph(filename=basename(filename), name=root[0].attrib['id'])
    dic = {}
    for node in root.iter('node'):
        label = node.find('attr')[0].text # Take only one attribute
        dic[node.attrib['id']] = index
        g.add_node(index, id=node.attrib['id'], label=label)
        index += 1

    for edge in root.iter('edge'):
        label = edge.find('attr')[0].text
        g.add_edge(dic[edge.attrib['from']], dic[edge.attrib['to']], label=label)
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
            #hard fix : force to chem before taking into account genericity
            #           must be attr instead of label
            label = 'chem'
            cur_attr = ET.SubElement(current_node, 'attr', attrib={'name' : label})
            cur_value = ET.SubElement(cur_attr,graph.nodes[v][attr].__class__.__name__)
            cur_value.text = graph.nodes[v][attr]

    for v1 in graph:
        for v2 in graph[v1]:
            if(v1 < v2): #Non oriented graphs
                cur_edge = ET.SubElement(graph_node, 'edge', attrib={'from' : str(v1),
                                                                     'to' : str(v2)})
                for attr in graph[v1][v2].keys():
                    #hard fix : force to chem before taking into account genericity
                    #           must be attr instead of label
                    label = 'valence'
                    cur_attr = ET.SubElement(cur_edge, 'attr', attrib={'name' : label})
                    cur_value = ET.SubElement(cur_attr, graph[v1][v2][attr].__class__.__name__)
                    cur_value.text = str(graph[v1][v2][attr])

    tree = ET.ElementTree(root_node)
    tree.write(filename)


def loadDataset(filename):
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

    return data, y
