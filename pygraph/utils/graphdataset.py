""" Obtain all kinds of attributes of a graph dataset.
"""


def get_dataset_attributes(Gn,
                           target=None,
                           attr_names=[],
                           node_label=None,
                           edge_label=None):
    import networkx as nx
    import numpy as np

    attrs = {}

    def get_dataset_size(Gn):
        return len(Gn)

    def get_all_graph_size(Gn):
        return [nx.number_of_nodes(G) for G in Gn]

    def get_ave_graph_size(all_graph_size):
        return np.mean(all_graph_size)

    def get_min_graph_size(all_graph_size):
        return np.amin(all_graph_size)

    def get_max_graph_size(Gn):
        return np.amax(all_graph_size)

    def get_all_graph_edge_num(Gn):
        return [nx.number_of_edges(G) for G in Gn]

    def get_ave_graph_edge_num(all_graph_edge_num):
        return np.mean(all_graph_edge_num)

    def get_min_graph_edge_num(all_graph_edge_num):
        return np.amin(all_graph_edge_num)

    def get_max_graph_edge_num(all_graph_edge_num):
        return np.amax(all_graph_edge_num)

    def is_node_labeled(Gn):
        return False if node_label is None else True

    def get_node_label_num(Gn):
        nl = set()
        for G in Gn:
            nl = nl | set(nx.get_node_attributes(G, node_label).values())
        return len(nl)

    def is_edge_labeled(Gn):
        return False if edge_label is None else True

    def get_edge_label_num(Gn):
        nl = set()
        for G in Gn:
            nl = nl | set(nx.get_edge_attributes(G, edge_label).values())
        return len(nl)

    def is_directed(Gn):
        return nx.is_directed(Gn[0])

    def get_ave_graph_degree(Gn):
        return np.mean([np.amax(list(dict(G.degree()).values())) for G in Gn])

    def get_max_graph_degree(Gn):
        return np.amax([np.amax(list(dict(G.degree()).values())) for G in Gn])

    def get_min_graph_degree(Gn):
        return np.amin([np.amax(list(dict(G.degree()).values())) for G in Gn])

    def get_substructures(Gn):
        subs = set()
        for G in Gn:
            degrees = list(dict(G.degree()).values())
            if any(i == 2 for i in degrees):
                subs.add('linear')
            if np.amax(degrees) >= 3:
                subs.add('non linear')
            if 'linear' in subs and 'non linear' in subs:
                break

        if is_directed(Gn):
            for G in Gn:
                if len(list(nx.find_cycle(G))) > 0:
                    subs.add('cyclic')
                    break
        # else:
        #     # @todo: this method does not work for big graph with large amount of edges like D&D, try a better way.
        #     upper = np.amin([nx.number_of_edges(G) for G in Gn]) * 2 + 10
        #     for G in Gn:
        #         if (nx.number_of_edges(G) < upper):
        #             cyc = list(nx.simple_cycles(G.to_directed()))
        #             if any(len(i) > 2 for i in cyc):
        #                 subs.add('cyclic')
        #                 break
        #     if 'cyclic' not in subs:
        #         for G in Gn:
        #             cyc = list(nx.simple_cycles(G.to_directed()))
        #             if any(len(i) > 2 for i in cyc):
        #                 subs.add('cyclic')
        #                 break

        return subs

    def get_class_num(target):
        return len(set(target))

    def get_node_attr_dim(Gn):
        attrs = Gn[0].nodes[0]
        if 'attributes' in attrs:
            return len(attrs['attributes'])
        else:
            return False

    def get_edge_attr_dim(Gn):
        for G in Gn:
            if nx.number_of_edges(G) > 0:
                for e in G.edges(data=True):
                    if 'attributes' in e[2]:
                        return len(e[2]['attributes'])
                    else:
                        return False
        return False

    if attr_names == []:
        attr_names = [
            'substructures',
            'node_labeled',
            'edge_labeled',
            'is_directed',
            'dataset_size',
            'ave_graph_size',
            'min_graph_size',
            'max_graph_size',
            'ave_graph_edge_num',
            'min_graph_edge_num',
            'max_graph_edge_num',
            'ave_graph_degree',
            'min_graph_degree',
            'max_graph_degree',
            'node_label_num',
            'edge_label_num',
            'node_attr_dim',
            'edge_attr_dim',
            'class_number',
        ]

    # dataset size
    if 'dataset_size' in attr_names:

        attrs.update({'dataset_size': get_dataset_size(Gn)})

    # graph size
    if any(i in attr_names
           for i in ['ave_graph_size', 'min_graph_size', 'max_graph_size']):

        all_graph_size = get_all_graph_size(Gn)

    if 'ave_graph_size' in attr_names:

        attrs.update({'ave_graph_size': get_ave_graph_size(all_graph_size)})

    if 'min_graph_size' in attr_names:

        attrs.update({'min_graph_size': get_min_graph_size(all_graph_size)})

    if 'max_graph_size' in attr_names:

        attrs.update({'max_graph_size': get_max_graph_size(all_graph_size)})

    # graph edge number
    if any(i in attr_names for i in
           ['ave_graph_edge_num', 'min_graph_edge_num', 'max_graph_edge_num']):

        all_graph_edge_num = get_all_graph_edge_num(Gn)

    if 'ave_graph_edge_num' in attr_names:

        attrs.update({
            'ave_graph_edge_num':
            get_ave_graph_edge_num(all_graph_edge_num)
        })

    if 'max_graph_edge_num' in attr_names:

        attrs.update({
            'max_graph_edge_num':
            get_max_graph_edge_num(all_graph_edge_num)
        })

    if 'min_graph_edge_num' in attr_names:

        attrs.update({
            'min_graph_edge_num':
            get_min_graph_edge_num(all_graph_edge_num)
        })

    # label number
    if any(i in attr_names for i in ['node_labeled', 'node_label_num']):
        is_nl = is_node_labeled(Gn)
        node_label_num = get_node_label_num(Gn)

    if 'node_labeled' in attr_names:
        # graphs are considered node unlabeled if all nodes have the same label.
        attrs.update({'node_labeled': is_nl if node_label_num > 1 else False})

    if 'node_label_num' in attr_names:
        attrs.update({'node_label_num': node_label_num})

    if any(i in attr_names for i in ['edge_labeled', 'edge_label_num']):
        is_el = is_edge_labeled(Gn)
        edge_label_num = get_edge_label_num(Gn)

    if 'edge_labeled' in attr_names:
        # graphs are considered edge unlabeled if all edges have the same label.
        attrs.update({'edge_labeled': is_el if edge_label_num > 1 else False})

    if 'edge_label_num' in attr_names:
        attrs.update({'edge_label_num': edge_label_num})

    if 'is_directed' in attr_names:
        attrs.update({'is_directed': is_directed(Gn)})

    if 'ave_graph_degree' in attr_names:
        attrs.update({'ave_graph_degree': get_ave_graph_degree(Gn)})

    if 'max_graph_degree' in attr_names:
        attrs.update({'max_graph_degree': get_max_graph_degree(Gn)})

    if 'min_graph_degree' in attr_names:
        attrs.update({'min_graph_degree': get_min_graph_degree(Gn)})

    if 'substructures' in attr_names:
        attrs.update({'substructures': get_substructures(Gn)})

    if 'class_number' in attr_names:
        attrs.update({'class_number': get_class_num(target)})

    if 'node_attr_dim' in attr_names:
        attrs['node_attr_dim'] = get_node_attr_dim(Gn)

    if 'edge_attr_dim' in attr_names:
        attrs['edge_attr_dim'] = get_edge_attr_dim(Gn)

    from collections import OrderedDict
    return OrderedDict(
        sorted(attrs.items(), key=lambda i: attr_names.index(i[0])))
