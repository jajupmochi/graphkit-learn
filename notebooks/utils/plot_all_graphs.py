#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:25:36 2020

@author: ljia
"""

# draw all the praphs
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from gklearn.utils.graphfiles import loadDataset, loadGXL


def main(): 
    # MUTAG dataset.
    dataset, y = loadDataset("../../datasets/MUTAG/MUTAG_A.txt")
    for idx in [6]: #[65]:#
        G = dataset[idx]
        ncolors= []
        for node in G.nodes:
            if G.nodes[node]['atom'] == '0':
                G.nodes[node]['atom'] = 'C'
                ncolors.append('#bd3182')
            elif G.nodes[node]['atom'] == '1':
                G.nodes[node]['atom'] = 'N'
                ncolors.append('#3182bd')
            elif G.nodes[node]['atom'] == '2':
                G.nodes[node]['atom'] = 'O'
                ncolors.append('#82bd31')
            elif G.nodes[node]['atom'] == '3':
                G.nodes[node]['atom'] = 'F'
            elif G.nodes[node]['atom'] == '4':
                G.nodes[node]['atom'] = 'I'
            elif G.nodes[node]['atom'] == '5':
                G.nodes[node]['atom'] = 'Cl'
            elif G.nodes[node]['atom'] == '6':
                G.nodes[node]['atom'] = 'Br'
        ecolors = []
        for edge in G.edges:
            if G.edges[edge]['bond_type'] == '0':
                ecolors.append('#bd3182')
            elif G.edges[edge]['bond_type'] == '1':
                ecolors.append('#3182bd')
            elif G.edges[edge]['bond_type'] == '2':
                ecolors.append('#82bd31')
            elif G.edges[edge]['bond_type'] == '3':
                ecolors.append('orange')

        print(idx)
        print(nx.get_node_attributes(G, 'atom'))
        edge_labels = nx.get_edge_attributes(G, 'bond_type')
        print(edge_labels)
        pos=nx.spring_layout(G)
        nx.draw(G, 
                pos,
                node_size=500,
                labels=nx.get_node_attributes(G, 'atom'), 
                node_color=ncolors, 
                font_color='w', 
                edge_color=ecolors,
                width=3,
                with_labels=True)
#        edge_labels = nx.draw_networkx_edge_labels(G, pos, 
#                                                   edge_labels=edge_labels,
#                                                   font_color='pink')
        plt.savefig('mol1_graph.svg', format='svg', dpi=300)
        plt.show()
        plt.clf()
    
    
#    # monoterpenoides dataset.
#    dataset, y = loadDataset("../../datasets/monoterpenoides/dataset_10+.ds")
#    for idx in [12,22,29,74]:
#        print(idx)
#        print(nx.get_node_attributes(dataset[idx], 'atom'))
#        edge_labels = nx.get_edge_attributes(dataset[idx], 'bond_type')
#        print(edge_labels)
#        pos=nx.spring_layout(dataset[idx])
#        nx.draw(dataset[idx], pos, labels=nx.get_node_attributes(dataset[idx], 'atom'), with_labels=True)
#        edge_labels = nx.draw_networkx_edge_labels(dataset[idx], pos, 
#                                                   edge_labels=edge_labels,
#                                                   font_color='pink')
#        plt.show()
    
    
#    # Fingerprint dataset.
#    dataset = '/media/ljia/DATA/research-repo/codes/Linlin/gedlib/data/collections/Fingerprint.xml'
#    graph_dir = '/media/ljia/DATA/research-repo/codes/Linlin/gedlib/data/datasets/Fingerprint/data/' 
#    Gn, y_all = loadDataset(dataset, extra_params=graph_dir)
##    dataset = '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/datasets/Fingerprint/Fingerprint_A.txt'
##    Gn, y_all = loadDataset(dataset)
#    
#    idx_no_node = []
#    idx_no_edge = []
#    idx_no_both = []
#    for idx, G in enumerate(Gn):
#        if nx.number_of_nodes(G) == 0:
#            idx_no_node.append(idx)
#            if nx.number_of_edges(G) == 0:
#                idx_no_both.append(idx)
#        if nx.number_of_edges(G) == 0:
#            idx_no_edge.append(idx)
##        file_prefix = '../results/graph_images/Fingerprint/' + G.graph['name']
##        draw_Fingerprint_graph(Gn[idx], file_prefix=file_prefix, save=True)
#    print('nb_no_node: ', len(idx_no_node))
#    print('nb_no_edge: ', len(idx_no_edge))
#    print('nb_no_both: ', len(idx_no_both))
#    print('idx_no_node: ', idx_no_node)
#    print('idx_no_edge: ', idx_no_edge)
#    print('idx_no_both: ', idx_no_both)
#    
#    for idx in [0, 10, 100, 1000]:
#        print(idx)
#        print(Gn[idx].nodes(data=True))
#        print(Gn[idx].edges(data=True))
#        draw_Fingerprint_graph(Gn[idx], file_prefix='')
    
    
#    # SYNTHETIC dataset.
#    dataset = '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/datasets/SYNTHETICnew/SYNTHETICnew_A.txt'
#    Gn, y_all = loadDataset(dataset)
#    
#    idx_no_node = []
#    idx_no_edge = []
#    idx_no_both = []
#    for idx, G in enumerate(Gn):
#        if nx.number_of_nodes(G) == 0:
#            idx_no_node.append(idx)
#            if nx.number_of_edges(G) == 0:
#                idx_no_both.append(idx)
#        if nx.number_of_edges(G) == 0:
#            idx_no_edge.append(idx)
##        file_prefix = '../results/graph_images/SYNTHETIC/' + G.graph['name']
##        draw_SYNTHETIC_graph(Gn[idx], file_prefix=file_prefix, save=True)
##        draw_SYNTHETIC_graph(Gn[idx])
#    print('nb_no_node: ', len(idx_no_node))
#    print('nb_no_edge: ', len(idx_no_edge))
#    print('nb_no_both: ', len(idx_no_both))
#    print('idx_no_node: ', idx_no_node)
#    print('idx_no_edge: ', idx_no_edge)
#    print('idx_no_both: ', idx_no_both)
#    
#    for idx in [0, 10, 100]:
#        print(idx)
#        print(Gn[idx].nodes(data=True))
#        print(Gn[idx].edges(data=True))
#        draw_SYNTHETIC_graph(Gn[idx], save=None)
        
        
def plot_a_graph(graph_filename):
    graph = loadGXL(graph_filename)
    print(nx.get_node_attributes(graph, 'atom'))
    edge_labels = nx.get_edge_attributes(graph, 'bond_type')
    print(edge_labels)
    pos=nx.spring_layout(graph)
    nx.draw(graph, pos, labels=nx.get_node_attributes(graph, 'atom'), with_labels=True)
    edge_labels = nx.draw_networkx_edge_labels(graph, pos, 
                                               edge_labels=edge_labels,
                                               font_color='pink')
    plt.show()
    
    
#Dessin median courrant
def draw_Fingerprint_graph(graph, file_prefix=None, save=None):
    plt.figure()
    pos = {}
    for n in graph.nodes:
        pos[n] = np.array([float(graph.node[n]['x']), float(graph.node[n]['y'])])
    # set plot settings.
    max_x = np.max([p[0] for p in pos.values()]) if len(pos) > 0 else 10
    min_x = np.min([p[0] for p in pos.values()]) if len(pos) > 0 else 10
    max_y = np.max([p[1] for p in pos.values()]) if len(pos) > 0 else 10
    min_y = np.min([p[1] for p in pos.values()]) if len(pos) > 0 else 10
    padding_x = (max_x - min_x + 10) * 0.1
    padding_y = (max_y - min_y + 10) * 0.1
    range_x = max_x + padding_x - (min_x - padding_x)
    range_y = max_y + padding_y - (min_y - padding_y)
    if range_x > range_y:
        plt.xlim(min_x - padding_x, max_x + padding_x)
        plt.ylim(min_y - padding_y - (range_x - range_y) / 2, 
                 max_y + padding_y + (range_x - range_y) / 2)
    else:
        plt.xlim(min_x - padding_x - (range_y - range_x) / 2, 
                 max_x + padding_x + (range_y - range_x) / 2)
        plt.ylim(min_y - padding_y, max_y + padding_y)
    plt.gca().set_aspect('equal', adjustable='box')
    nx.draw_networkx(graph, pos)
    if save is not None:
        plt.savefig(file_prefix + '.eps', format='eps', dpi=300)
    else:
        plt.show()
    plt.clf()
    
    
def draw_SYNTHETIC_graph(graph, file_prefix=None, save=None):
    plt.figure()
    nx.draw_networkx(graph)
    if save is not None:
        plt.savefig(file_prefix + '.eps', format='eps', dpi=300)
    else:
        plt.show()
    plt.clf()
    
    
if __name__ == '__main__':
    main()
#    gfn = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/output/tmp_ged/set_median.gxl'
#    plot_a_graph(gfn)
#    gfn = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/output/tmp_ged/gen_median.gxl'
#    plot_a_graph(gfn)