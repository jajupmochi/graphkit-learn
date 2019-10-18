#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:44:59 2019

@author: ljia
"""
import numpy as np
import networkx as nx
from tqdm import tqdm
import sys

from gedlibpy import librariesImport, gedlibpy

def GED(g1, g2, lib='gedlibpy', cost='CHEM_1', method='IPFP', 
        edit_cost_constant=[], saveGXL='benoit', stabilizer='min', repeat=50):
    """
    Compute GED for 2 graphs.
    """
    if lib == 'gedlibpy':
        def convertGraph(G):
            """Convert a graph to the proper NetworkX format that can be
            recognized by library gedlibpy.
            """
            G_new = nx.Graph()
            for nd, attrs in G.nodes(data=True):
                G_new.add_node(str(nd), chem=attrs['atom'])
            for nd1, nd2, attrs in G.edges(data=True):
#                G_new.add_edge(str(nd1), str(nd2), valence=attrs['bond_type'])
                G_new.add_edge(str(nd1), str(nd2))
                
            return G_new
        
        gedlibpy.restart_env()
        gedlibpy.add_nx_graph(convertGraph(g1), "")
        gedlibpy.add_nx_graph(convertGraph(g2), "")

        listID = gedlibpy.get_all_graph_ids()
        gedlibpy.set_edit_cost(cost, edit_cost_constant=edit_cost_constant)
        gedlibpy.init()
        gedlibpy.set_method(method, "")
        gedlibpy.init_method()

        g = listID[0]
        h = listID[1]
        if stabilizer == None:
            gedlibpy.run_method(g, h)
            pi_forward = gedlibpy.get_forward_map(g, h)
            pi_backward = gedlibpy.get_backward_map(g, h)
            upper = gedlibpy.get_upper_bound(g, h)
            lower = gedlibpy.get_lower_bound(g, h)        
        elif stabilizer == 'min':
            upper = np.inf
            for itr in range(repeat):                
                gedlibpy.run_method(g, h)                
                upper_tmp = gedlibpy.get_upper_bound(g, h)                
                if upper_tmp < upper:
                    upper = upper_tmp
                    pi_forward = gedlibpy.get_forward_map(g, h)
                    pi_backward = gedlibpy.get_backward_map(g, h)
                    lower = gedlibpy.get_lower_bound(g, h)
                if upper == 0:
                    break
                    
        dis = upper
        
        # make the map label correct (label remove map as np.inf)
        nodes1 = [n for n in g1.nodes()]
        nodes2 = [n for n in g2.nodes()]
        nb1 = nx.number_of_nodes(g1)
        nb2 = nx.number_of_nodes(g2)
        pi_forward = [nodes2[pi] if pi < nb2 else np.inf for pi in pi_forward]
        pi_backward = [nodes1[pi] if pi < nb1 else np.inf for pi in pi_backward]      
        
    return dis, pi_forward, pi_backward


def GED_n(Gn, lib='gedlibpy', cost='CHEM_1', method='IPFP', 
        edit_cost_constant=[], stabilizer='min', repeat=50):
    """
    Compute GEDs for a group of graphs.
    """
    if lib == 'gedlibpy':
        def convertGraph(G):
            """Convert a graph to the proper NetworkX format that can be
            recognized by library gedlibpy.
            """
            G_new = nx.Graph()
            for nd, attrs in G.nodes(data=True):
                G_new.add_node(str(nd), chem=attrs['atom'])
            for nd1, nd2, attrs in G.edges(data=True):
#                G_new.add_edge(str(nd1), str(nd2), valence=attrs['bond_type'])
                G_new.add_edge(str(nd1), str(nd2))
                
            return G_new
        
        gedlibpy.restart_env()
        gedlibpy.add_nx_graph(convertGraph(g1), "")
        gedlibpy.add_nx_graph(convertGraph(g2), "")

        listID = gedlibpy.get_all_graph_ids()
        gedlibpy.set_edit_cost(cost, edit_cost_constant=edit_cost_constant)
        gedlibpy.init()
        gedlibpy.set_method(method, "")
        gedlibpy.init_method()

        g = listID[0]
        h = listID[1]
        if stabilizer == None:
            gedlibpy.run_method(g, h)
            pi_forward = gedlibpy.get_forward_map(g, h)
            pi_backward = gedlibpy.get_backward_map(g, h)
            upper = gedlibpy.get_upper_bound(g, h)
            lower = gedlibpy.get_lower_bound(g, h)        
        elif stabilizer == 'min':
            upper = np.inf
            for itr in range(repeat):                
                gedlibpy.run_method(g, h)                
                upper_tmp = gedlibpy.get_upper_bound(g, h)                
                if upper_tmp < upper:
                    upper = upper_tmp
                    pi_forward = gedlibpy.get_forward_map(g, h)
                    pi_backward = gedlibpy.get_backward_map(g, h)
                    lower = gedlibpy.get_lower_bound(g, h)
                if upper == 0:
                    break
                    
        dis = upper
        
        # make the map label correct (label remove map as np.inf)
        nodes1 = [n for n in g1.nodes()]
        nodes2 = [n for n in g2.nodes()]
        nb1 = nx.number_of_nodes(g1)
        nb2 = nx.number_of_nodes(g2)
        pi_forward = [nodes2[pi] if pi < nb2 else np.inf for pi in pi_forward]
        pi_backward = [nodes1[pi] if pi < nb1 else np.inf for pi in pi_backward]      
        
    return dis, pi_forward, pi_backward


def ged_median(Gn, Gn_median, measure='ged', verbose=False, 
                    ged_cost='CHEM_1', ged_method='IPFP', saveGXL='benoit'):
    dis_list = []
    pi_forward_list = []
    for idx, G in tqdm(enumerate(Gn), desc='computing median distances', 
                       file=sys.stdout) if verbose else enumerate(Gn):
        dis_sum = 0
        pi_forward_list.append([])
        for G_p in Gn_median:
            dis_tmp, pi_tmp_forward, pi_tmp_backward = GED(G, G_p, 
                cost=ged_cost, method=ged_method, saveGXL=saveGXL)
            pi_forward_list[idx].append(pi_tmp_forward)
            dis_sum += dis_tmp
        dis_list.append(dis_sum)
    return dis_list, pi_forward_list


def get_nb_edit_operations(g1, g2, forward_map, backward_map):
    """Compute the number of each edit operations.
    """
    n_vi = 0
    n_vr = 0
    n_vs = 0
    n_ei = 0
    n_er = 0
    n_es = 0
    
    nodes1 = [n for n in g1.nodes()]
    for i, map_i in enumerate(forward_map):
        if map_i == np.inf:
            n_vr += 1
        elif g1.node[nodes1[i]]['atom'] != g2.node[map_i]['atom']:
            n_vs += 1
    for map_i in backward_map:
        if map_i == np.inf:
            n_vi += 1
    
#    idx_nodes1 = range(0, len(node1))
    
    edges1 = [e for e in g1.edges()]
    nb_edges2_cnted = 0
    for n1, n2 in edges1:
        idx1 = nodes1.index(n1)
        idx2 = nodes1.index(n2)
        # one of the nodes is removed, thus the edge is removed.
        if forward_map[idx1] == np.inf or forward_map[idx2] == np.inf:
            n_er += 1
        # corresponding edge is in g2. Edge label is not considered.
        elif (forward_map[idx1], forward_map[idx2]) in g2.edges() or \
            (forward_map[idx2], forward_map[idx1]) in g2.edges():
                nb_edges2_cnted += 1
        # corresponding nodes are in g2, however the edge is removed.
        else:
            n_er += 1
    n_ei = nx.number_of_edges(g2) - nb_edges2_cnted
    
    return n_vi, n_vr, n_vs, n_ei, n_er, n_es