#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:49:12 2019

Iterative alternate minimizations using GED.
@author: ljia
"""
import numpy as np
import random
import networkx as nx

import sys
#from Cython_GedLib_2 import librariesImport, script
import librariesImport, script
sys.path.insert(0, "../")
from pygraph.utils.graphfiles import saveDataset
from pygraph.utils.graphdataset import get_dataset_attributes


def iam(Gn, node_label='atom', edge_label='bond_type'):
    """See my name, then you know what I do.
    """
#    Gn = Gn[0:10]
    Gn = [nx.convert_node_labels_to_integers(g) for g in Gn]
    
    c_er = 1
    c_es = 1
    c_ei = 1
    
    # phase 1: initilize.
    # compute set-median.
    dis_min = np.inf
    pi_p = []
    pi_all = []
    for idx1, G_p in enumerate(Gn):
        dist_sum = 0
        pi_all.append([])
        for idx2, G_p_prime in enumerate(Gn):
            dist_tmp, pi_tmp = GED(G_p, G_p_prime)
            pi_all[idx1].append(pi_tmp)
            dist_sum += dist_tmp
        if dist_sum < dis_min:
            dis_min = dist_sum
            G = G_p.copy()
            idx_min = idx1
    # list of edit operations.        
    pi_p = pi_all[idx_min]
            
    # phase 2: iteration.
    ds_attrs = get_dataset_attributes(Gn, attr_names=['edge_labeled', 'node_attr_dim'], 
                                      edge_label=edge_label)
    for itr in range(0, 10):
        G_new = G.copy()
        # update vertex labels.
        # pre-compute h_i0 for each label.
#        for label in get_node_labels(Gn, node_label):
#            print(label)
#        for nd in G.nodes(data=True):
#            pass
        if not ds_attrs['node_attr_dim']: # labels are symbolic
            for nd, _ in G.nodes(data=True):
                h_i0_list = []
                label_list = []
                for label in get_node_labels(Gn, node_label):
                    h_i0 = 0
                    for idx, g in enumerate(Gn):
                        pi_i = pi_p[idx][nd]
                        if g.has_node(pi_i) and g.nodes[pi_i][node_label] == label:
                            h_i0 += 1
                    h_i0_list.append(h_i0)
                    label_list.append(label)
                # choose one of the best randomly.
                idx_max = np.argwhere(h_i0_list == np.max(h_i0_list)).flatten().tolist()
                idx_rdm = random.randint(0, len(idx_max) - 1)
                G_new.nodes[nd][node_label] = label_list[idx_max[idx_rdm]]
        else: # labels are non-symbolic
            for nd, _ in G.nodes(data=True):
                Si_norm = 0
                phi_i_bar = np.array([0.0 for _ in range(ds_attrs['node_attr_dim'])])
                for idx, g in enumerate(Gn):
                    pi_i = pi_p[idx][nd]
                    if g.has_node(pi_i): #@todo: what if no g has node? phi_i_bar = 0?
                        Si_norm += 1
                        phi_i_bar += np.array([float(itm) for itm in g.nodes[pi_i]['attributes']])                
                phi_i_bar /= Si_norm
                G_new.nodes[nd]['attributes'] = phi_i_bar
                                            
        # update edge labels and adjacency matrix.
        if ds_attrs['edge_labeled']:
            for nd1, nd2, _ in G.edges(data=True):
                h_ij0_list = []
                label_list = []
                for label in get_edge_labels(Gn, edge_label):
                    h_ij0 = 0
                    for idx, g in enumerate(Gn):
                        pi_i = pi_p[idx][nd1]
                        pi_j = pi_p[idx][nd2]
                        h_ij0_p = (g.has_node(pi_i) and g.has_node(pi_j) and 
                                   g.has_edge(pi_i, pi_j) and 
                                   g.edges[pi_i, pi_j][edge_label] == label)
                        h_ij0 += h_ij0_p
                    h_ij0_list.append(h_ij0)
                    label_list.append(label)
                # choose one of the best randomly.
                idx_max = np.argwhere(h_ij0_list == np.max(h_ij0_list)).flatten().tolist()
                h_ij0_max = h_ij0_list[idx_max[0]]
                idx_rdm = random.randint(0, len(idx_max) - 1)
                best_label = label_list[idx_max[idx_rdm]]
                       
                # check whether a_ij is 0 or 1.
                sij_norm = 0
                for idx, g in enumerate(Gn):
                    pi_i = pi_p[idx][nd1]
                    pi_j = pi_p[idx][nd2]
                    if g.has_node(pi_i) and g.has_node(pi_j) and g.has_edge(pi_i, pi_j):
                       sij_norm += 1
                if h_ij0_max > len(Gn) * c_er / c_es + sij_norm * (1 - (c_er + c_ei) / c_es):
                    if not G_new.has_edge(nd1, nd2):
                        G_new.add_edge(nd1, nd2)
                    G_new.edges[nd1, nd2][edge_label] = best_label
                else:
                    if G_new.has_edge(nd1, nd2):
                        G_new.remove_edge(nd1, nd2)                
        else: # if edges are unlabeled
            for nd1, nd2, _ in G.edges(data=True):
                sij_norm = 0
                for idx, g in enumerate(Gn):
                    pi_i = pi_p[idx][nd1]
                    pi_j = pi_p[idx][nd2]
                    if g.has_node(pi_i) and g.has_node(pi_j) and g.has_edge(pi_i, pi_j):
                       sij_norm += 1
                if sij_norm > len(Gn) * c_er / (c_er + c_ei):
                    if not G_new.has_edge(nd1, nd2):
                        G_new.add_edge(nd1, nd2)
                else:
                    if G_new.has_edge(nd1, nd2):
                        G_new.remove_edge(nd1, nd2)
                        
        G = G_new.copy()
    
    return G


def GED(g1, g2, lib='gedlib'):
    """
    Compute GED. It is a dummy function for now.
    """
    if lib == 'gedlib':
        saveDataset([g1, g2], [None, None], group='xml', filename='ged_tmp/tmp')
        script.appel()
        script.PyRestartEnv()
        script.PyLoadGXLGraph('ged_tmp/', 'collections/tmp.xml')
        listID = script.PyGetGraphIds()
        script.PySetEditCost("CHEM_1")
        script.PyInitEnv()
        script.PySetMethod("BIPARTITE", "")
        script.PyInitMethod()
        g = listID[0]
        h = listID[1]
        script.PyRunMethod(g, h)
        liste = script.PyGetAllMap(g, h)
        upper = script.PyGetUpperBound(g, h)
        lower = script.PyGetLowerBound(g, h)        
        dis = upper + lower
        pi = liste[0]
        
    return dis, pi


def get_node_labels(Gn, node_label):
    nl = set()
    for G in Gn:
        nl = nl | set(nx.get_node_attributes(G, node_label).values())
    return nl


def get_edge_labels(Gn, edge_label):
    el = set()
    for G in Gn:
        el = el | set(nx.get_edge_attributes(G, edge_label).values())
    return el


if __name__ == '__main__':
    from pygraph.utils.graphfiles import loadDataset
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG.mat',
          'extra_params': {'am_sp_al_nl_el': [0, 0, 3, 1, 2]}}  # node/edge symb
#    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
#          'extra_params': {}} # node nsymb
#    ds = {'name': 'Acyclic', 'dataset': '../datasets/monoterpenoides/trainset_9.ds',
#          'extra_params': {}}
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])

    iam(Gn)