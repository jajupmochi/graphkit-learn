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
from tqdm import tqdm

import sys
#from Cython_GedLib_2 import librariesImport, script
import librariesImport, script
sys.path.insert(0, "../")
from pygraph.utils.graphfiles import saveDataset
from pygraph.utils.graphdataset import get_dataset_attributes
from pygraph.utils.utils import graph_isIdentical, get_node_labels, get_edge_labels
#from pygraph.utils.utils import graph_deepcopy

def iam_moreGraphsAsInit_tryAllPossibleBestGraphs(Gn_median, Gn_candidate, 
        c_ei=3, c_er=3, c_es=1, ite_max=50, epsilon=0.001, 
        node_label='atom', edge_label='bond_type', 
        connected=False, removeNodes=True, allBestInit=False, allBestNodes=False,
        allBestEdges=False,
        params_ged={'ged_cost': 'CHEM_1', 'ged_method': 'IPFP', 'saveGXL': 'benoit'}):
    """See my name, then you know what I do.
    """
    from tqdm import tqdm
#    Gn_median = Gn_median[0:10]
#    Gn_median = [nx.convert_node_labels_to_integers(g) for g in Gn_median]
    if removeNodes:
        node_ir = np.inf # corresponding to the node remove and insertion.
        label_r = 'thanksdanny' # the label for node remove. # @todo: make this label unrepeatable.
    ds_attrs = get_dataset_attributes(Gn_median + Gn_candidate, 
                                      attr_names=['edge_labeled', 'node_attr_dim', 'edge_attr_dim'], 
                                      edge_label=edge_label)

    
    def generate_graph(G, pi_p_forward, label_set):
        G_new_list = [G.copy()] # all "best" graphs generated in this iteration.
#        nx.draw_networkx(G)
#        import matplotlib.pyplot as plt
#        plt.show()
#        print(pi_p_forward)
                    
        # update vertex labels.
        # pre-compute h_i0 for each label.
#        for label in get_node_labels(Gn, node_label):
#            print(label)
#        for nd in G.nodes(data=True):
#            pass
        if not ds_attrs['node_attr_dim']: # labels are symbolic
            for ndi, (nd, _) in enumerate(G.nodes(data=True)):
                h_i0_list = []
                label_list = []
                for label in label_set:
                    h_i0 = 0
                    for idx, g in enumerate(Gn_median):
                        pi_i = pi_p_forward[idx][ndi]
                        if pi_i != node_ir and g.nodes[pi_i][node_label] == label:
                            h_i0 += 1
                    h_i0_list.append(h_i0)
                    label_list.append(label)
                # case when the node is to be removed.
                if removeNodes:
                    h_i0_remove = 0 # @todo: maybe this can be added to the label_set above.
                    for idx, g in enumerate(Gn_median):
                        pi_i = pi_p_forward[idx][ndi]
                        if pi_i == node_ir:
                            h_i0_remove += 1
                    h_i0_list.append(h_i0_remove)
                    label_list.append(label_r)
                # get the best labels.
                idx_max = np.argwhere(h_i0_list == np.max(h_i0_list)).flatten().tolist()
                if allBestNodes: # choose all best graphs.                    
                    nlabel_best = [label_list[idx] for idx in idx_max]
                    # generate "best" graphs with regard to "best" node labels.
                    G_new_list_nd = []
                    for g in G_new_list: # @todo: seems it can be simplified. The G_new_list will only contain 1 graph for now.
                        for nl in nlabel_best:
                            g_tmp = g.copy()
                            if nl == label_r:
                                g_tmp.remove_node(nd)
                            else:
                                g_tmp.nodes[nd][node_label] = nl
                            G_new_list_nd.append(g_tmp)
    #                            nx.draw_networkx(g_tmp)
    #                            import matplotlib.pyplot as plt
    #                            plt.show()
    #                            print(g_tmp.nodes(data=True))
    #                            print(g_tmp.edges(data=True))
                    G_new_list = [ggg.copy() for ggg in G_new_list_nd]
                else: 
                    # choose one of the best randomly.
                    h_ij0_max = h_i0_list[idx_max[0]]
                    idx_rdm = random.randint(0, len(idx_max) - 1)
                    best_label = label_list[idx_max[idx_rdm]]
                           
                    # check whether a_ij is 0 or 1.
                    g_new = G_new_list[0]
                    if best_label == label_r:
                        g_new.remove_node(nd) 
                    else:
                        g_new.nodes[nd][node_label] = best_label
                    G_new_list = [g_new]
        else: # labels are non-symbolic
            for ndi, (nd, _) in enumerate(G.nodes(data=True)):
                Si_norm = 0
                phi_i_bar = np.array([0.0 for _ in range(ds_attrs['node_attr_dim'])])
                for idx, g in enumerate(Gn_median):
                    pi_i = pi_p_forward[idx][ndi]
                    if g.has_node(pi_i): #@todo: what if no g has node? phi_i_bar = 0?
                        Si_norm += 1
                        phi_i_bar += np.array([float(itm) for itm in g.nodes[pi_i]['attributes']])                
                phi_i_bar /= Si_norm
                G_new_list[0].nodes[nd]['attributes'] = phi_i_bar
                
#        for g in G_new_list:
#            import matplotlib.pyplot as plt 
#            nx.draw(g, labels=nx.get_node_attributes(g, 'atom'), with_labels=True)
#            plt.show()
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
                                            
        # update edge labels and adjacency matrix.
        if ds_attrs['edge_labeled']:
            G_new_list_edge = []
            for g_new in G_new_list:
                nd_list = [n for n in g_new.nodes()]
                g_tmp_list = [g_new.copy()]
                for nd1i in range(nx.number_of_nodes(g_new)): 
                    nd1 = nd_list[nd1i]# @todo: not just edges, but all pairs of nodes
                    for nd2i in range(nd1i + 1, nx.number_of_nodes(g_new)):
                        nd2 = nd_list[nd2i]
#                for nd1, nd2, _ in g_new.edges(data=True): 
                        h_ij0_list = []
                        label_list = []
                        # @todo: compute edge label set before.
                        for label in get_edge_labels(Gn_median, edge_label):
                            h_ij0 = 0
                            for idx, g in enumerate(Gn_median):
                                pi_i = pi_p_forward[idx][nd1i]
                                pi_j = pi_p_forward[idx][nd2i]
                                h_ij0_p = (g.has_node(pi_i) and g.has_node(pi_j) and 
                                           g.has_edge(pi_i, pi_j) and 
                                           g.edges[pi_i, pi_j][edge_label] == label)
                                h_ij0 += h_ij0_p
                            h_ij0_list.append(h_ij0)
                            label_list.append(label)
    #                    # case when the edge is to be removed.
    #                    h_ij0_remove = 0
    #                    for idx, g in enumerate(Gn_median):
    #                        pi_i = pi_p_forward[idx][nd1i]
    #                        pi_j = pi_p_forward[idx][nd2i]
    #                        if g.has_node(pi_i) and g.has_node(pi_j) and not 
    #                            g.has_edge(pi_i, pi_j):
    #                            h_ij0_remove += 1
    #                    h_ij0_list.append(h_ij0_remove)
    #                    label_list.append(label_r)
                        
                        # get the best labels.
                        idx_max = np.argwhere(h_ij0_list == np.max(h_ij0_list)).flatten().tolist()
                        if allBestEdges: # choose all best graphs.
                            elabel_best = [label_list[idx] for idx in idx_max]
                            h_ij0_max = [h_ij0_list[idx] for idx in idx_max]
                            # generate "best" graphs with regard to "best" node labels.
                            G_new_list_ed = []
                            for g_tmp in g_tmp_list: # @todo: seems it can be simplified. The G_new_list will only contain 1 graph for now.
                                for idxl, el in enumerate(elabel_best):
                                    g_tmp_copy = g_tmp.copy()
                                    # check whether a_ij is 0 or 1.
                                    sij_norm = 0
                                    for idx, g in enumerate(Gn_median):
                                        pi_i = pi_p_forward[idx][nd1i]
                                        pi_j = pi_p_forward[idx][nd2i]
                                        if g.has_node(pi_i) and g.has_node(pi_j) and \
                                            g.has_edge(pi_i, pi_j):
                                           sij_norm += 1
                                    if h_ij0_max[idxl] > len(Gn_median) * c_er / c_es + \
                                        sij_norm * (1 - (c_er + c_ei) / c_es):
                                        if not g_tmp_copy.has_edge(nd1, nd2):
                                            g_tmp_copy.add_edge(nd1, nd2)
                                        g_tmp_copy.edges[nd1, nd2][edge_label] = elabel_best[idxl]
                                    else:
                                        if g_tmp_copy.has_edge(nd1, nd2):
                                            g_tmp_copy.remove_edge(nd1, nd2)
                                    G_new_list_ed.append(g_tmp_copy)
                            g_tmp_list = [ggg.copy() for ggg in G_new_list_ed]
                        else: # choose one of the best randomly.
                            h_ij0_max = h_ij0_list[idx_max[0]]
                            idx_rdm = random.randint(0, len(idx_max) - 1)
                            best_label = label_list[idx_max[idx_rdm]]
                                   
                            # check whether a_ij is 0 or 1.
                            sij_norm = 0
                            for idx, g in enumerate(Gn_median):
                                pi_i = pi_p_forward[idx][nd1i]
                                pi_j = pi_p_forward[idx][nd2i]
                                if g.has_node(pi_i) and g.has_node(pi_j) and g.has_edge(pi_i, pi_j):
                                   sij_norm += 1
                            if h_ij0_max > len(Gn_median) * c_er / c_es + sij_norm * (1 - (c_er + c_ei) / c_es):
                                if not g_new.has_edge(nd1, nd2):
                                    g_new.add_edge(nd1, nd2)
                                g_new.edges[nd1, nd2][edge_label] = best_label
                            else:
                                if g_new.has_edge(nd1, nd2):
                                    g_new.remove_edge(nd1, nd2) 
                            g_tmp_list = [g_new]
                G_new_list_edge += g_tmp_list
            G_new_list = [ggg.copy() for ggg in G_new_list_edge]    
                    
               
        else: # if edges are unlabeled
            # @todo: is this even right? G or g_tmp? check if the new one is right
            # @todo: works only for undirected graphs.
            
            for g_tmp in G_new_list:
                nd_list = [n for n in g_tmp.nodes()]
                for nd1i in range(nx.number_of_nodes(g_tmp)):
                    nd1 = nd_list[nd1i]
                    for nd2i in range(nd1i + 1, nx.number_of_nodes(g_tmp)):
                        nd2 = nd_list[nd2i]
                        sij_norm = 0
                        for idx, g in enumerate(Gn_median):
                            pi_i = pi_p_forward[idx][nd1i]
                            pi_j = pi_p_forward[idx][nd2i]
                            if g.has_node(pi_i) and g.has_node(pi_j) and g.has_edge(pi_i, pi_j):
                               sij_norm += 1
                        if sij_norm > len(Gn_median) * c_er / (c_er + c_ei):
                            # @todo: should we consider if nd1 and nd2 in g_tmp?
                            # or just add the edge anyway?
                            if g_tmp.has_node(nd1) and g_tmp.has_node(nd2) \
                                and not g_tmp.has_edge(nd1, nd2):
                                g_tmp.add_edge(nd1, nd2)
#                        else: # @todo: which to use?
                        elif sij_norm < len(Gn_median) * c_er / (c_er + c_ei):
                            if g_tmp.has_edge(nd1, nd2):
                                g_tmp.remove_edge(nd1, nd2)
                        # do not change anything when equal.     
                        
#        for i, g in enumerate(G_new_list):
#            import matplotlib.pyplot as plt 
#            nx.draw(g, labels=nx.get_node_attributes(g, 'atom'), with_labels=True)
##            plt.savefig("results/gk_iam/simple_two/xx" + str(i) + ".png", format="PNG")
#            plt.show()
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
        
#        # find the best graph generated in this iteration and update pi_p.
        # @todo: should we update all graphs generated or just the best ones?
        dis_list, pi_forward_list = median_distance(G_new_list, Gn_median, 
            **params_ged)
        # @todo: should we remove the identical and connectivity check? 
        # Don't know which is faster.
        if ds_attrs['node_attr_dim'] == 0 and ds_attrs['edge_attr_dim'] == 0:
            G_new_list, idx_list = remove_duplicates(G_new_list)
            pi_forward_list = [pi_forward_list[idx] for idx in idx_list]
            dis_list = [dis_list[idx] for idx in idx_list]
#        if connected == True:
#            G_new_list, idx_list = remove_disconnected(G_new_list)
#            pi_forward_list = [pi_forward_list[idx] for idx in idx_list]
#        idx_min_list = np.argwhere(dis_list == np.min(dis_list)).flatten().tolist()
#        dis_min = dis_list[idx_min_tmp_list[0]]
#        pi_forward_list = [pi_forward_list[idx] for idx in idx_min_list]
#        G_new_list = [G_new_list[idx] for idx in idx_min_list] 
        
#        for g in G_new_list:
#            import matplotlib.pyplot as plt 
#            nx.draw_networkx(g)
#            plt.show()
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
        
        return G_new_list, pi_forward_list, dis_list
    
    
    def best_median_graphs(Gn_candidate, pi_all_forward, dis_all):
        idx_min_list = np.argwhere(dis_all == np.min(dis_all)).flatten().tolist()
        dis_min = dis_all[idx_min_list[0]]
        pi_forward_min_list = [pi_all_forward[idx] for idx in idx_min_list]
        G_min_list = [Gn_candidate[idx] for idx in idx_min_list]
        return G_min_list, pi_forward_min_list, dis_min
    
    
    def iteration_proc(G, pi_p_forward, cur_sod):
        G_list = [G]
        pi_forward_list = [pi_p_forward]
        old_sod = cur_sod * 2
        sod_list = [cur_sod]
        dis_list = [cur_sod]
        # iterations.
        itr = 0
        # @todo: what if difference == 0?
#        while itr < ite_max and (np.abs(old_sod - cur_sod) > epsilon or
#                                 np.abs(old_sod - cur_sod) == 0):
        while itr < ite_max and np.abs(old_sod - cur_sod) > epsilon:
#        for itr in range(0, 5): # the convergence condition?
            print('itr_iam is', itr)
            G_new_list = []
            pi_forward_new_list = []
            dis_new_list = []
            for idx, g in enumerate(G_list):
                label_set = get_node_labels(Gn_median + [g], node_label)                        
                G_tmp_list, pi_forward_tmp_list, dis_tmp_list = generate_graph(
                    g, pi_forward_list[idx], label_set)
                G_new_list += G_tmp_list
                pi_forward_new_list += pi_forward_tmp_list
                dis_new_list += dis_tmp_list
            # @todo: need to remove duplicates here?
            G_list = [ggg.copy() for ggg in G_new_list]
            pi_forward_list = [pitem.copy() for pitem in pi_forward_new_list]
            dis_list = dis_new_list[:]
            
            old_sod = cur_sod
            cur_sod = np.min(dis_list)
            sod_list.append(cur_sod)
            
            itr += 1
        
        # @todo: do we return all graphs or the best ones?
        # get the best ones of the generated graphs.
        G_list, pi_forward_list, dis_min = best_median_graphs(
            G_list, pi_forward_list, dis_list)
        
        if ds_attrs['node_attr_dim'] == 0 and ds_attrs['edge_attr_dim'] == 0:
            G_list, idx_list = remove_duplicates(G_list)
            pi_forward_list = [pi_forward_list[idx] for idx in idx_list]
#            dis_list = [dis_list[idx] for idx in idx_list]
            
#        import matplotlib.pyplot as plt
#        for g in G_list:             
#            nx.draw_networkx(g)
#            plt.show()
#            print(g.nodes(data=True))
#            print(g.edges(data=True))
            
        print('\nsods:', sod_list, '\n')
            
        return G_list, pi_forward_list, dis_min
    
    
    def remove_duplicates(Gn):
        """Remove duplicate graphs from list.
        """
        Gn_new = []
        idx_list = []
        for idx, g in enumerate(Gn):
            dupl = False
            for g_new in Gn_new:
                if graph_isIdentical(g_new, g):
                    dupl = True
                    break
            if not dupl:
                Gn_new.append(g)
                idx_list.append(idx)
        return Gn_new, idx_list
    
    
    def remove_disconnected(Gn):
        """Remove disconnected graphs from list.
        """
        Gn_new = []
        idx_list = []
        for idx, g in enumerate(Gn):
            if nx.is_connected(g):
                Gn_new.append(g)
                idx_list.append(idx)
        return Gn_new, idx_list

   
    # phase 1: initilize.
    # compute set-median.
    dis_min = np.inf
    dis_list, pi_forward_all = median_distance(Gn_candidate, Gn_median,
        **params_ged)
    # find all smallest distances.
    if allBestInit: # try all best init graphs.
        idx_min_list = range(len(dis_list))
        dis_min = dis_list
    else:
        idx_min_list = np.argwhere(dis_list == np.min(dis_list)).flatten().tolist()
        dis_min = [dis_list[idx_min_list[0]]] * len(idx_min_list)
        
    
    # phase 2: iteration.
    G_list = []
    dis_list = []
    pi_forward_list = []
    for idx_tmp, idx_min in enumerate(idx_min_list):
#        print('idx_min is', idx_min)
        G = Gn_candidate[idx_min].copy()
        # list of edit operations.        
        pi_p_forward = pi_forward_all[idx_min]
#        pi_p_backward = pi_all_backward[idx_min]        
        Gi_list, pi_i_forward_list, dis_i_min = iteration_proc(G, pi_p_forward, dis_min[idx_tmp])            
        G_list += Gi_list
        dis_list += [dis_i_min] * len(Gi_list)
        pi_forward_list += pi_i_forward_list
        
        
    if ds_attrs['node_attr_dim'] == 0 and ds_attrs['edge_attr_dim'] == 0:
        G_list, idx_list = remove_duplicates(G_list)
        dis_list = [dis_list[idx] for idx in idx_list]
        pi_forward_list = [pi_forward_list[idx] for idx in idx_list]
    if connected == True:
        G_list_con, idx_list = remove_disconnected(G_list)
        # if there is no connected graphs at all, then remain the disconnected ones.
        if len(G_list_con) > 0: # @todo: ??????????????????????????
            G_list = G_list_con
            dis_list = [dis_list[idx] for idx in idx_list]
            pi_forward_list = [pi_forward_list[idx] for idx in idx_list]

#    import matplotlib.pyplot as plt 
#    for g in G_list:
#        nx.draw_networkx(g)
#        plt.show()
#        print(g.nodes(data=True))
#        print(g.edges(data=True))
    
    # get the best median graphs
#    dis_list, pi_forward_list = median_distance(G_list, Gn_median,
#        **params_ged)
    G_min_list, pi_forward_min_list, dis_min = best_median_graphs(
            G_list, pi_forward_list, dis_list)
#    for g in G_min_list:
#        nx.draw_networkx(g)
#        plt.show()
#        print(g.nodes(data=True))
#        print(g.edges(data=True))
    # randomly choose one graph.
    idx_rdm = random.randint(0, len(G_min_list) - 1)
    G_min_list = [G_min_list[idx_rdm]]   
    
    return G_min_list, dis_min
















###############################################################################
    
def iam(Gn, c_ei=3, c_er=3, c_es=1, node_label='atom', edge_label='bond_type', 
        connected=True):
    """See my name, then you know what I do.
    """
#    Gn = Gn[0:10]
    Gn = [nx.convert_node_labels_to_integers(g) for g in Gn]
    
    # phase 1: initilize.
    # compute set-median.
    dis_min = np.inf
    pi_p = []
    pi_all = []
    for idx1, G_p in enumerate(Gn):
        dist_sum = 0
        pi_all.append([])
        for idx2, G_p_prime in enumerate(Gn):
            dist_tmp, pi_tmp, _ = GED(G_p, G_p_prime)
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
    for itr in range(0, 10): # @todo: the convergence condition?
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
        
        # update pi_p
        pi_p = []
        for idx1, G_p in enumerate(Gn):
            dist_tmp, pi_tmp, _ = GED(G, G_p)
            pi_p.append(pi_tmp)
    
    return G


def GED(g1, g2, lib='gedlib', cost='CHEM_1', method='IPFP', saveGXL='benoit', 
        stabilizer='min'):
    """
    Compute GED.
    """
    if lib == 'gedlib':
        # transform dataset to the 'xml' file as the GedLib required.
        saveDataset([g1, g2], [None, None], group='xml', filename='ged_tmp/tmp',
                    xparams={'method': saveGXL})
    #        script.appel()
        script.PyRestartEnv()
        script.PyLoadGXLGraph('ged_tmp/', 'ged_tmp/tmp.xml')
        listID = script.PyGetGraphIds()
        script.PySetEditCost(cost) #("CHEM_1")
        script.PyInitEnv()
        script.PySetMethod(method, "")
        script.PyInitMethod()
        g = listID[0]
        h = listID[1]
        if stabilizer == None:
            script.PyRunMethod(g, h)
            pi_forward, pi_backward = script.PyGetAllMap(g, h)
            upper = script.PyGetUpperBound(g, h)
            lower = script.PyGetLowerBound(g, h)        
        elif stabilizer == 'min':
            upper = np.inf
            for itr in range(50):                
                script.PyRunMethod(g, h)                
                upper_tmp = script.PyGetUpperBound(g, h)                
                if upper_tmp < upper:
                    upper = upper_tmp
                    pi_forward, pi_backward = script.PyGetAllMap(g, h)
                    lower = script.PyGetLowerBound(g, h)
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


def median_distance(Gn, Gn_median, measure='ged', verbose=False, 
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


# --------------------------- These are tests --------------------------------#
    
def test_iam_with_more_graphs_as_init(Gn, G_candidate, c_ei=3, c_er=3, c_es=1, 
                                      node_label='atom', edge_label='bond_type'):
    """See my name, then you know what I do.
    """
#    Gn = Gn[0:10]
    Gn = [nx.convert_node_labels_to_integers(g) for g in Gn]
    
    # phase 1: initilize.
    # compute set-median.
    dis_min = np.inf
#    pi_p = []
    pi_all_forward = []
    pi_all_backward = []
    for idx1, G_p in tqdm(enumerate(G_candidate), desc='computing GEDs', file=sys.stdout):
        dist_sum = 0
        pi_all_forward.append([])
        pi_all_backward.append([])
        for idx2, G_p_prime in enumerate(Gn):
            dist_tmp, pi_tmp_forward, pi_tmp_backward = GED(G_p, G_p_prime)
            pi_all_forward[idx1].append(pi_tmp_forward)
            pi_all_backward[idx1].append(pi_tmp_backward)
            dist_sum += dist_tmp
        if dist_sum <= dis_min:
            dis_min = dist_sum
            G = G_p.copy()
            idx_min = idx1
    # list of edit operations.        
    pi_p_forward = pi_all_forward[idx_min]
    pi_p_backward = pi_all_backward[idx_min]
            
    # phase 2: iteration.
    ds_attrs = get_dataset_attributes(Gn + [G], attr_names=['edge_labeled', 'node_attr_dim'], 
                                      edge_label=edge_label)
    label_set = get_node_labels(Gn + [G], node_label)
    for itr in range(0, 10): # @todo: the convergence condition?
        G_new = G.copy()
        # update vertex labels.
        # pre-compute h_i0 for each label.
#        for label in get_node_labels(Gn, node_label):
#            print(label)
#        for nd in G.nodes(data=True):
#            pass
        if not ds_attrs['node_attr_dim']: # labels are symbolic
            for nd in G.nodes():
                h_i0_list = []
                label_list = []
                for label in label_set:
                    h_i0 = 0
                    for idx, g in enumerate(Gn):
                        pi_i = pi_p_forward[idx][nd]
                        if g.has_node(pi_i) and g.nodes[pi_i][node_label] == label:
                            h_i0 += 1
                    h_i0_list.append(h_i0)
                    label_list.append(label)
                # choose one of the best randomly.
                idx_max = np.argwhere(h_i0_list == np.max(h_i0_list)).flatten().tolist()
                idx_rdm = random.randint(0, len(idx_max) - 1)
                G_new.nodes[nd][node_label] = label_list[idx_max[idx_rdm]]
        else: # labels are non-symbolic
            for nd in G.nodes():
                Si_norm = 0
                phi_i_bar = np.array([0.0 for _ in range(ds_attrs['node_attr_dim'])])
                for idx, g in enumerate(Gn):
                    pi_i = pi_p_forward[idx][nd]
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
                        pi_i = pi_p_forward[idx][nd1]
                        pi_j = pi_p_forward[idx][nd2]
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
                    pi_i = pi_p_forward[idx][nd1]
                    pi_j = pi_p_forward[idx][nd2]
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
            # @todo: works only for undirected graphs.
            for nd1 in range(nx.number_of_nodes(G)):
                for nd2 in range(nd1 + 1, nx.number_of_nodes(G)):
                    sij_norm = 0
                    for idx, g in enumerate(Gn):
                        pi_i = pi_p_forward[idx][nd1]
                        pi_j = pi_p_forward[idx][nd2]
                        if g.has_node(pi_i) and g.has_node(pi_j) and g.has_edge(pi_i, pi_j):
                           sij_norm += 1
                    if sij_norm > len(Gn) * c_er / (c_er + c_ei):
                        if not G_new.has_edge(nd1, nd2):
                            G_new.add_edge(nd1, nd2)
                    elif sij_norm < len(Gn) * c_er / (c_er + c_ei):
                        if G_new.has_edge(nd1, nd2):
                            G_new.remove_edge(nd1, nd2)
                    # do not change anything when equal.
                        
        G = G_new.copy()
        
        # update pi_p
        pi_p_forward = []
        for G_p in Gn:
            dist_tmp, pi_tmp_forward, pi_tmp_backward = GED(G, G_p)
            pi_p_forward.append(pi_tmp_forward)
    
    return G


###############################################################################




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