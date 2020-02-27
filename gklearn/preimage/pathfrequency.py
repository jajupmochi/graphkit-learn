#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:12:15 2019

inferring a graph grom path frequency.
@author: ljia
"""
#import numpy as np
import networkx as nx
from scipy.spatial.distance import hamming
import itertools

def SISF(K, v):
    if output:
        return output
    else:
        return 'no solution'

    
def SISF_M(K, v):
    return output


def GIPF_tree(v_obj, K=1, alphabet=[0, 1]):
    if K == 1:
        n_graph = v_obj[0] + v_obj[1]
        D_T, father_idx = getDynamicTable(n_graph, alphabet)
        
        # get the vector the closest to v_obj.
        if v_obj not in D_T:
            print('no exact solution')
            dis_lim = 1 / len(v_obj) # the possible shortest distance.
            dis_min = 1.0 # minimum proportional distance
            v_min = v_obj
            for vc in D_T:
                if vc[0] + vc[1] == n_graph:
#                    print(vc)
                    dis = hamming(vc, v_obj)
                    if dis < dis_min:
                        dis_min = dis
                        v_min = vc
                    if dis_min <= dis_lim:
                        break
            v_obj = v_min
            
        # obtain required graph by traceback procedure.        
        return getObjectGraph(v_obj, D_T, father_idx, alphabet), v_obj
    
def GIPF_M(K, v):
    return G


def getDynamicTable(n_graph, alphabet=[0, 1]):
    # init. When only one node exists.
    D_T = {(1, 0, 0, 0, 0, 0): 1, (0, 1, 0, 0, 0, 0): 1, (0, 0, 1, 0, 0, 0): 0, 
           (0, 0, 0, 1, 0, 0): 0, (0, 0, 0, 0, 1, 0): 0, (0, 0, 0, 0, 0, 1): 0,}
    D_T = [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)]
    father_idx = [-1, -1] # index of each vector's father
    # add possible vectors.
    for idx, v in enumerate(D_T):
        if v[0] + v[1] < n_graph:
            D_T.append((v[0] + 1, v[1], v[2] + 2, v[3], v[4], v[5]))
            D_T.append((v[0] + 1, v[1], v[2], v[3] + 1, v[4] + 1, v[5]))
            D_T.append((v[0], v[1] + 1, v[2], v[3] + 1, v[4] + 1, v[5]))
            D_T.append((v[0], v[1] + 1, v[2], v[3], v[4], v[5] + 2))
            father_idx += [idx, idx, idx, idx]
    
#    D_T = itertools.chain([(1, 0, 0, 0, 0, 0)], [(0, 1, 0, 0, 0, 0)])
#    father_idx = itertools.chain([-1], [-1]) # index of each vector's father
#    # add possible vectors.
#    for idx, v in enumerate(D_T):
#        if v[0] + v[1] < n_graph:
#            D_T = itertools.chain(D_T, [(v[0] + 1, v[1], v[2] + 2, v[3], v[4], v[5])])
#            D_T = itertools.chain(D_T, [(v[0] + 1, v[1], v[2], v[3] + 1, v[4] + 1, v[5])])
#            D_T = itertools.chain(D_T, [(v[0], v[1] + 1, v[2], v[3] + 1, v[4] + 1, v[5])])
#            D_T = itertools.chain(D_T, [(v[0], v[1] + 1, v[2], v[3], v[4], v[5] + 2)])
#            father_idx = itertools.chain(father_idx, [idx, idx, idx, idx])
    return D_T, father_idx


def getObjectGraph(v_obj, D_T, father_idx, alphabet=[0, 1]):
    g_obj = nx.Graph()
    
    # do vector traceback.
    v_tb = [list(v_obj)] # traceback vectors.
    v_tb_idx = [D_T.index(v_obj)] # indices of traceback vectors.
    while v_tb_idx[-1] > 1:
        idx_pre = father_idx[v_tb_idx[-1]]
        v_tb_idx.append(idx_pre)
        v_tb.append(list(D_T[idx_pre]))
    v_tb = v_tb[::-1] # reverse
#    v_tb_idx = v_tb_idx[::-1]

    # construct tree.
    v_c = v_tb[0] # current vector.
    if v_c[0] == 1:
        g_obj.add_node(0, node_label=alphabet[0])
    else:
        g_obj.add_node(0, node_label=alphabet[1])
    for vct in v_tb[1:]:
        if vct[0] - v_c[0] == 1:
            if vct[2] - v_c[2] == 2: # transfer 1
                label1 = alphabet[0]
                label2 = alphabet[0]
            else: # transfer 2
                label1 = alphabet[1]
                label2 = alphabet[0]
        else: 
            if vct[3] - v_c[3] == 1: # transfer 3
                label1 = alphabet[0]
                label2 = alphabet[1]
            else: # transfer 4
                label1 = alphabet[1]
                label2 = alphabet[1]
        for nd, attr in g_obj.nodes(data=True):
            if attr['node_label'] == label1:
                nb_node = nx.number_of_nodes(g_obj)
                g_obj.add_node(nb_node, node_label=label2)
                g_obj.add_edge(nd, nb_node)
                break
        v_c = vct
    return g_obj


import random
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 

    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


if __name__ == '__main__':
    v_obj = (6, 4, 10, 3, 3, 2)
#    v_obj = (6, 5, 10, 3, 3, 2)
    tree_obj, v_obj = GIPF_tree(v_obj)
    print('One closest vector is', v_obj)
    # plot
    pos = hierarchy_pos(tree_obj, 0) 
    node_labels = nx.get_node_attributes(tree_obj, 'node_label')
    nx.draw(tree_obj, pos=pos, labels=node_labels, with_labels=True)