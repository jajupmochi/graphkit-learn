import sys
import pathlib
sys.path.insert(0, "../")
import time

from collections import Counter
from itertools import chain

import networkx as nx
import numpy as np


def find_paths(G, source_node, length):
    if length == 0:
        return [[source_node]]
    path = [ [source_node] + path for neighbor in G[source_node] \
        for path in find_paths(G, neighbor, length - 1) if source_node not in path ]
    return path

def find_all_paths(G, length):
    all_paths = []
    for node in G:
        all_paths.extend(find_paths(G, node, length))
    all_paths_r = [ path[::-1] for path in all_paths ]
    
    # remove double direction
    for idx, path in enumerate(all_paths[:-1]):
        for path2 in all_paths_r[idx+1::]:
            if path == path2:
                all_paths[idx] = []
                break
            
    return list(filter(lambda a: a != [], all_paths))

def get_canonkey(G, node_label = 'atom', edge_label = 'bond_type', labeled = True):
    
        patterns = {}
        canonkey = {} # canonical key
    
        ### structural analysis ###
        # linear patterns
        patterns['0'] = G.nodes()
        canonkey['0'] = nx.number_of_nodes(G)
        for i in range(1, 6):
            patterns[str(i)] = find_all_paths(G, i)
            canonkey[str(i)] = len(patterns[str(i)])
                
        # n-star patterns
        patterns['3star'] = [ [node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 3 ]
        patterns['4star'] = [ [node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 4 ]
        patterns['5star'] = [ [node] + [neighbor for neighbor in G[node]] for node in G.nodes() if G.degree(node) == 5 ]        
        # n-star patterns
        canonkey['6'] = len(patterns['3star'])
        canonkey['8'] = len(patterns['4star'])
        canonkey['d'] = len(patterns['5star'])
        
        # pattern 7
        patterns['7'] = []
        for pattern in patterns['3star']:
            for i in range(1, len(pattern)):
                if G.degree(pattern[i]) >= 2:
                    pattern_t = pattern[:]
                    pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
                    for neighborx in G[pattern[i]]:
                        if neighborx != pattern[0]:
                            new_pattern = pattern_t + [ neighborx ]
                            patterns['7'].append(new_pattern)
        canonkey['7'] = len(patterns['7'])
        
        # pattern 11
        patterns['11'] = []
        for pattern in patterns['4star']:
            for i in range(1, len(pattern)):
                if G.degree(pattern[i]) >= 2:
                    pattern_t = pattern[:]
                    pattern_t[i], pattern_t[4] = pattern_t[4], pattern_t[i]
                    for neighborx in G[pattern[i]]:
                        if neighborx != pattern[0]:
                            new_pattern = pattern_t + [ neighborx ]
                            patterns['11'].append(new_pattern)
        canonkey['b'] = len(patterns['11'])
        
        # pattern 12
        patterns['12'] = []
        rootlist = []
        for pattern in patterns['3star']:
            if pattern[0] not in rootlist:
                rootlist.append(pattern[0])
                for i in range(1, len(pattern)):
                    if G.degree(pattern[i]) >= 3:
                        rootlist.append(pattern[i])
                        pattern_t = pattern[:]
                        pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
                        for neighborx1 in G[pattern[i]]:
                            if neighborx1 != pattern[0]:
                                for neighborx2 in G[pattern[i]]:
                                    if neighborx1 > neighborx2 and neighborx2 != pattern[0]:
                                        new_pattern = pattern_t + [neighborx1] + [neighborx2]
#                         new_patterns = [ pattern + [neighborx1] + [neighborx2] for neighborx1 in G[pattern[i]] if neighborx1 != pattern[0] for neighborx2 in G[pattern[i]] if (neighborx1 > neighborx2 and neighborx2 != pattern[0]) ]
                                        patterns['12'].append(new_pattern)
        canonkey['c'] = int(len(patterns['12']) / 2)
        
        # pattern 9
        patterns['9'] = []
        for pattern in patterns['3star']:
            for pairs in [ [neighbor1, neighbor2] for neighbor1 in G[pattern[0]] if G.degree(neighbor1) >= 2 \
                for neighbor2 in G[pattern[0]] if G.degree(neighbor2) >= 2 if neighbor1 > neighbor2 ]:
                pattern_t = pattern[:]
                pattern_t[pattern_t.index(pairs[0])], pattern_t[2] = pattern_t[2], pattern_t[pattern_t.index(pairs[0])]
                pattern_t[pattern_t.index(pairs[1])], pattern_t[3] = pattern_t[3], pattern_t[pattern_t.index(pairs[1])]
                for neighborx1 in G[pairs[0]]:
                    if neighborx1 != pattern[0]:
                        for neighborx2 in G[pairs[1]]:
                            if neighborx2 != pattern[0]:
                                new_pattern = pattern_t + [neighborx1] + [neighborx2]
                                patterns['9'].append(new_pattern)
        canonkey['9'] = len(patterns['9'])
        
        # pattern 10
        patterns['10'] = []
        for pattern in patterns['3star']:        
            for i in range(1, len(pattern)):
                if G.degree(pattern[i]) >= 2:
                    for neighborx in G[pattern[i]]:
                        if neighborx != pattern[0] and G.degree(neighborx) >= 2:
                            pattern_t = pattern[:]
                            pattern_t[i], pattern_t[3] = pattern_t[3], pattern_t[i]
                            new_patterns = [ pattern_t + [neighborx] + [neighborxx] for neighborxx in G[neighborx] if neighborxx != pattern[i] ]
                            patterns['10'].extend(new_patterns)
        canonkey['a'] = len(patterns['10'])
        
        ### labeling information ###
        if labeled == True:
            canonkey_l = {}
            
            # linear patterns
            canonkey_t = Counter(list(nx.get_node_attributes(G, node_label).values()))
            for key in canonkey_t:
                canonkey_l['0' + key] = canonkey_t[key]
            
            for i in range(1, 6):
                treelet = []
                for pattern in patterns[str(i)]:
                    canonlist = list(chain.from_iterable((G.node[node][node_label], \
                        G[node][pattern[idx+1]][edge_label]) for idx, node in enumerate(pattern[:-1])))
                    canonlist.append(G.node[pattern[-1]][node_label])
                    canonkey_t = ''.join(canonlist)
                    canonkey_t = canonkey_t if canonkey_t < canonkey_t[::-1] else canonkey_t[::-1]
                    treelet.append(str(i) + canonkey_t)
                canonkey_l.update(Counter(treelet))
            
            # n-star patterns
            for i in range(3, 6):
                treelet = []
                for pattern in patterns[str(i) + 'star']:
                    canonlist = [ G.node[leaf][node_label] + G[leaf][pattern[0]][edge_label] for leaf in pattern[1:] ]
                    canonlist.sort()
                    canonkey_t = ('d' if i == 5 else str(i * 2)) + G.node[pattern[0]][node_label] + ''.join(canonlist)
                    treelet.append(canonkey_t)
                canonkey_l.update(Counter(treelet))
            
            # pattern 7
            treelet = []
            for pattern in patterns['7']:
                canonlist = [ G.node[leaf][node_label] + G[leaf][pattern[0]][edge_label] for leaf in pattern[1:3] ]
                canonlist.sort()
                canonkey_t = '7' + G.node[pattern[0]][node_label] + ''.join(canonlist) \
                    + G.node[pattern[3]][node_label] + G[pattern[3]][pattern[0]][edge_label] \
                     + G.node[pattern[4]][node_label] + G[pattern[4]][pattern[3]][edge_label]
                treelet.append(canonkey_t)
            canonkey_l.update(Counter(treelet))
            
            # pattern 11
            treelet = []
            for pattern in patterns['11']:
                canonlist = [ G.node[leaf][node_label] + G[leaf][pattern[0]][edge_label] for leaf in pattern[1:4] ]
                canonlist.sort()
                canonkey_t = 'b' + G.node[pattern[0]][node_label] + ''.join(canonlist) \
                    + G.node[pattern[4]][node_label] + G[pattern[4]][pattern[0]][edge_label] \
                     + G.node[pattern[5]][node_label] + G[pattern[5]][pattern[4]][edge_label]
                treelet.append(canonkey_t)
            canonkey_l.update(Counter(treelet))

            # pattern 10
            treelet = []
            for pattern in patterns['10']:
                canonkey4 = G.node[pattern[5]][node_label] + G[pattern[5]][pattern[4]][edge_label]
                canonlist = [ G.node[leaf][node_label] + G[leaf][pattern[0]][edge_label] for leaf in pattern[1:3] ]
                canonlist.sort()
                canonkey0 = ''.join(canonlist)
                canonkey_t = 'a' + G.node[pattern[3]][node_label] \
                    + G.node[pattern[4]][node_label] + G[pattern[4]][pattern[3]][edge_label] \
                    + G.node[pattern[0]][node_label] + G[pattern[0]][pattern[3]][edge_label] \
                    + canonkey4 + canonkey0
                treelet.append(canonkey_t)
            canonkey_l.update(Counter(treelet))
            
            # pattern 12
            treelet = []
            for pattern in patterns['12']:
                canonlist0 = [ G.node[leaf][node_label] + G[leaf][pattern[0]][edge_label] for leaf in pattern[1:3] ]
                canonlist0.sort()
                canonlist3 = [ G.node[leaf][node_label] + G[leaf][pattern[3]][edge_label] for leaf in pattern[4:6] ]
                canonlist3.sort()
                canonkey_t1 = 'c' + G.node[pattern[0]][node_label] \
                    + ''.join(canonlist0) \
                    + G.node[pattern[3]][node_label] + G[pattern[3]][pattern[0]][edge_label] \
                    + ''.join(canonlist3)
                    
                canonkey_t2 = 'c' + G.node[pattern[3]][node_label] \
                    + ''.join(canonlist3) \
                    + G.node[pattern[0]][node_label] + G[pattern[0]][pattern[3]][edge_label] \
                    + ''.join(canonlist0)
                    
                treelet.append(canonkey_t1 if canonkey_t1 < canonkey_t2 else canonkey_t2)
            canonkey_l.update(Counter(treelet))
            
            # pattern 9
            treelet = []
            for pattern in patterns['9']:
                canonkey2 = G.node[pattern[4]][node_label] + G[pattern[4]][pattern[2]][edge_label]
                canonkey3 = G.node[pattern[5]][node_label] + G[pattern[5]][pattern[3]][edge_label]
                prekey2 = G.node[pattern[2]][node_label] + G[pattern[2]][pattern[0]][edge_label]
                prekey3 = G.node[pattern[3]][node_label] + G[pattern[3]][pattern[0]][edge_label]
                if prekey2 + canonkey2 < prekey3 + canonkey3:
                    canonkey_t = G.node[pattern[1]][node_label] + G[pattern[1]][pattern[0]][edge_label] \
                        + prekey2 + prekey3 + canonkey2 + canonkey3
                else:
                    canonkey_t = G.node[pattern[1]][node_label] + G[pattern[1]][pattern[0]][edge_label] \
                        + prekey3 + prekey2 + canonkey3 + canonkey2
                treelet.append('9' + G.node[pattern[0]][node_label] + canonkey_t)
            canonkey_l.update(Counter(treelet))
            
            return canonkey_l
                    
        return canonkey
        

def treeletkernel(*args, node_label = 'atom', edge_label = 'bond_type', labeled = True):
    if len(args) == 1: # for a list of graphs
        Gn = args[0]
        Kmatrix = np.zeros((len(Gn), len(Gn)))

        start_time = time.time()
        
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = treeletkernel(Gn[i], Gn[j], labeled = labeled, node_label = node_label, edge_label = edge_label)
                Kmatrix[j][i] = Kmatrix[i][j]

        run_time = time.time() - start_time
        print("\n --- treelet kernel matrix of size %d built in %s seconds ---" % (len(Gn), run_time))
        
        return Kmatrix, run_time
    
    else: # for only 2 graphs
        
        G1 = args[0]
        G = args[1]
        kernel = 0
        
#         start_time = time.time()
                        
        canonkey2 = get_canonkey(G, node_label = node_label, edge_label = edge_label, labeled = labeled)
        canonkey1 = get_canonkey(G1, node_label = node_label, edge_label = edge_label, labeled = labeled)
        
        keys = set(canonkey1.keys()) & set(canonkey2.keys()) # find same canonical keys in both graphs
        vector1 = np.matrix([ (canonkey1[key] if (key in canonkey1.keys()) else 0) for key in keys ])
        vector2 = np.matrix([ (canonkey2[key] if (key in canonkey2.keys()) else 0) for key in keys ])        
        kernel = np.sum(np.exp(- np.square(vector1 - vector2) / 2))
        
#         run_time = time.time() - start_time
#         print("\n --- treelet kernel built in %s seconds ---" % (run_time))

        return kernel#, run_time