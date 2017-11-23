import networkx as nx
import numpy as np


def getSPLengths(G1):
    sp = nx.shortest_path(G1)
    distances = np.zeros((G1.number_of_nodes(), G1.number_of_nodes()))
    for i in sp.keys():
        for j in sp[i].keys():
            distances[i, j] = len(sp[i][j])-1
    return distances
