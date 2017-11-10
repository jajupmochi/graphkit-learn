import numpy as np
from scipy.optimize import linear_sum_assignment
from ged.costfunctions import BasicCostFunction


def computeBipartiteCostMatrix(G1, G2, cf=BasicCostFunction(1, 3, 1, 3)):
    """Compute a Cost Matrix according to cost function cf"""
    n = G1.number_of_nodes()
    m = G2.number_of_nodes()
    nm = n + m
    C = np.ones([nm, nm])*np.inf
    C[n:, m:] = 0

    for u in G1.nodes_iter():
        for v in G2.nodes_iter():
            cost = cf.cns(u, v, G1, G2)
            C[u, v] = cost

    for v in G1.nodes_iter():
        C[v, m + v] = cf.cnd(v, G1)

    for v in G2.nodes_iter():
        C[n + v, v] = cf.cni(v, G2)
    return C


def getOptimalMapping(C):
    """Compute an optimal linear mapping according to cost Matrix C
    inclure les progs C de Seb

    """
    row_ind, col_ind = linear_sum_assignment(C)
    return col_ind, row_ind[np.argsort(col_ind)]
