import numpy as np
from scipy.optimize import linear_sum_assignment


class BasicCostFunction:
    def __init__(self, cns, cni, ces, cei):
        self.cns_ = cns
        self.cni_ = self.cnd_ = cni
        self.ces_ = ces
        self.cei_ = self.ced_ = cei

    def cns(self, u, v, G1, G2):
        return (G1.node[u]['label'] != G2.node[v]['label'])*self.cns_

    def cnd(self, u, G1):
        return self.cnd_

    def cni(self, v, G2):
        return self.cni_

    def ces(self, e1, e2, G1, G2):
        """tester avec des attributs autres que symboliques en testant
        l'operateur __eq__"""
        return (e1[2]['label'] != e2[2]['label'])*self.ces_

    def ced(self, e1, G1):
        return self.ced_

    def cei(self, e2, G2):
        return self.cei_


class RiesenCostFunction(BasicCostFunction):
    def __init__(self, cf):
        BasicCostFunction.__init__(self, cf.cns_, cf.cni_, cf.ces_, cf.cei_)

    def cns(self, u, v, G1, G2):
        """ u et v sont des id de noeuds """
        n = len(G1[u])
        m = len(G2[v])
        sub_C = np.ones([n+m, n+m]) * np.inf
        sub_C[n:, m:] = 0
        i = 0
        l_nbr_u = G1[u]
        l_nbr_v = G2[v]
        for nbr_u in l_nbr_u:
            j = 0
            e1 = [u, nbr_u, G1[u][nbr_u]]
            for nbr_v in G2[v]:
                e2 = [v, nbr_v, G2[v][nbr_v]]
                sub_C[i, j] = self.ces(e1, e2, G1, G2)
                j += 1
            i += 1

        i = 0
        for nbr_u in l_nbr_u:
            sub_C[i, m+i] = self.ced([u, nbr_u, G1[u][nbr_u]], G1)
            i += 1

        j = 0
        for nbr_v in l_nbr_v:
            sub_C[n+j, j] = self.cei([v, nbr_v, G2[v][nbr_v]], G2)
            j += 1
        row_ind, col_ind = linear_sum_assignment(sub_C)
        cost = np.sum(sub_C[row_ind, col_ind])
        return BasicCostFunction.cns(self, u, v, G1, G2) + cost

    def cnd(self, u, G1):
        cost = 0
        for nbr in G1[u]:
            cost += BasicCostFunction.ced(self,[u,nbr,G1[u][nbr]],G1)
            
        return BasicCostFunction.cnd(self,u,G1) + cost

    def cni(self, v, G2):
        cost = 0
        for nbr in G2[v]:
            cost += BasicCostFunction.cei(self, [v,nbr,G2[v][nbr]], G2)
            
        return BasicCostFunction.cni(self, v, G2) + cost


class NeighboorhoodCostFunction(BasicCostFunction):
    def __init__(self, cf):
        BasicCostFunction.__init__(self, cf.cns_, cf.cni_, cf.ces_, cf.cei_)

    def cns(self, u, v, G1, G2):
        """ u et v sont des id de noeuds """
        n = len(G1[u])
        m = len(G2[v])
        sub_C = np.ones([n+m, n+m]) * np.inf
        sub_C[n:, m:] = 0
        i = 0
        l_nbr_u = G1[u]
        l_nbr_v = G2[v]
        for nbr_u in l_nbr_u:
            j = 0
            e1 = [u, nbr_u, G1[u][nbr_u]]
            for nbr_v in G2[v]:
                e2 = [v, nbr_v, G2[v][nbr_v]]
                sub_C[i, j] = self.ces(e1, e2, G1, G2)
                sub_C[i, j] += BasicCostFunction.cns(self,
                                                     nbr_u, nbr_v, G1, G2)
                j += 1
            i += 1

        i = 0
        for nbr_u in l_nbr_u:
            sub_C[i, m+i] = self.ced([u, nbr_u, G1[u][nbr_u]], G1)
            sub_C[i, m+i] += BasicCostFunction.cnd(self, nbr_u, G1)
            i += 1

        j = 0
        for nbr_v in l_nbr_v:
            sub_C[n+j, j] = self.cei([v, nbr_v, G2[v][nbr_v]], G2)
            sub_C[n+j, j] += BasicCostFunction.cni(self, nbr_v, G2)
            j += 1

        row_ind, col_ind = linear_sum_assignment(sub_C)
        cost = np.sum(sub_C[row_ind, col_ind])
        return BasicCostFunction.cns(self, u, v, G1, G2) + cost

    def cnd(self, u, G1):
        cost = 0
        for nbr in G1[u]:
            cost += BasicCostFunction.ced(self, [u, nbr, G1[u][nbr]], G1)
        return BasicCostFunction.cnd(self, u, G1) + cost

    def cni(self, v, G2):
        cost = 0
        for nbr in G2[v]:
            cost += BasicCostFunction.cei(self, [v, nbr, G2[v][nbr]], G2)
        return BasicCostFunction.cni(self, v, G2) + cost
