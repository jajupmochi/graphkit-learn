from pygraph.ged.costfunctions import ConstantCostFunction, RiesenCostFunction
from pygraph.ged.costfunctions import NeighboorhoodCostFunction
from pygraph.ged.bipartiteGED import computeBipartiteCostMatrix, getOptimalMapping
from scipy.optimize import linear_sum_assignment

def ged(G1, G2, method='Riesen', rho=None, varrho=None,
        cf=ConstantCostFunction(1, 3, 1, 3),
        solver=linear_sum_assignment):
    """Compute Graph Edit Distance between G1 and G2 according to mapping
    encoded within rho and varrho. Graph's node must be indexed by a
    index which is used is rho and varrho 
    NB: Utilisation de
    dictionnaire pour etre plus versatile ?

    """
    if ((rho is None) or (varrho is None)):
        if(method == 'Riesen'):
            cf_bp = RiesenCostFunction(cf,lsap_solver=solver)
        elif(method == 'Neighboorhood'):
            cf_bp = NeighboorhoodCostFunction(cf,lsap_solver=solver)
        elif(method == 'Basic'):
            cf_bp = cf
        else:
            raise NameError('Non existent method ')

        rho, varrho = getOptimalMapping(
            computeBipartiteCostMatrix(G1, G2, cf_bp), lsap_solver=solver)

    n = G1.number_of_nodes()
    m = G2.number_of_nodes()
    ged = 0
    for i in G1.nodes():
        phi_i = rho[i]
        if(phi_i >= m):
            ged += cf.cnd(i, G1)
        else:
            ged += cf.cns(i, phi_i, G1, G2)
    for j in G2.nodes():
        phi_j = varrho[j]
        if(phi_j >= n):
            ged += cf.cni(j, G2)

    for e in G1.edges(data=True):
        i = e[0]
        j = e[1]
        phi_i = rho[i]
        phi_j = rho[j]
        if (phi_i < m) and (phi_j < m):
            mappedEdge = len(list(filter(lambda x: True if
                                         x == phi_j else False, G2[phi_i])))
            if(mappedEdge):
                e2 = [phi_i, phi_j, G2[phi_i][phi_j]]
                min_cost = min(cf.ces(e, e2, G1, G2),
                               cf.ced(e, G1) + cf.cei(e2, G2))
                ged += min_cost
            else:
                ged += cf.ced(e, G1)
        else:
            ged += cf.ced(e, G1)
    for e in G2.edges(data=True):
        i = e[0]
        j = e[1]
        phi_i = varrho[i]
        phi_j = varrho[j]
        if (phi_i < n) and (phi_j < n):
            mappedEdge = len(list(filter(lambda x: True if x == phi_j
                                         else False, G1[phi_i])))
            if(not mappedEdge):
                ged += cf.cei(e, G2)
        else:
            ged += cf.ced(e, G2)
    return ged, rho, varrho
