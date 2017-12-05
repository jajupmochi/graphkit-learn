import ot
import sys
import pathlib
sys.path.insert(0, "../")

from pygraph.utils.graphfiles import loadDataset
from pygraph.ged.costfunctions import ConstantCostFunction
from pygraph.utils.utils import getSPLengths
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment
from pygraph.ged.GED import ged
import scipy

def pad(C, n):
    C_pad = np.zeros((n, n))
    C_pad[:C.shape[0], :C.shape[1]] = C
    return C_pad

if (__name__ == "__main__"):
    ds_filename = "/home/bgauzere/work/Datasets/Acyclic/dataset_bps.ds"
    dataset, y = loadDataset(ds_filename)
    cf = ConstantCostFunction(1, 3, 1, 3)
    N = len(dataset)

    pairs = list()
    
    ged_distances = list() #np.zeros((N, N))
    gw_distances = list() #np.zeros((N, N))
    for i in tqdm(range(0, N)):
        for j in tqdm(range(i, N)):
            G1 = dataset[i]
            G2 = dataset[j]
            n = G1.number_of_nodes()
            m = G2.number_of_nodes()
            if(n == m):
                C1 = getSPLengths(G1)
                C2 = getSPLengths(G2)

                C1 /= C1.max()
                C2 /= C2.max()

                dim = max(n, m)
                if(n < m):
                    C1 = pad(C1, dim)
                elif (m < n):
                    C2 = pad(C2, dim)

                p = ot.unif(dim)
                q = ot.unif(dim)

                gw = ot.gromov_wasserstein(C1, C2, p, q,
                                           'square_loss', epsilon=5e-3)
                row_ind, col_ind = linear_sum_assignment(-gw)
                rho = col_ind
                varrho = row_ind[np.argsort(col_ind)]
                pairs.append((i,j))
                gw_distances.append(ged(G1, G2, cf=cf, rho=rho, varrho=varrho)[0])

                ged_distances.append(ged(G1, G2, cf=cf)[0])

    print("Moyenne sur Riesen : {}".format(np.mean(ged_distances)))
    print("Moyenne sur GW : {} ".format(np.mean(gw_distances)))

    np.save("distances_riesen", ged_distances)
    np.save("distances_gw", gw_distances)
