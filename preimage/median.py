import sys
sys.path.insert(0, "../")
#import pathlib
import numpy as np
import networkx as nx
import time

#import librariesImport
#import script
#sys.path.insert(0, "/home/bgauzere/dev/optim-graphes/")
#import pygraph
from pygraph.utils.graphfiles import loadDataset

def replace_graph_in_env(script, graph, old_id, label='median'):
    """
    Replace a graph in script

    If old_id is -1, add a new graph to the environnemt

    """
    if(old_id > -1):
        script.PyClearGraph(old_id)
    new_id = script.PyAddGraph(label)
    for i in graph.nodes():
        script.PyAddNode(new_id,str(i),graph.node[i]) # !! strings are required bt gedlib
    for e in graph.edges:
        script.PyAddEdge(new_id, str(e[0]),str(e[1]), {})
    script.PyInitEnv()
    script.PySetMethod("IPFP", "")
    script.PyInitMethod()

    return new_id
    
#Dessin median courrant
def draw_Letter_graph(graph, savepath=''):
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    plt.figure()
    pos = {}
    for n in graph.nodes:
        pos[n] = np.array([float(graph.node[n]['attributes'][0]),
           float(graph.node[n]['attributes'][1])])
    nx.draw_networkx(graph, pos)
    if savepath != '':
        plt.savefig(savepath + str(time.time()) + '.eps', format='eps', dpi=300)
    plt.show()
    plt.clf()
    
#compute new mappings
def update_mappings(script,median_id,listID):
    med_distances = {}
    med_mappings = {}
    sod = 0
    for i in range(0,len(listID)):
        script.PyRunMethod(median_id,listID[i])
        med_distances[i] = script.PyGetUpperBound(median_id,listID[i])
        med_mappings[i] = script.PyGetForwardMap(median_id,listID[i])
        sod += med_distances[i]
    return med_distances, med_mappings, sod

def calcul_Sij(all_mappings, all_graphs,i,j):
    s_ij = 0
    for k in range(0,len(all_mappings)):
        cur_graph =  all_graphs[k]
        cur_mapping = all_mappings[k]
        size_graph = cur_graph.order()
        if ((cur_mapping[i] < size_graph) and 
            (cur_mapping[j] < size_graph) and 
            (cur_graph.has_edge(cur_mapping[i], cur_mapping[j]) == True)):
                s_ij += 1
        
    return s_ij

# def update_median_nodes_L1(median,listIdSet,median_id,dataset, mappings):
#     from scipy.stats.mstats import gmean

#     for i in median.nodes():
#         for k in listIdSet:
#             vectors = [] #np.zeros((len(listIdSet),2))
#             if(k != median_id):
#                 phi_i = mappings[k][i]
#                 if(phi_i < dataset[k].order()):
#                     vectors.append([float(dataset[k].node[phi_i]['x']),float(dataset[k].node[phi_i]['y'])])

#         new_labels = gmean(vectors)
#         median.node[i]['x'] = str(new_labels[0])
#         median.node[i]['y'] = str(new_labels[1])
#     return median

def update_median_nodes(median,dataset,mappings):
    #update node attributes
    for i in median.nodes():
        nb_sub=0
        mean_label = {'x' : 0, 'y' : 0}
        for k in range(0,len(mappings)):
            phi_i = mappings[k][i]
            if ( phi_i < dataset[k].order() ):
                nb_sub += 1
                mean_label['x'] += 0.75*float(dataset[k].node[phi_i]['x'])
                mean_label['y'] += 0.75*float(dataset[k].node[phi_i]['y'])
        median.node[i]['x'] = str((1/0.75)*(mean_label['x']/nb_sub))
        median.node[i]['y'] = str((1/0.75)*(mean_label['y']/nb_sub))
    return median

def update_median_edges(dataset, mappings, median, cei=0.425,cer=0.425):
#for letter high, ceir = 1.7, alpha = 0.75
    size_dataset = len(dataset)
    ratio_cei_cer = cer/(cei + cer)
    threshold = size_dataset*ratio_cei_cer
    order_graph_median = median.order()
    for i in range(0,order_graph_median):
        for j in range(i+1,order_graph_median):
            s_ij = calcul_Sij(mappings,dataset,i,j)
            if(s_ij > threshold):
                median.add_edge(i,j)
            else:
                if(median.has_edge(i,j)):
                    median.remove_edge(i,j)
    return median



def compute_median(script, listID, dataset,verbose=False):
    """Compute a graph median of a dataset according to an environment

    Parameters

    script : An gedlib initialized environnement 
    listID (list): a list of ID in script: encodes the dataset 
    dataset (list): corresponding graphs in networkX format. We assume that graph
    listID[i] corresponds to dataset[i]

    Returns:
    A networkX graph, which is the median, with corresponding sod
    """
    print(len(listID))
    median_set_index, median_set_sod = compute_median_set(script, listID)
    print(median_set_index)
    print(median_set_sod)
    sods = []
    #Ajout median dans environnement
    set_median = dataset[median_set_index].copy()
    median = dataset[median_set_index].copy()
    cur_med_id = replace_graph_in_env(script,median,-1)
    med_distances, med_mappings, cur_sod = update_mappings(script,cur_med_id,listID)
    sods.append(cur_sod)
    if(verbose):
        print(cur_sod)
    ite_max = 50
    old_sod = cur_sod * 2
    ite = 0
    epsilon = 0.001

    best_median 
    while((ite < ite_max) and (np.abs(old_sod - cur_sod) > epsilon )):
        median = update_median_nodes(median,dataset, med_mappings)
        median = update_median_edges(dataset,med_mappings,median)

        cur_med_id = replace_graph_in_env(script,median,cur_med_id)
        med_distances, med_mappings, cur_sod = update_mappings(script,cur_med_id,listID)
        
        
        sods.append(cur_sod)
        if(verbose):
            print(cur_sod)
        ite += 1
    return median, cur_sod, sods, set_median
    
    draw_Letter_graph(median)


def compute_median_set(script,listID):
    'Returns the id in listID corresponding to median set'
    #Calcul median set
    N=len(listID)
    map_id_to_index = {}
    map_index_to_id = {}
    for i in range(0,len(listID)):
        map_id_to_index[listID[i]] = i
        map_index_to_id[i] = listID[i]
        
    distances = np.zeros((N,N))
    for i in listID:
        for j in listID:
            script.PyRunMethod(i,j)
            distances[map_id_to_index[i],map_id_to_index[j]] = script.PyGetUpperBound(i,j)

    median_set_index = np.argmin(np.sum(distances,0))
    sod = np.min(np.sum(distances,0))
    
    return median_set_index, sod

#if __name__ == "__main__":
#    #Chargement du dataset
#    script.PyLoadGXLGraph('/home/bgauzere/dev/gedlib/data/datasets/Letter/HIGH/', '/home/bgauzere/dev/gedlib/data/collections/Letter_Z.xml')
#    script.PySetEditCost("LETTER")
#    script.PyInitEnv()
#    script.PySetMethod("IPFP", "")
#    script.PyInitMethod()
#
#    dataset,my_y = pygraph.utils.graphfiles.loadDataset("/home/bgauzere/dev/gedlib/data/datasets/Letter/HIGH/Letter_Z.cxl")
#
#    listID = script.PyGetAllGraphIds()
#    median, sod = compute_median(script,listID,dataset,verbose=True)
#    
#    print(sod)
#    draw_Letter_graph(median)


if __name__ == '__main__':
    # test draw_Letter_graph
    ds = {'name': 'Letter-high', 'dataset': '../datasets/Letter-high/Letter-high_A.txt',
          'extra_params': {}} # node nsymb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    print(y_all)
    for g in Gn:
        draw_Letter_graph(g)