#export LD_LIBRARY_PATH=.:/export/home/lambertn/Documents/gedlibpy/lib/fann/:/export/home/lambertn/Documents/gedlibpy/lib/libsvm.3.22:/export/home/lambertn/Documents/gedlibpy/lib/nomad

#Pour que "import script" trouve les librairies qu'a besoin GedLib
#Equivalent à définir la variable d'environnement LD_LIBRARY_PATH sur un bash
import gedlibpy.librariesImport
from  gedlibpy import gedlibpy
import networkx as nx


def init() :
    print("List of Edit Cost Options : ")
    for i in gedlibpy.list_of_edit_cost_options :
        print (i)
    print("")

    print("List of Method Options : ")
    for j in gedlibpy.list_of_method_options :
        print (j)
    print("")

    print("List of Init Options : ")
    for k in gedlibpy.list_of_init_options :
        print (k)
    print("")
    
def test():
    
    gedlibpy.load_GXL_graphs('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')
    listID = gedlibpy.get_all_graph_ids()
    gedlibpy.set_edit_cost("CHEM_1")
    gedlibpy.init()
    gedlibpy.set_method("IPFP", "")
    gedlibpy.init_method()
    g = listID[0]
    h = listID[1]
    gedlibpy.run_method(g, h)
    print("Node Map : ", gedlibpy.get_node_map(g,h))
    print("Forward map : " , gedlibpy.get_forward_map(g, h), ", Backward map : ", gedlibpy.get_backward_map(g, h))
    print("Assignment Matrix : ")
    print(gedlibpy.get_assignment_matrix(g, h))
    print ("Upper Bound = " + str(gedlibpy.get_upper_bound(g,h)) + ", Lower Bound = " + str(gedlibpy.get_lower_bound(g, h)) + ", Runtime = " + str(gedlibpy.get_runtime(g, h)))


def convertGraph(G):
    G_new = nx.Graph()
    for nd, attrs in G.nodes(data=True):
        G_new.add_node(str(nd), chem=attrs['atom'])
    for nd1, nd2, attrs in G.edges(data=True):
        G_new.add_edge(str(nd1), str(nd2), valence=attrs['bond_type'])
        
    return G_new


def testNxGrapĥ():
    import sys
    sys.path.insert(0, "../")
    from gklearn.utils.graphfiles import loadDataset
    ds = {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt',
          'extra_params': {}}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['extra_params'])
    
    gedlibpy.restart_env()
    for graph in Gn:
        g_new = convertGraph(graph)
        gedlibpy.add_nx_graph(g_new, "")
        
    listID = gedlibpy.get_all_graph_ids()
    gedlibpy.set_edit_cost("CHEM_1")
    gedlibpy.init()
    gedlibpy.set_method("IPFP", "")
    gedlibpy.init_method()

    print(listID)
    g = listID[0]
    h = listID[1]

    gedlibpy.run_method(g, h)

    print("Node Map : ", gedlibpy.get_node_map(g, h))
    print("Forward map : " , gedlibpy.get_forward_map(g, h), ", Backward map : ", gedlibpy.get_backward_map(g, h))
    print ("Upper Bound = " + str(gedlibpy.get_upper_bound(g, h)) + ", Lower Bound = " + str(gedlibpy.get_lower_bound(g, h)) + ", Runtime = " + str(gedlibpy.get_runtime(g, h)))

#test()
init() 
#testNxGrapĥ()
