#export LD_LIBRARY_PATH=.:/export/home/lambertn/Documents/gedlibpy/lib/fann/:/export/home/lambertn/Documents/gedlibpy/lib/libsvm.3.22:/export/home/lambertn/Documents/gedlibpy/lib/nomad

#Pour que "import script" trouve les librairies qu'a besoin GedLib
#Equivalent à définir la variable d'environnement LD_LIBRARY_PATH sur un bash
import librariesImport
import gedlibpy
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
    
init()

def afficheMatrix(mat) :
    for i in mat :
        line = ""
        for j in i :
            line+=str(j)
            line+=" "
        print(line)

def createNxGraph() :
    G = nx.Graph()
    G.add_node("1", chem = "C")
    G.add_node("2", chem = "0")
    G.add_edge("1", "2", valence = "1")
    G.add_node("3", chem = "N")
    G.add_node("4", chem = "C")
    G.add_edge("3", "4", valence = "1")
    G.add_edge("3", "2", valence = "1")
    return G

#G = createNxGraph()

def addGraphTest() :
    gedlibpy.restart_env()
    gedlibpy.load_GXL_graphs('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')

    currentID = gedlibpy.add_graph()
    print(currentID)
    
    gedlibpy.add_node(currentID, "_1", {"chem" : "C"})
    gedlibpy.add_node(currentID, "_2", {"chem" : "O"})
    gedlibpy.add_edge(currentID,"_1", "_2",  {"valence": "1"} )

    listID = gedlibpy.get_all_graph_ids()
    print(listID)
    print(gedlibpy.get_graph_node_labels(10))
    print(gedlibpy.get_graph_edges(10))
    
    for i in listID : 
        print(gedlibpy.get_graph_node_labels(i))
        print(gedlibpy.get_graph_edges(i))

#addGraphTest()

def shortTest() :
    gedlibpy.restart_env()
    
    print("Here is the mini Python function !")
    
    gedlibpy.load_GXL_graphs("include/gedlib-master/data/datasets/Mutagenicity/data/", "include/gedlib-master/data/collections/Mutagenicity.xml")
    listID = gedlibpy.get_all_graph_ids()
    gedlibpy.set_edit_cost("CHEM_1")

    gedlibpy.init()

    gedlibpy.set_method("BIPARTITE", "")
    gedlibpy.init_method()

    g = listID[0]
    h = listID[1]

    gedlibpy.run_method(g,h)

    print("Node Map : ", gedlibpy.get_node_map(g,h))
    print("Assignment Matrix : ")
    afficheMatrix(gedlibpy.get_assignment_matrix(g,h))
    print ("Upper Bound = " + str(gedlibpy.get_upper_bound(g,h)) + ", Lower Bound = " + str(gedlibpy.get_lower_bound(g,h)) + ", Runtime = " + str(gedlibpy.get_runtime(g,h)))

#shortTest()

def classiqueTest() :
    gedlibpy.restart_env()
    
    gedlibpy.load_GXL_graphs('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')
    listID = gedlibpy.get_all_graph_ids()
    
    afficheId = ""
    for i in listID :
        afficheId+=str(i) + " "
    print("Number of graphs = " + str(len(listID)) + ", list of Ids = " + afficheId)

    gedlibpy.set_edit_cost("CHEM_1")

    gedlibpy.init()

    gedlibpy.set_method("IPFP", "")
    gedlibpy.init_method()

    g = listID[0]
    h = listID[0]

    gedlibpy.run_method(g,h)
    liste = gedlibpy.get_all_map(g,h)
    print("Forward map : " , gedlibpy.get_forward_map(g,h), ", Backward map : ", gedlibpy.get_backward_map(g,h))
    print("Node Map : ", gedlibpy.get_node_map(g,h))
    print ("Upper Bound = " + str(gedlibpy.get_upper_bound(g,h)) + ", Lower Bound = " + str(gedlibpy.get_lower_bound(g,h)) + ", Runtime = " + str(gedlibpy.get_runtime(g,h)))

#classiqueTest()

def nxTest(dataset) :
    gedlibpy.restart_env()
    
    for graph in dataset :
        gedlibpy.add_nx_graph(graph, "")
        
    listID = gedlibpy.get_all_graph_ids()
    gedlibpy.set_edit_cost("CHEM_1")
    gedlibpy.init()
    gedlibpy.set_method("IPFP", "")
    gedlibpy.init_method()

    print(listID)
    g = listID[0]
    h = listID[1]

    gedlibpy.run_method(g,h)

    print("Node Map : ", gedlibpy.get_node_map(g,h))
    print ("Upper Bound = " + str(gedlibpy.get_upper_bound(g,h)) + ", Lower Bound = " + str(gedlibpy.get_lower_bound(g,h)) + ", Runtime = " + str(gedlibpy.get_runtime(g,h)))

#dataset = [createNxGraph(), createNxGraph()]
#nxTest(dataset)

def LSAPETest(matrixCost) :
    result = gedlibpy.hungarian_LSAPE(matrixCost)
    print("Rho = ", result[0], " Varrho = ", result[1], " u = ", result[2], " v = ", result[3])

#LSAPETest([[2,3,4], [5,1,9], [7,10,3]])

def LSAPTest(matrixCost) :
    result = gedlibpy.hungarian_LSAP(matrixCost)
    print("Rho = ", result[0], " Varrho = ", result[1], " u = ", result[2], " v = ", result[3])

#LSAPETest([[2,3,4], [5,1,9], [7,10,3]])
