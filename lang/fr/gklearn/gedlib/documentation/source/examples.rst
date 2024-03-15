Examples
==============

Before using each example, please make sure to put these lines on the beginnig of your code : 

.. code-block:: python 

  import librariesImport
  import gedlibpy

Use your path to access it, without changing the library architecture. After that, you are ready to use the library. 

When you want to make new computation, please use this function : 

.. code-block:: python 

  gedlibpy.restart_env()

All the graphs and results will be delete so make sure you don't need it. 

Classique case with GXL graphs
------------------------------------
.. code-block:: python 

  gedlibpy.load_GXL_graphs('include/gedlib-master/data/datasets/Mutagenicity/data/', 'collections/MUTA_10.xml')
  listID = gedlibpy.get_all_graph_ids()
  gedlibpy.set_edit_cost("CHEM_1")

  gedlibpy.init()

  gedlibpy.set_method("IPFP", "")
  gedlibpy.init_method()

  g = listID[0]
  h = listID[1]

  gedlibpy.run_method(g,h)

  print("Node Map : ", gedlibpy.get_node_map(g,h))
  print ("Upper Bound = " + str(gedlibpy.get_upper_bound(g,h)) + ", Lower Bound = " + str(gedlibpy.get_lower_bound(g,h)) + ", Runtime = " + str(gedlibpy.get_runtime(g,h)))


You can also use this function :

.. code-block:: python 

  compute_edit_distance_on_GXl_graphs(path_folder, path_XML, edit_cost, method, options="", init_option = "EAGER_WITHOUT_SHUFFLED_COPIES")
    
This function compute all edit distance between all graphs, even itself. You can see the result with some functions and graphs IDs. Please see the documentation of the function for more informations. 

Classique case with NX graphs
------------------------------------
.. code-block:: python 

  for graph in dataset :
    gedlibpy.add_nx_graph(graph, classe)
  listID = gedlibpy.get_all_graph_ids()
  gedlibpy.set_edit_cost("CHEM_1")

  gedlibpy.init()  

  gedlibpy.set_method("IPFP", "")
  gedlibpy.init_method()

  g = listID[0]
  h = listID[1]

  gedlibpy.run_method(g,h)

  print("Node Map : ", gedlibpy.get_node_map(g,h))
  print ("Upper Bound = " + str(gedlibpy.get_upper_bound(g,h)) + ", Lower Bound = " + str(gedlibpy.get_lower_bound(g,h)) + ", Runtime = " + str(gedlibpy.get_runtime(g,h)))

You can also use this function :

.. code-block:: python 

  compute_edit_distance_on_nx_graphs(dataset, classes, edit_cost, method, options, init_option = "EAGER_WITHOUT_SHUFFLED_COPIES")
    
This function compute all edit distance between all graphs, even itself. You can see the result in the return and with some functions and graphs IDs. Please see the documentation of the function for more informations. 

Or this function : 

.. code-block:: python 

  compute_ged_on_two_graphs(g1,g2, edit_cost, method, options, init_option = "EAGER_WITHOUT_SHUFFLED_COPIES")

This function allow to compute the edit distance just for two graphs. Please see the documentation of the function for more informations. 

Add a graph from scratch
------------------------------------
.. code-block:: python 

  currentID = gedlibpy.add_graph()
  gedlibpy.add_node(currentID, "_1", {"chem" : "C"})
  gedlibpy.add_node(currentID, "_2", {"chem" : "O"})
  gedlibpy.add_edge(currentID,"_1", "_2",  {"valence": "1"} )

Please make sure as the type are the same (string for Ids and a dictionnary for labels). If you want a symmetrical graph, you can use this function to ensure the symmetry : 

.. code-block:: python 

  add_symmetrical_edge(graph_id, tail, head, edge_label) 

If you have a Nx structure, you can use directly this function : 

.. code-block:: python 

  add_nx_graph(g, classe, ignore_duplicates=True)

Even if you have another structure, you can use this function : 

.. code-block:: python
 
  add_random_graph(name, classe, list_of_nodes, list_of_edges, ignore_duplicates=True)

Please read the documentation before using and respect the types.

Median computation
------------------------------------

An example is available in the Median_Example folder. It contains the necessary to compute a median graph. You can launch xp-letter-gbr.py to compute median graph on all letters in the dataset, or median.py for le letter Z. 

To summarize the use, you can follow this example : 

.. code-block:: python
 
  import pygraph #Available with the median example
  from median import draw_Letter_graph, compute_median, compute_median_set

  gedlibpy.load_GXL_graphs('../include/gedlib-master/data/datasets/Letter/HIGH/', '../include/gedlib-master/data/collections/Letter_Z.xml')
  gedlibpy.set_edit_cost("LETTER")
  gedlibpy.init()
  gedlibpy.set_method("IPFP", "")
  gedlibpy.init_method()
  listID = gedlibpy.get_all_graph_ids()

  dataset,my_y = pygraph.utils.graphfiles.loadDataset("../include/gedlib-master/data/datasets/Letter/HIGH/Letter_Z.cxl")
  median, sod, sods_path,set_median = compute_median(gedlibpy,listID,dataset,verbose=True)
  draw_Letter_graph(median)

Please use the function in the median.py code to simplify your use. You can adapt this example to your case. Also, some function in the PythonGedLib module can make the work easier. Ask Benoît Gauzere if you want more information.     

Hungarian algorithm
------------------------------------


LSAPE
~~~~~~

.. code-block:: python

  result = gedlibpy.hungarian_LSAPE(matrixCost)
  print("Rho = ", result[0], " Varrho = ", result[1], " u = ", result[2], " v = ", result[3])


LSAP
~~~~~~

.. code-block:: python

  result = gedlibpy.hungarian_LSAP(matrixCost)
  print("Rho = ", result[0], " Varrho = ", result[1], " u = ", result[2], " v = ", result[3])



