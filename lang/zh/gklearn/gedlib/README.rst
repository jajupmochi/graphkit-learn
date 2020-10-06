GEDLIBPY
====================================

Please Read https://dbblumenthal.github.io/gedlib/ before using Python code.
You can also find this module documentation in documentation/build/html folder. 

Make sure you have numpy installed (and Cython if you have to recompile the library). You can use pip for this. 


Running the script
-------------------

After donwloading the entire folder, you can run test.py to ensure the library works. 

For your code, you have to make two imports::

  import librariesImport
  import gedlibpy

You can call each function in the library with this. You can't move any folder or files on the library, please make sure that the architecture remains the same. 

This library is compiled for Python3 only. If you want to use it with Python 2, you have to recompile it with setup.py. You have to use this command on your favorite shell::

  python setup.py build_ext --inplace

After this step, you can use the same lines as Python3 for import, it will be ok. Check the documentation inside the documentation/build/html folder before using function. You can also copy the tests examples for basic use.


A problem with the library ? 
-------------------------------

If the library isn't found, you can recompile the Python library because your Linux is different to mine. Please delete gedlibpy.so, gedlibpy.cpp and build folder. Then use this command on a linux shell ::

  python3 setup.py build_ext --inplace

You can make it with Python 2 but make sure you use the same version with your code and the compilation.

If it's doesn't work, maybe the version of GedLib or another library can be a problem. If it is, you can re-install GedLib for your computer. You can download it on this git : https://dbblumenthal.github.io/gedlib/

You have to install Gedlib with the Python installer after that. 
Just call::

  python3 install.py

Make the links like indicate on the documentation. Use the same architecture like this library, but just change the .so and folders with your installation. You can recompile the Python library with setup command, after that. 

If you use Mac OS, you have to follow all this part, and install the external libraries with this command::

  install_name_tool -change <mylib> <path>/<to>/<mylib> <myexec>

For an example, you have to write these lines::

  install_name_tool -change libdoublefann.2.dylib lib/fann/libdoublefann.2.dylib gedlibpy.so
  install_name_tool -change libsvm.so lib/libsvm.3.22/libsvm.so gedlibpy.so
  install_name_tool -change libnomad.so lib/nomad/libnomad.so gedlibpy.so
  install_name_tool -change libsgtelib.so lib/nomad/libsgtelib.so gedlibpy.so

The name of the library gedlibpy can be different if you use Python 3.

If your problem is still here, you can contact me on : natacha.lambert@unicaen.fr

How to use this library
-------------------------

This library allow to compute edit distance between two graphs. You have to follow these steps to use it : 

- Add your graphs (GXL files, NX Structures or your structure, make sure that the internal type is the same)
- Choose your cost function 
- Init your environnment (After that, the cost function and your graphs can't be modified)
- Choose your method computation
- Run the computation with the IDs of the two graphs. You can have the ID when you add the graph or with some functions
- Find the result with differents functions (NodeMap, edit distance, etc)

Here is an example of code with GXL graphs::

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


Please read the documentation for more examples and functions. 


An advice if you don't code in a shell
---------------------------------------

Python library don't indicate each C++ error. If you have a restart causing by an error in your code, please use on a linux shell for having C++ errors. 
