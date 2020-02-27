#export LD_LIBRARY_PATH=.:/export/home/lambertn/Documents/gedlibpy/lib/fann/:/export/home/lambertn/Documents/gedlibpy/lib/libsvm.3.22:/export/home/lambertn/Documents/gedlibpy/lib/nomad

#Pour que "import script" trouve les librairies qu'a besoin GedLib
#Equivalent à définir la variable d'environnement LD_LIBRARY_PATH sur un bash
#import gedlibpy_linlin.librariesImport
#from  gedlibpy_linlin import gedlibpy
from libs import *
import networkx as nx
import numpy as np
from tqdm import tqdm
import sys


def test_NON_SYMBOLIC_cost():
    """Test edit cost LETTER2.
    """
    import sys
    sys.path.insert(0, "../")
    from preimage.ged import GED, get_nb_edit_operations_nonsymbolic, get_nb_edit_operations_letter
    from preimage.test_k_closest_graphs import reform_attributes
    from gklearn.utils.graphfiles import loadDataset

    dataset = '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/datasets/Letter-high/Letter-high_A.txt'
    Gn, y_all = loadDataset(dataset)

    g1 = Gn[200]
    g2 = Gn[1780]
    reform_attributes(g1)
    reform_attributes(g2)

    c_vi = 0.675
    c_vr = 0.675
    c_vs = 0.75
    c_ei = 0.425
    c_er = 0.425
    c_es = 0

    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
    dis, pi_forward, pi_backward = GED(g1, g2, lib='gedlibpy',
        cost='NON_SYMBOLIC', method='IPFP', edit_cost_constant=edit_cost_constant,
        algo_options='', stabilizer=None)
    n_vi, n_vr, sod_vs, n_ei, n_er, sod_es = get_nb_edit_operations_nonsymbolic(g1, g2,
        pi_forward, pi_backward)

    print('# of operations:', n_vi, n_vr, sod_vs, n_ei, n_er, sod_es)
    print('c_vi, c_vr, c_vs, c_ei, c_er:', c_vi, c_vr, c_vs, c_ei, c_er, c_es)
    cost_computed = c_vi * n_vi + c_vr * n_vr + c_vs * sod_vs \
        + c_ei * n_ei + c_er * n_er + c_es * sod_es
    print('dis (cost computed by GED):', dis)
    print('cost computed by # of operations and edit cost constants:', cost_computed)


def test_LETTER2_cost():
    """Test edit cost LETTER2.
    """
    import sys
    sys.path.insert(0, "../")
    from preimage.ged import GED, get_nb_edit_operations_letter
    from preimage.test_k_closest_graphs import reform_attributes
    from gklearn.utils.graphfiles import loadDataset

    ds = {'dataset': '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/collections/Letter.xml',
          'graph_dir': '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/datasets/Letter/HIGH/'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['graph_dir'])

    g1 = Gn[200]
    g2 = Gn[1780]
    reform_attributes(g1)
    reform_attributes(g2)

    c_vi = 0.675
    c_vr = 0.675
    c_vs = 0.75
    c_ei = 0.425
    c_er = 0.425

    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er]
    dis, pi_forward, pi_backward = GED(g1, g2, dataset='letter', lib='gedlibpy',
        cost='LETTER2', method='IPFP', edit_cost_constant=edit_cost_constant,
        algo_options='', stabilizer=None)
    n_vi, n_vr, n_vs, sod_vs, n_ei, n_er = get_nb_edit_operations_letter(g1, g2,
        pi_forward, pi_backward)

    print('# of operations:', n_vi, n_vr, n_vs, sod_vs, n_ei, n_er)
    print('c_vi, c_vr, c_vs, c_ei, c_er:', c_vi, c_vr, c_vs, c_ei, c_er)
    cost_computed = c_vi * n_vi + c_vr * n_vr + c_vs * sod_vs \
        + c_ei * n_ei + c_er * n_er
    print('dis (cost computed by GED):', dis)
    print('cost computed by # of operations and edit cost constants:', cost_computed)



def test_get_nb_edit_operations_letter():
    """Test whether function preimage.ged.get_nb_edit_operations_letter returns
    correct numbers of edit operations. The distance/cost computed by GED
    should be the same as the cost computed by number of operations and edit
    cost constants.
    """
    import sys
    sys.path.insert(0, "../")
    from preimage.ged import GED, get_nb_edit_operations_letter
    from preimage.test_k_closest_graphs import reform_attributes
    from gklearn.utils.graphfiles import loadDataset

    ds = {'dataset': '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/collections/Letter.xml',
          'graph_dir': '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/data/datasets/Letter/HIGH/'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'], extra_params=ds['graph_dir'])

    g1 = Gn[200]
    g2 = Gn[1780]
    reform_attributes(g1)
    reform_attributes(g2)

    c_vir = 0.9
    c_eir = 1.7
    alpha = 0.75

    edit_cost_constant = [c_vir, c_eir, alpha]
    dis, pi_forward, pi_backward = GED(g1, g2, dataset='letter', lib='gedlibpy',
        cost='LETTER', method='IPFP', edit_cost_constant=edit_cost_constant,
        algo_options='', stabilizer=None)
    n_vi, n_vr, n_vs, c_vs, n_ei, n_er = get_nb_edit_operations_letter(g1, g2,
        pi_forward, pi_backward)

    print('# of operations and costs:', n_vi, n_vr, n_vs, c_vs, n_ei, n_er)
    print('c_vir, c_eir, alpha:', c_vir, c_eir, alpha)
    cost_computed = alpha * c_vir * (n_vi + n_vr) \
        + alpha * c_vs \
        + (1 - alpha) * c_eir * (n_ei + n_er)
    print('dis (cost computed by GED):', dis)
    print('cost computed by # of operations and edit cost constants:', cost_computed)


def test_get_nb_edit_operations():
    """Test whether function preimage.ged.get_nb_edit_operations returns correct
    numbers of edit operations. The distance/cost computed by GED should be the
    same as the cost computed by number of operations and edit cost constants.
    """
    import sys
    sys.path.insert(0, "../")
    from preimage.ged import GED, get_nb_edit_operations
    from gklearn.utils.graphfiles import loadDataset

    ds = {'dataset': '../datasets/monoterpenoides/dataset_10+.ds',
          'graph_dir': '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/datasets/monoterpenoides/'}  # node/edge symb
    Gn, y_all = loadDataset(ds['dataset'])

    g1 = Gn[20]
    g2 = Gn[108]

    c_vi = 3
    c_vr = 3
    c_vs = 1
    c_ei = 3
    c_er = 3
    c_es = 1

    edit_cost_constant = [c_vi, c_vr, c_vs, c_ei, c_er, c_es]
    dis, pi_forward, pi_backward = GED(g1, g2, dataset='monoterpenoides', lib='gedlibpy',
        cost='CONSTANT', method='IPFP', edit_cost_constant=edit_cost_constant,
        algo_options='', stabilizer=None)
    n_vi, n_vr, n_vs, n_ei, n_er, n_es = get_nb_edit_operations(g1, g2,
        pi_forward, pi_backward)

    print('# of operations and costs:', n_vi, n_vr, n_vs, n_ei, n_er, n_es)
    print('edit costs:', c_vi, c_vr, c_vs, c_ei, c_er, c_es)
    cost_computed = n_vi * c_vi + n_vr * c_vr + n_vs * c_vs \
        + n_ei * c_ei + n_er * c_er + n_es * c_es
    print('dis (cost computed by GED):', dis)
    print('cost computed by # of operations and edit cost constants:', cost_computed)


def test_ged_python_bash_cpp():
    """Test ged computation with python invoking the c++ code by bash command (with updated library).
    """
    sys.path.insert(0, "../")
    from gklearn.utils.graphfiles import loadDataset
    from preimage.ged import GED

    data_dir_prefix = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/'
#    collection_file = data_dir_prefix + 'generated_datsets/monoterpenoides/gxl/monoterpenoides.xml'
    collection_file = data_dir_prefix + 'generated_datsets/monoterpenoides/monoterpenoides_3_20.xml'
    graph_dir = data_dir_prefix +'generated_datsets/monoterpenoides/gxl/'

    Gn, y = loadDataset(collection_file, extra_params=graph_dir)

    algo_options = '--threads 8 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'

    for repeat in range(0, 3):
        # Generate the result file.
        ged_filename = data_dir_prefix + 'output/test_ged/ged_mat_python_bash_' + str(repeat) + '_init40.3_20.txt'
#        runtime_filename = data_dir_prefix + 'output/test_ged/runtime_mat_python_min_' + str(repeat) + '.txt'

        ged_file = open(ged_filename, 'a')
#        runtime_file = open(runtime_filename, 'a')

        ged_mat = np.empty((len(Gn), len(Gn)))
#        runtime_mat = np.empty((len(Gn), len(Gn)))

        for i in tqdm(range(len(Gn)), desc='computing GEDs', file=sys.stdout):
            for j in range(len(Gn)):
                print(i, j)
                g1 = Gn[i]
                g2 = Gn[j]
                upper_bound, _, _ = GED(g1, g2, lib='gedlib-bash', cost='CONSTANT',
                                method='IPFP',
                                edit_cost_constant=[3.0, 3.0, 1.0, 3.0, 3.0, 1.0],
                                algo_options=algo_options)
#                runtime = gedlibpy.get_runtime(g1, g2)
                ged_mat[i][j] = upper_bound
#                runtime_mat[i][j] = runtime

                # Write to files.
                ged_file.write(str(int(upper_bound)) + ' ')
#                runtime_file.write(str(runtime) + ' ')

            ged_file.write('\n')
#            runtime_file.write('\n')

        ged_file.close()
#        runtime_file.close()

    print('ged_mat')
    print(ged_mat)
#    print('runtime_mat:')
#    print(runtime_mat)

    return



def test_ged_best_settings_updated():
    """Test ged computation with best settings the same as in the C++ code (with updated library).
    """

    data_dir_prefix = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/'
    collection_file = data_dir_prefix + 'generated_datsets/monoterpenoides/gxl/monoterpenoides.xml'
#    collection_file = data_dir_prefix + 'generated_datsets/monoterpenoides/monoterpenoides_3_20.xml'

    graph_dir = data_dir_prefix +'generated_datsets/monoterpenoides/gxl/'

    algo_options = '--threads 8 --initial-solutions 40 --ratio-runs-from-initial-solutions 1'

    for repeat in range(0, 3):
        # Generate the result file.
        ged_filename = data_dir_prefix + 'output/test_ged/ged_mat_python_updated_' + str(repeat) + '_init40.txt'
        runtime_filename = data_dir_prefix + 'output/test_ged/runtime_mat_python_updated_' + str(repeat) + '_init40.txt'

        gedlibpy.restart_env()
        gedlibpy.load_GXL_graphs(graph_dir, collection_file)
        listID = gedlibpy.get_all_graph_ids()
        gedlibpy.set_edit_cost('CONSTANT', [3.0, 3.0, 1.0, 3.0, 3.0, 1.0])
        gedlibpy.init()
        gedlibpy.set_method("IPFP", algo_options)
        gedlibpy.init_method()

        ged_mat = np.empty((len(listID), len(listID)))
        runtime_mat = np.empty((len(listID), len(listID)))

        for i in tqdm(range(len(listID)), desc='computing GEDs', file=sys.stdout):
            ged_file = open(ged_filename, 'a')
            runtime_file = open(runtime_filename, 'a')

            for j in range(len(listID)):
                g1 = listID[i]
                g2 = listID[j]
                gedlibpy.run_method(g1, g2)
                upper_bound = gedlibpy.get_upper_bound(g1, g2)
                runtime = gedlibpy.get_runtime(g1, g2)
                ged_mat[i][j] = upper_bound
                runtime_mat[i][j] = runtime

                # Write to files.
                ged_file.write(str(int(upper_bound)) + ' ')
                runtime_file.write(str(runtime) + ' ')

            ged_file.write('\n')
            runtime_file.write('\n')

            ged_file.close()
            runtime_file.close()

    print('ged_mat')
    print(ged_mat)
    print('runtime_mat:')
    print(runtime_mat)

    return


def test_ged_best_settings():
    """Test ged computation with best settings the same as in the C++ code.
    """

    data_dir_prefix = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/'
    collection_file = data_dir_prefix + 'generated_datsets/monoterpenoides/gxl/monoterpenoides.xml'
    graph_dir = data_dir_prefix +'generated_datsets/monoterpenoides/gxl/'

    algo_options = '--threads 6 --initial-solutions 10 --ratio-runs-from-initial-solutions .5'

    for repeat in range(0, 3):
        # Generate the result file.
        ged_filename = data_dir_prefix + 'output/test_ged/ged_mat_python_best_settings_' + str(repeat) + '.txt'
        runtime_filename = data_dir_prefix + 'output/test_ged/runtime_mat_python_best_settings_' + str(repeat) + '.txt'

        ged_file = open(ged_filename, 'a')
        runtime_file = open(runtime_filename, 'a')

        gedlibpy.restart_env()
        gedlibpy.load_GXL_graphs(graph_dir, collection_file)
        listID = gedlibpy.get_all_graph_ids()
        gedlibpy.set_edit_cost('CONSTANT', [3.0, 3.0, 1.0, 3.0, 3.0, 1.0])
        gedlibpy.init()
        gedlibpy.set_method("IPFP", algo_options)
        gedlibpy.init_method()

        ged_mat = np.empty((len(listID), len(listID)))
        runtime_mat = np.empty((len(listID), len(listID)))

        for i in tqdm(range(len(listID)), desc='computing GEDs', file=sys.stdout):
            for j in range(len(listID)):
                g1 = listID[i]
                g2 = listID[j]
                gedlibpy.run_method(g1, g2)
                upper_bound = gedlibpy.get_upper_bound(g1, g2)
                runtime = gedlibpy.get_runtime(g1, g2)
                ged_mat[i][j] = upper_bound
                runtime_mat[i][j] = runtime

                # Write to files.
                ged_file.write(str(int(upper_bound)) + ' ')
                runtime_file.write(str(runtime) + ' ')

            ged_file.write('\n')
            runtime_file.write('\n')

        ged_file.close()
        runtime_file.close()

    print('ged_mat')
    print(ged_mat)
    print('runtime_mat:')
    print(runtime_mat)

    return



def test_ged_default():
    """Test ged computation with default settings.
    """

    data_dir_prefix = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/'
    collection_file = data_dir_prefix + 'generated_datsets/monoterpenoides/gxl/monoterpenoides.xml'
    graph_dir = data_dir_prefix +'generated_datsets/monoterpenoides/gxl/'

    for repeat in range(3):
        # Generate the result file.
        ged_filename = data_dir_prefix + 'output/test_ged/ged_mat_python_default_' + str(repeat) + '.txt'
        runtime_filename = data_dir_prefix + 'output/test_ged/runtime_mat_python_default_' + str(repeat) + '.txt'

        ged_file = open(ged_filename, 'a')
        runtime_file = open(runtime_filename, 'a')

        gedlibpy.restart_env()
        gedlibpy.load_GXL_graphs(graph_dir, collection_file)
        listID = gedlibpy.get_all_graph_ids()
        gedlibpy.set_edit_cost('CONSTANT', [3.0, 3.0, 1.0, 3.0, 3.0, 1.0])
        gedlibpy.init()
        gedlibpy.set_method("IPFP", "")
        gedlibpy.init_method()

        ged_mat = np.empty((len(listID), len(listID)))
        runtime_mat = np.empty((len(listID), len(listID)))

        for i in tqdm(range(len(listID)), desc='computing GEDs', file=sys.stdout):
            for j in range(len(listID)):
                g1 = listID[i]
                g2 = listID[j]
                gedlibpy.run_method(g1, g2)
                upper_bound = gedlibpy.get_upper_bound(g1, g2)
                runtime = gedlibpy.get_runtime(g1, g2)
                ged_mat[i][j] = upper_bound
                runtime_mat[i][j] = runtime

                # Write to files.
                ged_file.write(str(int(upper_bound)) + ' ')
                runtime_file.write(str(runtime) + ' ')

            ged_file.write('\n')
            runtime_file.write('\n')

        ged_file.close()
        runtime_file.close()

    print('ged_mat')
    print(ged_mat)
    print('runtime_mat:')
    print(runtime_mat)

    return


def test_ged_min():
    """Test ged computation with the "min" stabilizer.
    """
    sys.path.insert(0, "../")
    from gklearn.utils.graphfiles import loadDataset
    from preimage.ged import GED

    data_dir_prefix = '/media/ljia/DATA/research-repo/codes/others/gedlib/tests_linlin/'
    collection_file = data_dir_prefix + 'generated_datsets/monoterpenoides/gxl/monoterpenoides.xml'
    graph_dir = data_dir_prefix +'generated_datsets/monoterpenoides/gxl/'

    Gn, y = loadDataset(collection_file, extra_params=graph_dir)

#    algo_options = '--threads 6 --initial-solutions 10 --ratio-runs-from-initial-solutions .5'

    for repeat in range(0, 3):
        # Generate the result file.
        ged_filename = data_dir_prefix + 'output/test_ged/ged_mat_python_min_' + str(repeat) + '.txt'
#        runtime_filename = data_dir_prefix + 'output/test_ged/runtime_mat_python_min_' + str(repeat) + '.txt'

        ged_file = open(ged_filename, 'a')
#        runtime_file = open(runtime_filename, 'a')

        ged_mat = np.empty((len(Gn), len(Gn)))
#        runtime_mat = np.empty((len(Gn), len(Gn)))

        for i in tqdm(range(len(Gn)), desc='computing GEDs', file=sys.stdout):
            for j in range(len(Gn)):
                g1 = Gn[i]
                g2 = Gn[j]
                upper_bound, _, _ = GED(g1, g2, lib='gedlibpy', cost='CONSTANT',
                                method='IPFP',
                                edit_cost_constant=[3.0, 3.0, 1.0, 3.0, 3.0, 1.0],
                                stabilizer='min', repeat=10)
#                runtime = gedlibpy.get_runtime(g1, g2)
                ged_mat[i][j] = upper_bound
#                runtime_mat[i][j] = runtime

                # Write to files.
                ged_file.write(str(int(upper_bound)) + ' ')
#                runtime_file.write(str(runtime) + ' ')

            ged_file.write('\n')
#            runtime_file.write('\n')

        ged_file.close()
#        runtime_file.close()

    print('ged_mat')
    print(ged_mat)
#    print('runtime_mat:')
#    print(runtime_mat)

    return


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

if __name__ == '__main__':
#    test_ged_default()
#    test_ged_min()
#    test_ged_best_settings()
#    test_ged_best_settings_updated()
#    test_ged_python_bash_cpp()
#    test_get_nb_edit_operations()
#    test_get_nb_edit_operations_letter()
#    test_LETTER2_cost()
    test_NON_SYMBOLIC_cost()


    #init()
    #testNxGrapĥ()
