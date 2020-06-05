#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:37:57 2020

@author: ljia
"""
import multiprocessing
import numpy as np
import networkx as nx
import os
from gklearn.utils.graphfiles import saveGXL
from gklearn.preimage import RandomPreimageGenerator
from gklearn.utils import Dataset


dir_root = '../results/xp_random_preimage_generation/'


def xp_random_preimage_generation(kernel_name):
	"""
	Experiment similar to the one in Bakir's paper. A test to check if RandomPreimageGenerator class works correctly.

	Returns
	-------
	None.

	"""
	alpha1_list = np.linspace(0, 1, 11)
	k_dis_datasets = []
	k_dis_preimages = []
	preimages = []
	bests_from_dataset = []
	for alpha1 in alpha1_list:
		print('alpha1 =', alpha1, ':\n')
		# set parameters.
		ds_name = 'MUTAG'
		rpg_options = {'k': 5,
					   'r_max': 10, #
					   'l': 500,
					   'alphas': None,
					   'parallel': True,
					   'verbose': 2}
		if kernel_name == 'PathUpToH':
			kernel_options = {'name': 'PathUpToH',
							  'depth': 2, #
							  'k_func': 'MinMax', #
							  'compute_method': 'trie',
		 					  'parallel': 'imap_unordered', 
		                      # 'parallel': None, 
							  'n_jobs': multiprocessing.cpu_count(),
							  'normalize': True,
							  'verbose': 0}
		elif kernel_name == 'Marginalized':
			kernel_options = {'name': 'Marginalized',
							  'p_quit': 0.8, #
							  'n_iteration': 7, #
							  'remove_totters': False,
		 					  'parallel': 'imap_unordered', 
		                      # 'parallel': None, 
							  'n_jobs': multiprocessing.cpu_count(),
							  'normalize': True,
							  'verbose': 0}
		edge_required = True
		irrelevant_labels = {'edge_labels': ['label_0']}
		cut_range = None
		
		# create/get Gram matrix.
		dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/'
		if not os.path.exists(dir_save):
			os.makedirs(dir_save)
		gm_fname = dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm.npz'
		gmfile_exist = os.path.isfile(os.path.abspath(gm_fname))
		if gmfile_exist:
			gmfile = np.load(gm_fname, allow_pickle=True) # @todo: may not be safe.
			gram_matrix_unnorm = gmfile['gram_matrix_unnorm']
			time_precompute_gm = gmfile['run_time']
	
		# 1. get dataset.
		print('1. getting dataset...')
		dataset_all = Dataset()
		dataset_all.load_predefined_dataset(ds_name)
		dataset_all.trim_dataset(edge_required=edge_required)
		if irrelevant_labels is not None:
			dataset_all.remove_labels(**irrelevant_labels)
		if cut_range is not None:
			dataset_all.cut_graphs(cut_range)
		
# 		# add two "random" graphs.
# 		g1 = nx.Graph()
# 		g1.add_nodes_from(range(0, 16), label_0='0')
# 		g1.add_nodes_from(range(16, 25), label_0='1')
# 		g1.add_node(25, label_0='2')
# 		g1.add_nodes_from([26, 27], label_0='3')
# 		g1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (5, 0), (4, 9), (12, 3), (10, 13), (13, 14), (14, 15), (15, 8), (0, 16), (1, 17), (2, 18), (12, 19), (11, 20), (13, 21), (15, 22), (7, 23), (6, 24), (14, 25), (25, 26), (25, 27)])
# 		g2 = nx.Graph()
# 		g2.add_nodes_from(range(0, 12), label_0='0')
# 		g2.add_nodes_from(range(12, 19), label_0='1')
# 		g2.add_nodes_from([19, 20, 21], label_0='2')
# 		g2.add_nodes_from([22, 23], label_0='3')
# 		g2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 19), (19, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 20), (20, 7), (5, 0), (4, 8), (0, 12), (1, 13), (2, 14), (9, 15), (10, 16), (11, 17), (6, 18), (3, 21), (21, 22), (21, 23)])
# 		dataset_all.load_graphs([g1, g2] + dataset_all.graphs, targets=None)
		
		# 2. initialize rpg and setting parameters.
		print('2. initializing rpg and setting parameters...')
# 		nb_graphs = len(dataset_all.graphs) - 2
# 		rpg_options['alphas'] = [alpha1, 1 - alpha1] + [0] * nb_graphs
		nb_graphs = len(dataset_all.graphs)
		alphas = [0] * nb_graphs
		alphas[1] = alpha1
		alphas[6] = 1 - alpha1
		rpg_options['alphas'] = alphas
		if gmfile_exist:
			rpg_options['gram_matrix_unnorm'] = gram_matrix_unnorm
			rpg_options['runtime_precompute_gm'] = time_precompute_gm
		rpg = RandomPreimageGenerator()
		rpg.dataset = dataset_all
		rpg.set_options(**rpg_options.copy())
		rpg.kernel_options = kernel_options.copy()
	
		# 3. compute preimage.
		print('3. computing preimage...')
		rpg.run()
		results = rpg.get_results()
		k_dis_datasets.append(results['k_dis_dataset'])
		k_dis_preimages.append(results['k_dis_preimage'])
		bests_from_dataset.append(rpg.best_from_dataset)
		preimages.append(rpg.preimage)
		
		# 4. save results.
		# write Gram matrices to file.
		if not gmfile_exist:
			np.savez(dir_save + 'gram_matrix_unnorm.' + ds_name + '.' + kernel_options['name'] + '.gm', gram_matrix_unnorm=rpg.gram_matrix_unnorm, run_time=results['runtime_precompute_gm'])
		
		# save graphs.
		fn_best_dataset = dir_save + 'g_best_dataset.' + 'alpha1_' + str(alpha1)[0:3]
		saveGXL(rpg.best_from_dataset, fn_best_dataset + '.gxl', method='default', 
			  node_labels=dataset_all.node_labels, edge_labels=dataset_all.edge_labels,	
			  node_attrs=dataset_all.node_attrs, edge_attrs=dataset_all.edge_attrs)
		fn_preimage = dir_save + 'g_preimage.' + 'alpha1_' + str(alpha1)[0:3]
		saveGXL(rpg.preimage, fn_preimage + '.gxl', method='default', 
			  node_labels=dataset_all.node_labels, edge_labels=dataset_all.edge_labels,	
			  node_attrs=dataset_all.node_attrs, edge_attrs=dataset_all.edge_attrs)
		
		# draw graphs.
		__draw_graph(rpg.best_from_dataset, fn_best_dataset)
		__draw_graph(rpg.preimage, fn_preimage)
		
	# plot results figure.
	__plot_results(alpha1_list, k_dis_datasets, k_dis_preimages, dir_save)
		
	print('\ncomplete.\n')
	
	return k_dis_datasets, k_dis_preimages, bests_from_dataset, preimages


def __draw_graph(graph, file_prefix):
# 	import matplotlib
# 	matplotlib.use('agg')
	import matplotlib.pyplot as plt
	plt.figure()
	pos = nx.spring_layout(graph)
	nx.draw(graph, pos, node_size=500, labels=nx.get_node_attributes(graph, 'label_0'), font_color='w', width=3, with_labels=True)
	plt.savefig(file_prefix + '.eps', format='eps', dpi=300)
#	plt.show()
	plt.clf()
	plt.close()


def __plot_results(alpha1_list, k_dis_datasets, k_dis_preimages, dir_save):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

	ind = np.arange(len(alpha1_list))    # the x locations for the groups
	width = 0.35       # the width of the bars: can also be len(x) sequence
	
	p1 = ax.bar(ind, k_dis_preimages, width, label='Reconstructed pre-image', zorder=3, color='#133AAC')
	
	ax.set_xlabel(r'$\alpha \in [0,1]$')
	ax.set_ylabel(r'$d(g_i,g^\star(\alpha))$')
	#ax.set_title('Runtime of the shortest path kernel on all datasets')
	plt.xticks(ind, [str(i)[0:3] for i in alpha1_list])
	#ax.set_yticks(np.logspace(-16, -3, num=20, base=10))
	#ax.set_ylim(bottom=1e-15)
	ax.grid(axis='y', zorder=0)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.xaxis.set_ticks_position('none')

	p2 = ax.plot(ind, k_dis_datasets, 'b.-', label=r'Nearest neighbor in $D_N$', color='orange', zorder=4)
	ax.yaxis.set_ticks_position('none')
	
	fig.subplots_adjust(bottom=.2)
	fig.legend(loc='lower center', ncol=2, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
	
	plt.savefig(dir_save + 'distances in kernel space.eps', format='eps', dpi=300,
	            transparent=True, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()	
	

if __name__ == '__main__':
# 	kernel_name = 'PathUpToH'
	kernel_name = 'Marginalized'
	k_dis_datasets, k_dis_preimages, bests_from_dataset, preimages = xp_random_preimage_generation(kernel_name)
	
# 	# save graphs.	
# 	dir_save = dir_root + 'MUTAG.PathUpToH/'
# 	for i, alpha1 in enumerate(np.linspace(0, 1, 11)):
# 		fn_best_dataset = dir_save + 'g_best_dataset.' + 'alpha1_' + str(alpha1)[0:3]
# 		saveGXL(bests_from_dataset[i], fn_best_dataset + '.gxl', method='default', 
# 			  node_labels=['label_0'], edge_labels=[],	
# 			  node_attrs=[], edge_attrs=[])
# 		fn_preimage = dir_save + 'g_preimage.' + 'alpha1_' + str(alpha1)[0:3]
# 		saveGXL(preimages[i], fn_preimage + '.gxl', method='default', 
# 			  node_labels=['label_0'], edge_labels=[],	
# 			  node_attrs=[], edge_attrs=[])

# 	# draw graphs.
# 	dir_save = dir_root + 'MUTAG.PathUpToH/'
# 	for i, alpha1 in enumerate(np.linspace(0, 1, 11)):
# 		fn_best_dataset = dir_save + 'g_best_dataset.' + 'alpha1_' + str(alpha1)[0:3]
# 		__draw_graph(bests_from_dataset[i], fn_best_dataset)
# 		fn_preimage = dir_save + 'g_preimage.' + 'alpha1_' + str(alpha1)[0:3]
# 		__draw_graph(preimages[i], fn_preimage)

# 	# plot results figure.
# 	alpha1_list = np.linspace(0, 1, 11)
# 	dir_save = dir_root + 'MUTAG.PathUpToH/'
# 	__plot_results(alpha1_list, k_dis_datasets, k_dis_preimages, dir_save)

		
		
# k_dis_datasets = [0.0,
#  0.08882515554098754,
#  0.17765031108197632,
#  0.2664754666229643,
#  0.35530062216395264,
#  0.44412577770494066,
#  0.35530062216395236,
#  0.2664754666229643,
#  0.17765031108197632,
#  0.08882515554098878,
#  0.0]

# k_dis_preimages = [0.0,
#  0.08882515554098754,
#  0.17765031108197632,
#  0.2664754666229643,
#  0.35530062216395264,
#  0.44412577770494066,
#  0.35530062216395236,
#  0.2664754666229643,
#  0.17765031108197632,
#  0.08882515554098878,
#  0.0]