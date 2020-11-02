#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:35:32 2018

@author: ljia
"""

#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
# import pickle
import os
import sys
from tqdm import tqdm
# from mpl_toolkits.mplot3d import Axes3D


root_dir = '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/gklearn/experiments/ged/stability/outputs/'

root_dir_criann = '/media/ljia/DATA/research-repo/codes/Linlin/graphkit-learn/gklearn/experiments/ged/stability/outputs/CRIANN/'

Dataset_List = ['MAO', 'Monoterpenoides', 'MUTAG', 'AIDS_symb']

Legend_Labels = ['common walk', 'marginalized', 'Sylvester equation', 'conjugate gradient', 'fixed-point iterations', 'Spectral decomposition', 'shortest path', 'structural sp', 'path up to length $h$', 'treelet', 'WL subtree']

# Colors = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', 
#               '#54278f', '#756bb1', '#9e9ac8', '#de2d26', '#fc9272']
Colors=[
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5']

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


def read_trials_group(save_dir, ds_name, num_sols, ratio, label):
	file_name = save_dir + 'groups/ged_mats.' + ds_name + '.' + label + '_' + str(num_sols) + '.ratio_' + "{:.2f}".format(ratio) + '.npy'
	if os.path.isfile(file_name):
		with open(file_name, 'rb') as f:
			ged_mats = np.load(f)
			return ged_mats
	else:
		return []
			
#	ged_mats = []
#	for trial in range(1, 101):
#		file_name = file_prefix + '.trial_' + str(trial) + '.pkl'
#		if os.path.isfile(file_name):
#			ged_matrix = pickle.load(open(file_name, 'rb'))
#			ged_mats.append(ged_matrix)
#		else:
# #			print(trial)
#			pass	
		
		
# Check average relative error along elements in two ged matrices.
def matrices_ave_relative_error(m1, m2):
    error = 0
    base = 0
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            error += np.abs(m1[i, j] - m2[i, j])
            base += (np.abs(m1[i, j]) + np.abs(m2[i, j])) / 2
                
    return error / base


def compute_relative_error(ged_mats):
		
	if len(ged_mats) != 0:
		# get the smallest "correct" GED matrix.
		ged_mat_s = np.ones(ged_mats[0].shape) * np.inf
		for i in range(ged_mats[0].shape[0]):
			for j in range(ged_mats[0].shape[1]):
				ged_mat_s[i, j] = np.min([mat[i, j] for mat in ged_mats])
		
		# compute average error.
		errors = []
		for i, mat in enumerate(ged_mats):
			err = matrices_ave_relative_error(mat, ged_mat_s)
	    #             if not per_correct:
	    #                 print('matrix # ', str(i))
	    #                 pass
			errors.append(err)
	else:
		errors = [0]
		
	return np.mean(errors)
		
			 


#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

#fig, _ = plt.subplots(2, 2, figsize=(13, 12))
#ax1 = plt.subplot(221)
#ax2 = plt.subplot(222)
#ax3 = plt.subplot(223)
#ax4 = plt.subplot(224)
gs = gridspec.GridSpec(2, 2)
gs.update(hspace=0.3)
fig = plt.figure(figsize=(11, 12))
ax = fig.add_subplot(111)    # The big subplot for common labels
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ax3 = fig.add_subplot(gs[1, 0], projection='3d')
ax4 = fig.add_subplot(gs[1, 1], projection='3d')
# ax5 = fig.add_subplot(gs[2, 0])
# ax6 = fig.add_subplot(gs[2, 1])

# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# Set common labels
#ax.set_xlabel('accuracy(%)')
ax.yaxis.set_label_coords(-0.105, 0.5)
# ax.set_ylabel('runtime($s$)')


# -------------- num_sols, IPFP --------------
def get_num_sol_results():
	save_dir = root_dir_criann + 'edit_costs.num_sols.ratios.IPFP/'
	errors = {}
	print('-------- num_sols, IPFP --------')
	for ds_name in Dataset_List:
		print(ds_name)
		errors[ds_name] = []
		for num_sols in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			errors[ds_name].append([])
			for ratio in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], desc='num_sols = ' + str(num_sols), file=sys.stdout):
				ged_mats = read_trials_group(save_dir, ds_name, num_sols, ratio, 'num_sols')
				error = compute_relative_error(ged_mats)
				errors[ds_name][-1].append(error)

	return errors
	
x_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y_values = range(0, 19)
X, Y = np.meshgrid(x_values, y_values)
errors = get_num_sol_results()
for i, ds_name in enumerate(Dataset_List):
	if ds_name in errors:
		z_values = np.array(errors[ds_name])
		ax1.plot_wireframe(X, Y, z_values.T, label=Dataset_List[i], color=Colors[i]) #, '.-', label=Legend_Labels[i], color=Colors[i])

# ax1.set_yscale('squareroot')
# ax1.grid(axis='y')
ax1.set_xlabel('# of solutions')
ax1.set_ylabel('ratios')
ax1.set_zlabel('average relative errors (%)')
ax1.set_title('(a) num_sols, IPFP')
ax1.set_yticks(range(0, 19, 2))
ax1.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9, 2, 4, 6, 8, 10])
# ax1.set_axisbelow(True)
# ax1.spines['top'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['left'].set_visible(False)
# ax1.xaxis.set_ticks_position('none')
# ax1.yaxis.set_ticks_position('none')
# ax1.set_ylim(bottom=-1000)
handles, labels = ax1.get_legend_handles_labels()



# # -------------- repeats, IPFP --------------
def get_repeats_results():
	save_dir = root_dir_criann + 'edit_costs.repeats.ratios.IPFP/'
	errors = {}
	print('-------- repeats, IPFP --------')
	for ds_name in Dataset_List:
		print(ds_name)
		errors[ds_name] = []
		for num_sols in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			errors[ds_name].append([])
			for ratio in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], desc='num_sols = ' + str(num_sols), file=sys.stdout):
				ged_mats = read_trials_group(save_dir, ds_name, num_sols, ratio, 'repeats')
				error = compute_relative_error(ged_mats)
				errors[ds_name][-1].append(error)

	return errors
	
x_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y_values = range(0, 19)
X, Y = np.meshgrid(x_values, y_values)
errors = get_repeats_results()
for i, ds_name in enumerate(Dataset_List):
	if ds_name in errors:
		z_values = np.array(errors[ds_name])
		ax2.plot_wireframe(X, Y, z_values.T, label=Dataset_List[i], color=Colors[i]) #, '.-', label=Legend_Labels[i], color=Colors[i])

# ax2.set_yscale('squareroot')
# ax2.grid(axis='y')
ax2.set_xlabel('# of solutions')
ax2.set_ylabel('ratios')
ax2.set_zlabel('average relative errors (%)')
ax2.set_title('(b) repeats, IPFP')
ax2.set_yticks(range(0, 19, 2))
ax2.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9, 2, 4, 6, 8, 10])
# ax2.set_axisbelow(True)
# ax2.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.xaxis.set_ticks_position('none')
# ax2.yaxis.set_ticks_position('none')
# ax2.set_ylim(bottom=-1000)
handles, labels = ax2.get_legend_handles_labels()


# # -------------- degrees --------------
# def get_degree_results():
#	save_dir = root_dir_criann + '28 cores/synthesized_graphs_degrees/'
#	run_times = {}
#	for kernel_name in Graph_Kernel_List:
#		run_times[kernel_name] = []
#		for num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#			file_name = save_dir + 'run_time.' + kernel_name + '.' + str(num) + '.pkl'
#			if os.path.isfile(file_name):
#				run_time = pickle.load(open(file_name, 'rb'))
#			else:
#				run_time = 0
#			run_times[kernel_name].append(run_time)
#	return run_times

# x_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# run_times = get_degree_results()
# for i, kernel_name in enumerate(Graph_Kernel_List):
#	if kernel_name in run_times:
#		ax3.plot(x_labels, run_times[kernel_name], '.-', label=Legend_Labels[i], color=Colors[i])

# ax3.set_yscale('log', nonposy='clip')
# ax3.grid(axis='y')
# ax3.set_xlabel('degrees')
# ax3.set_ylabel('runtime($s$)')
# #ax3.set_ylabel('runtime($s$) per pair of graphs')
# ax3.set_title('(c) degrees')
# ax3.set_axisbelow(True)
# ax3.spines['top'].set_visible(False)
# ax3.spines['bottom'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['left'].set_visible(False)
# ax3.xaxis.set_ticks_position('none')
# ax3.yaxis.set_ticks_position('none')


# # -------------- Node labels --------------
# def get_node_label_results():
#	save_dir = root_dir_criann + '28 cores/synthesized_graphs_num_node_label_alphabet/'
#	run_times = {}
#	for kernel_name in Graph_Kernel_List_VSym:
#		run_times[kernel_name] = []
#		for num in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
#			file_name = save_dir + 'run_time.' + kernel_name + '.' + str(num) + '.pkl'
#			if os.path.isfile(file_name):
#				run_time = pickle.load(open(file_name, 'rb'))
#			else:
#				run_time = 0
#			run_times[kernel_name].append(run_time)
#	return run_times

# #	save_dir = root_dir_criann + 'synthesized_graphs_num_node_label_alphabet/'
# #	run_times = pickle.load(open(save_dir + 'run_times.pkl', 'rb'))
# #	return run_times

# x_labels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# run_times = get_node_label_results()
# for i, kernel_name in enumerate(Graph_Kernel_List):
#	if kernel_name in run_times:
#		ax4.plot(x_labels[1:], run_times[kernel_name][1:], '.-', label=Legend_Labels[i], color=Colors[i])

# ax4.set_yscale('log', nonposy='clip')
# ax4.grid(axis='y')
# ax4.set_xlabel('# of alphabets')
# ax4.set_ylabel('runtime($s$)')
# #ax4.set_ylabel('runtime($s$) per pair of graphs')
# ax4.set_title('(d) alphabet size of vertex labels')
# ax4.set_axisbelow(True)
# ax4.spines['top'].set_visible(False)
# ax4.spines['bottom'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['left'].set_visible(False)
# ax4.xaxis.set_ticks_position('none')
# ax4.yaxis.set_ticks_position('none')


from matplotlib.lines import Line2D
custom_lines = []
for color in Colors:
	custom_lines.append(Line2D([0], [0], color=color, lw=4))

fig.subplots_adjust(bottom=0.135)
fig.legend(custom_lines, labels, loc='lower center', ncol=4, frameon=False) # , ncol=5, labelspacing=0.1, handletextpad=0.4, columnspacing=0.6)
plt.savefig('stability.real_data.relative_error.eps', format='eps', dpi=300, transparent=True,
            bbox_inches='tight')
plt.show()