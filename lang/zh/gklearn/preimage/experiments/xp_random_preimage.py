#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:39:29 2020

@author: ljia
"""
import multiprocessing
import functools
import sys
import os
import logging
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from gklearn.preimage import generate_random_preimages_by_class
from gklearn.utils import compute_gram_matrices_by_class


dir_root = '../results/xp_random_preimage/'


def xp_median_preimage_15_1():
	"""xp 15_1: AIDS, StructuralSP, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'AIDS' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/'
	irrelevant_labels = {'node_attrs': ['chem', 'charge', 'x', 'y'], 'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
				

def xp_median_preimage_15_2():
	"""xp 15_2: AIDS, PathUpToH, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'AIDS' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 1, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/' 
	irrelevant_labels = {'node_attrs': ['chem', 'charge', 'x', 'y'], 'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
 
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_15_3():
	"""xp 15_3: AIDS, Treelet, using CONSTANT, symbolic only.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'AIDS' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	pkernel = functools.partial(polynomialkernel, d=1, c=1e+2)
	kernel_options = {'name': 'Treelet', #
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/' 
	irrelevant_labels = {'node_attrs': ['chem', 'charge', 'x', 'y'], 'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_15_4():
	"""xp 15_4: AIDS, WeisfeilerLehman, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'AIDS' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 10,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/' 
	irrelevant_labels = {'node_attrs': ['chem', 'charge', 'x', 'y'], 'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
# 	# compute gram matrices for each class a priori.
# 	print('Compute gram matrices for each class a priori.')
# 	compute_gram_matrices_by_class(ds_name, kernel_options, save_results=True, dir_save=dir_save, irrelevant_labels=irrelevant_labels)
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))


def xp_median_preimage_14_1():
	"""xp 14_1: DD, PathUpToH, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'DD' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 2, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = None #
	edge_required = False #

	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output
	
# 	# compute gram matrices for each class a priori.
# 	print('Compute gram matrices for each class a priori.')
# 	compute_gram_matrices_by_class(ds_name, kernel_options, save_results=save_results, dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))


def xp_median_preimage_12_1():
	"""xp 12_1: PAH, StructuralSP, using NON_SYMBOLIC, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.unlabeled/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
		
		
def xp_median_preimage_12_2():
	"""xp 12_2: PAH, PathUpToH, using CONSTANT, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 1, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.unlabeled/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
		

def xp_median_preimage_12_3():
	"""xp 12_3: PAH, Treelet, using CONSTANT, unlabeled.
	"""
	from gklearn.utils.kernels import gaussiankernel
	# set parameters.
	ds_name = 'PAH' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	pkernel = functools.partial(gaussiankernel, gamma=None) # @todo
	kernel_options = {'name': 'Treelet', #
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.unlabeled/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = False #

	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
		
		
def xp_median_preimage_12_4():
	"""xp 12_4: PAH, WeisfeilerLehman, using CONSTANT, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 14,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.unlabeled/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
# 	# compute gram matrices for each class a priori.
# 	print('Compute gram matrices for each class a priori.')
# 	compute_gram_matrices_by_class(ds_name, kernel_options, save_results=True, dir_save=dir_save, irrelevant_labels=irrelevant_labels)
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
		
		
def xp_median_preimage_12_5():
	"""xp 12_5: PAH, ShortestPath, using NON_SYMBOLIC, unlabeled.
	"""
	# set parameters.
	ds_name = 'PAH' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'ShortestPath',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.unlabeled/'  # 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']} # 
	edge_required = True #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))


def xp_median_preimage_9_1():
	"""xp 9_1: MAO, StructuralSP, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'MAO' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_type', 'bond_stereo']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
		
			
def xp_median_preimage_9_2():
	"""xp 9_2: MAO, PathUpToH, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'MAO' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 9, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_type', 'bond_stereo']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
		
			
def xp_median_preimage_9_3():
	"""xp 9_3: MAO, Treelet, using CONSTANT, symbolic only.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'MAO' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	pkernel = functools.partial(polynomialkernel, d=4, c=1e+7)
	kernel_options = {'name': 'Treelet', #
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_type', 'bond_stereo']} # 
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
		
			
def xp_median_preimage_9_4():
	"""xp 9_4: MAO, WeisfeilerLehman, using CONSTANT, symbolic only.
	"""
	# set parameters.
	ds_name = 'MAO' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 6,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '.symb/' 
	irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_type', 'bond_stereo']} # 
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
# 	# compute gram matrices for each class a priori.
# 	print('Compute gram matrices for each class a priori.')
# 	compute_gram_matrices_by_class(ds_name, kernel_options, save_results=True, dir_save=dir_save, irrelevant_labels=irrelevant_labels)
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))


def xp_median_preimage_8_1():
	"""xp 8_1: Monoterpenoides, StructuralSP, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'Monoterpenoides' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_8_2():
	"""xp 8_2: Monoterpenoides, PathUpToH, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'Monoterpenoides' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 7, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_8_3():
	"""xp 8_3: Monoterpenoides, Treelet, using CONSTANT.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'Monoterpenoides' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 0}
	pkernel = functools.partial(polynomialkernel, d=2, c=1e+5)
	kernel_options = {'name': 'Treelet',
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_8_4():
	"""xp 8_4: Monoterpenoides, WeisfeilerLehman, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'Monoterpenoides' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 4,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['valence']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			

def xp_median_preimage_7_1():
	"""xp 7_1: MUTAG, StructuralSP, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'MUTAG' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
	sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
	kernel_options = {'name': 'StructuralSP',
					  'edge_weight': None,
					  'node_kernels': sub_kernels,
					  'edge_kernels': sub_kernels, 
					  'compute_method': 'naive',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['label_0']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_7_2():
	"""xp 7_2: MUTAG, PathUpToH, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'MUTAG' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'PathUpToH',
					  'depth': 2, #
					  'k_func': 'MinMax', #
					  'compute_method': 'trie',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['label_0']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output
	
	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required, cut_range=None)
	except Exception as exp:
		print('An exception occured when running experiment on xp_median_preimage_7_2:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_7_3():
	"""xp 7_3: MUTAG, Treelet, using CONSTANT.
	"""
	from gklearn.utils.kernels import polynomialkernel
	# set parameters.
	ds_name = 'MUTAG' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	pkernel = functools.partial(polynomialkernel, d=3, c=1e+8)
	kernel_options = {'name': 'Treelet',
				      'sub_kernel': pkernel,
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['label_0']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
			
			
def xp_median_preimage_7_4():
	"""xp 7_4: MUTAG, WeisfeilerLehman, using CONSTANT.
	"""
	# set parameters.
	ds_name = 'MUTAG' #
	rpg_options = {'k': 5,
				   'r_max': 10, #
				   'l': 500,
				   'alphas': None,
				   'parallel': True,
				   'verbose': 2}
	kernel_options = {'name': 'WeisfeilerLehman',
				      'height': 1,
					  'base_kernel': 'subtree',
 					  'parallel': 'imap_unordered', 
                      # 'parallel': None, 
					  'n_jobs': multiprocessing.cpu_count(),
					  'normalize': True,
					  'verbose': 0}
	save_results = True
	dir_save = dir_root + ds_name + '.' + kernel_options['name'] + '/' 
	irrelevant_labels = {'edge_labels': ['label_0']} #
	edge_required = False #
	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	file_output = open(dir_save + 'output.txt', 'a')
	sys.stdout = file_output

	# print settings.
	print('parameters:')
	print('dataset name:', ds_name)
	print('kernel_options:', kernel_options)
	print('save_results:', save_results)
	print('irrelevant_labels:', irrelevant_labels)
	print()
	
	# generate preimages.
	try:
		generate_random_preimages_by_class(ds_name, rpg_options, kernel_options, save_results=save_results, save_preimages=True, load_gm='auto', dir_save=dir_save, irrelevant_labels=irrelevant_labels, edge_required=edge_required)
	except Exception as exp:
		print('An exception occured when running this experiment:')
		LOG_FILENAME = dir_save + 'error.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
		logging.exception('')
		print(repr(exp))
				
				
if __name__ == "__main__":

# 	#### xp 7_2: MUTAG, PathUpToH, using CONSTANT.
 	xp_median_preimage_7_2()

# 	#### xp 7_3: MUTAG, Treelet, using CONSTANT.
 	xp_median_preimage_7_3()

# 	#### xp 7_4: MUTAG, WeisfeilerLehman, using CONSTANT.
 	xp_median_preimage_7_4()
# 	 
# 	#### xp 7_1: MUTAG, StructuralSP, using CONSTANT.
 	xp_median_preimage_7_1()

# 	#### xp 8_2: Monoterpenoides, PathUpToH, using CONSTANT.
 	xp_median_preimage_8_2()

# 	#### xp 8_3: Monoterpenoides, Treelet, using CONSTANT.
 	xp_median_preimage_8_3()

# 	#### xp 8_4: Monoterpenoides, WeisfeilerLehman, using CONSTANT.
 	xp_median_preimage_8_4()
	 
#     #### xp 8_1: Monoterpenoides, StructuralSP, using CONSTANT.
 	xp_median_preimage_8_1()

# 	#### xp 9_2: MAO, PathUpToH, using CONSTANT, symbolic only.
 	xp_median_preimage_9_2()

# 	#### xp 9_3: MAO, Treelet, using CONSTANT, symbolic only.
 	xp_median_preimage_9_3()

# 	#### xp 9_4: MAO, WeisfeilerLehman, using CONSTANT, symbolic only.
 	xp_median_preimage_9_4()
	 
# 	#### xp 9_1: MAO, StructuralSP, using CONSTANT, symbolic only.
 	xp_median_preimage_9_1()

 	#### xp 12_1: PAH, StructuralSP, using NON_SYMBOLIC, unlabeled.
 	xp_median_preimage_12_1()

 	#### xp 12_2: PAH, PathUpToH, using CONSTANT, unlabeled.
 	xp_median_preimage_12_2()

 	#### xp 12_3: PAH, Treelet, using CONSTANT, unlabeled.
 	xp_median_preimage_12_3()

 	#### xp 12_4: PAH, WeisfeilerLehman, using CONSTANT, unlabeled.
 	xp_median_preimage_12_4()

 	#### xp 12_5: PAH, ShortestPath, using NON_SYMBOLIC, unlabeled.
 	xp_median_preimage_12_5()
	 
	 # 	#### xp 15_1: AIDS, StructuralSP, using CONSTANT, symbolic only.
 	xp_median_preimage_15_1()

# 	#### xp 15_2: AIDS, PathUpToH, using CONSTANT, symbolic only.
 	xp_median_preimage_15_2()

# 	#### xp 15_3: AIDS, Treelet, using CONSTANT, symbolic only.
 	xp_median_preimage_15_3()

# 	#### xp 15_4: AIDS, WeisfeilerLehman, using CONSTANT, symbolic only.
 	xp_median_preimage_15_4()
# 	
 	#### xp 14_1: DD, PathUpToH, using CONSTANT.
 	xp_median_preimage_14_1()
