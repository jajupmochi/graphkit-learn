#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:17:36 2020

@author: ljia
"""
from gklearn.utils import Dataset


def get_dataset(ds_name):
	# The node/edge labels that will not be used in the computation.
	if ds_name == 'MAO':
		irrelevant_labels = {'node_attrs': ['x', 'y', 'z'], 'edge_labels': ['bond_stereo']}
	elif ds_name == 'Monoterpenoides':
		irrelevant_labels = {'edge_labels': ['valence']}
	elif ds_name == 'MUTAG':
		irrelevant_labels = {'edge_labels': ['label_0']}
	elif ds_name == 'AIDS_symb':
		irrelevant_labels = {'node_attrs': ['chem', 'charge', 'x', 'y'], 'edge_labels': ['valence']}
		ds_name = 'AIDS'

	# Initialize a Dataset.
	dataset = Dataset()
	# Load predefined dataset.
	dataset.load_predefined_dataset(ds_name)
	# Remove irrelevant labels.
	dataset.remove_labels(**irrelevant_labels)
	print('dataset size:', len(dataset.graphs))
	return dataset