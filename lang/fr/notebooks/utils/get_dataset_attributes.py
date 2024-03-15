#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:07:38 2018

@author: ljia
"""

import sys
sys.path.insert(0, "../../")
from gklearn.utils.graphfiles import loadDataset
from gklearn.utils.graphdataset import get_dataset_attributes

dslist = [
    {'name': 'Acyclic', 'dataset': '../../datasets/acyclic/dataset_bps.ds',},
    {'name': 'Alkane', 'dataset': '../../datasets/Alkane/dataset.ds',
        'dataset_y': '../../datasets/Alkane/dataset_boiling_point_names.txt',},
    {'name': 'MAO', 'dataset': '../../datasets/MAO/dataset.ds',},
    {'name': 'PAH', 'dataset': '../../datasets/PAH/dataset.ds',},
    {'name': 'MUTAG', 'dataset': '../../datasets/MUTAG/MUTAG.mat',
        'extra_params': {'am_sp_al_nl_el': [0, 0, 3, 1, 2]}},
    {'name': 'Letter-med', 'dataset': '../../datasets/Letter-med/Letter-med_A.txt'},
    {'name': 'ENZYMES', 'dataset': '../../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'},
    {'name': 'Mutagenicity', 'dataset': '../../datasets/Mutagenicity/Mutagenicity_A.txt'},
    {'name': 'D&D', 'dataset': '../../datasets/D&D/DD.mat',
     'extra_params': {'am_sp_al_nl_el': [0, 1, 2, 1, -1]}},
    {'name': 'AIDS', 'dataset': '../../datasets/AIDS/AIDS_A.txt'},
    {'name': 'FIRSTMM_DB', 'dataset': '../../datasets/FIRSTMM_DB/FIRSTMM_DB_A.txt'},
    {'name': 'MSRC9', 'dataset': '../../datasets/MSRC_9_txt/MSRC_9_A.txt'},
    {'name': 'MSRC21', 'dataset': '../../datasets/MSRC_21_txt/MSRC_21_A.txt'},
    {'name': 'SYNTHETIC', 'dataset': '../../datasets/SYNTHETIC_txt/SYNTHETIC_A_sparse.txt'},
    {'name': 'BZR', 'dataset': '../../datasets/BZR_txt/BZR_A_sparse.txt'},
    {'name': 'COX2', 'dataset': '../../datasets/COX2_txt/COX2_A_sparse.txt'},
    {'name': 'DHFR', 'dataset': '../../datasets/DHFR_txt/DHFR_A_sparse.txt'},    
    {'name': 'PROTEINS', 'dataset': '../../datasets/PROTEINS_txt/PROTEINS_A_sparse.txt'},
    {'name': 'PROTEINS_full', 'dataset': '../../datasets/PROTEINS_full_txt/PROTEINS_full_A_sparse.txt'},   
    {'name': 'NCI1', 'dataset': '../../datasets/NCI1/NCI1.mat',
        'extra_params': {'am_sp_al_nl_el': [1, 1, 2, 0, -1]}},
    {'name': 'NCI109', 'dataset': '../../datasets/NCI109/NCI109.mat',
        'extra_params': {'am_sp_al_nl_el': [1, 1, 2, 0, -1]}},
    {'name': 'NCI-HIV', 'dataset': '../../datasets/NCI-HIV/AIDO99SD.sdf',
        'dataset_y': '../../datasets/NCI-HIV/aids_conc_may04.txt',},

#     # not working below
#     {'name': 'PTC_FM', 'dataset': '../../datasets/PTC/Train/FM.ds',},
#     {'name': 'PTC_FR', 'dataset': '../../datasets/PTC/Train/FR.ds',},
#     {'name': 'PTC_MM', 'dataset': '../../datasets/PTC/Train/MM.ds',},
#     {'name': 'PTC_MR', 'dataset': '../../datasets/PTC/Train/MR.ds',},
]

for ds in dslist:
    dataset, y = loadDataset(
        ds['dataset'],
        filename_y=(ds['dataset_y'] if 'dataset_y' in ds else None),
        extra_params=(ds['extra_params'] if 'extra_params' in ds else None))
    attrs = get_dataset_attributes(
        dataset, target=y, node_label='atom', edge_label='bond_type')
    print()
    print(ds['name'] + ':')
    for atr in attrs:
        print(atr, ':', attrs[atr])
    print()