dslist = [
    {
        'name': 'Acyclic',
        'dataset': '../datasets/acyclic/dataset_bps.ds',
        'task': 'regression'
    },  # node symb
    #     {'name': 'COIL-DEL', 'dataset': '../datasets/COIL-DEL/COIL-DEL_A.txt'}, # edge symb, node nsymb
    {
        'name': 'PAH',
        'dataset': '../datasets/PAH/dataset.ds',
    },  # unlabeled
    {
        'name': 'MAO',
        'dataset': '../datasets/MAO/dataset.ds',
    },  # node/edge symb
    {
        'name': 'MUTAG',
        'dataset': '../datasets/MUTAG/MUTAG.mat',
        'extra_params': {
            'am_sp_al_nl_el': [0, 0, 3, 1, 2]
        }
    },  # node/edge symb
    {
        'name': 'Alkane',
        'dataset': '../datasets/Alkane/dataset.ds',
        'task': 'regression',
        'dataset_y': '../datasets/Alkane/dataset_boiling_point_names.txt',
    },  # contains single node graph, node symb
    #     {'name': 'BZR', 'dataset': '../datasets/BZR_txt/BZR_A_sparse.txt'}, # node symb/nsymb
    #     {'name': 'COX2', 'dataset': '../datasets/COX2_txt/COX2_A_sparse.txt'}, # node symb/nsymb
    {
        'name': 'Mutagenicity',
        'dataset': '../datasets/Mutagenicity/Mutagenicity_A.txt'
    },  # node/edge symb
    {
        'name': 'ENZYMES',
        'dataset': '../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'
    },  # node symb/nsymb
    #     {'name': 'Fingerprint', 'dataset': '../datasets/Fingerprint/Fingerprint_A.txt'},
    {
        'name': 'Letter-med',
        'dataset': '../datasets/Letter-med/Letter-med_A.txt'
    },
    #     {'name': 'DHFR', 'dataset': '../datasets/DHFR_txt/DHFR_A_sparse.txt'}, # node symb/nsymb
    #     {'name': 'SYNTHETIC', 'dataset': '../datasets/SYNTHETIC_txt/SYNTHETIC_A_sparse.txt'}, # node symb/nsymb
    #     {'name': 'MSRC9', 'dataset': '../datasets/MSRC_9_txt/MSRC_9_A.txt'}, # node symb
    #     {'name': 'MSRC21', 'dataset': '../datasets/MSRC_21_txt/MSRC_21_A.txt'}, # node symb
    #     {'name': 'FIRSTMM_DB', 'dataset': '../datasets/FIRSTMM_DB/FIRSTMM_DB_A.txt'}, # node symb/nsymb ,edge nsymb

    #     {'name': 'PROTEINS', 'dataset': '../datasets/PROTEINS_txt/PROTEINS_A_sparse.txt'}, # node symb/nsymb
    #     {'name': 'PROTEINS_full', 'dataset': '../datasets/PROTEINS_full_txt/PROTEINS_full_A_sparse.txt'}, # node symb/nsymb
    {
        'name': 'D&D',
        'dataset': '../datasets/D&D/DD.mat',
        'extra_params': {
            'am_sp_al_nl_el': [0, 1, 2, 1, -1]
        }
    },  # node symb
    #     {'name': 'AIDS', 'dataset': '../datasets/AIDS/AIDS_A.txt'}, # node symb/nsymb, edge symb
    #     {'name': 'NCI1', 'dataset': '../datasets/NCI1/NCI1.mat',
    #         'extra_params': {'am_sp_al_nl_el': [1, 1, 2, 0, -1]}}, # node symb
    #     {'name': 'NCI109', 'dataset': '../datasets/NCI109/NCI109.mat',
    #         'extra_params': {'am_sp_al_nl_el': [1, 1, 2, 0, -1]}}, # node symb
    #     {'name': 'NCI-HIV', 'dataset': '../datasets/NCI-HIV/AIDO99SD.sdf',
    #         'dataset_y': '../datasets/NCI-HIV/aids_conc_may04.txt',}, # node/edge symb

    #     # not working below
    #     {'name': 'PTC_FM', 'dataset': '../datasets/PTC/Train/FM.ds',},
    #     {'name': 'PTC_FR', 'dataset': '../datasets/PTC/Train/FR.ds',},
    #     {'name': 'PTC_MM', 'dataset': '../datasets/PTC/Train/MM.ds',},
    #     {'name': 'PTC_MR', 'dataset': '../datasets/PTC/Train/MR.ds',},
]

# dslist = [
#     {
#         'name': 'Acyclic',
#         'dataset': '../datasets/acyclic/dataset_bps.ds',
#         'task': 'regression'
#     },  # node_labeled
#     {
#         'name': 'COIL-DEL',
#         'dataset': '../datasets/COIL-DEL/COIL-DEL_A.txt'
#     },  # edge_labeled
#     {
#         'name': 'PAH',
#         'dataset': '../datasets/PAH/dataset.ds',
#     },  # unlabeled
#     {
#         'name': 'Mutagenicity',
#         'dataset': '../datasets/Mutagenicity/Mutagenicity_A.txt'
#     },  # fully_labeled
#     {
#         'name': 'MAO',
#         'dataset': '../datasets/MAO/dataset.ds',
#     },
#     {
#         'name': 'MUTAG',
#         'dataset': '../datasets/MUTAG/MUTAG.mat',
#         'extra_params': {
#             'am_sp_al_nl_el': [0, 0, 3, 1, 2]
#         }
#     },
#     {
#         'name': 'Alkane',
#         'dataset': '../datasets/Alkane/dataset.ds',
#         'task': 'regression',
#         'dataset_y': '../datasets/Alkane/dataset_boiling_point_names.txt',
#     },
#     {
#         'name': 'BZR',
#         'dataset': '../datasets/BZR_txt/BZR_A_sparse.txt'
#     },
#     {
#         'name': 'COX2',
#         'dataset': '../datasets/COX2_txt/COX2_A_sparse.txt'
#     },
#     {
#         'name': 'ENZYMES',
#         'dataset': '../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'
#     },
#     {
#         'name': 'DHFR',
#         'dataset': '../datasets/DHFR_txt/DHFR_A_sparse.txt'
#     },
#     {
#         'name': 'SYNTHETIC',
#         'dataset': '../datasets/SYNTHETIC_txt/SYNTHETIC_A_sparse.txt'
#     },
#     {
#         'name': 'MSRC9',
#         'dataset': '../datasets/MSRC_9_txt/MSRC_9_A.txt'
#     },
#     {
#         'name': 'MSRC21',
#         'dataset': '../datasets/MSRC_21_txt/MSRC_21_A.txt'
#     },
#     {
#         'name': 'FIRSTMM_DB',
#         'dataset': '../datasets/FIRSTMM_DB/FIRSTMM_DB_A.txt'
#     },
#     {
#         'name': 'PROTEINS',
#         'dataset': '../datasets/PROTEINS_txt/PROTEINS_A_sparse.txt'
#     },
#     {
#         'name': 'PROTEINS_full',
#         'dataset': '../datasets/PROTEINS_full_txt/PROTEINS_full_A_sparse.txt'
#     },
#     {
#         'name': 'D&D',
#         'dataset': '../datasets/D&D/DD.mat',
#         'extra_params': {
#             'am_sp_al_nl_el': [0, 1, 2, 1, -1]
#         }
#     },
#     {
#         'name': 'AIDS',
#         'dataset': '../datasets/AIDS/AIDS_A.txt'
#     },
#     {
#         'name': 'NCI1',
#         'dataset': '../datasets/NCI1/NCI1.mat',
#         'extra_params': {
#             'am_sp_al_nl_el': [1, 1, 2, 0, -1]
#         }
#     },
#     {
#         'name': 'NCI109',
#         'dataset': '../datasets/NCI109/NCI109.mat',
#         'extra_params': {
#             'am_sp_al_nl_el': [1, 1, 2, 0, -1]
#         }
#     },
#     {
#         'name': 'NCI-HIV',
#         'dataset': '../datasets/NCI-HIV/AIDO99SD.sdf',
#         'dataset_y': '../datasets/NCI-HIV/aids_conc_may04.txt',
#     },

#     #     # not working below
#     #     {'name': 'PTC_FM', 'dataset': '../datasets/PTC/Train/FM.ds',},
#     #     {'name': 'PTC_FR', 'dataset': '../datasets/PTC/Train/FR.ds',},
#     #     {'name': 'PTC_MM', 'dataset': '../datasets/PTC/Train/MM.ds',},
#     #     {'name': 'PTC_MR', 'dataset': '../datasets/PTC/Train/MR.ds',},
# ]
