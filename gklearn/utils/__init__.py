# -*-coding:utf-8 -*-
"""gklearn - utils module

Implement some methods to manage graphs
 graphfiles.py : load .gxl and .ct files
 utils.py : compute some properties on networkX graphs


"""

# info
__version__ = "0.1"
__author__ = "Benoit Gaüzère"
__date__ = "November 2017"

# from utils import graphfiles
# from utils import utils
from gklearn.utils.dataset import Dataset, split_dataset_by_target
from gklearn.utils.graph_files import load_dataset, save_dataset
from gklearn.utils.timer import Timer
from gklearn.utils.utils import get_graph_kernel_by_name
from gklearn.utils.utils import compute_gram_matrices_by_class
from gklearn.utils.utils import SpecialLabel
from gklearn.utils.utils import normalize_gram_matrix, compute_distance_matrix
from gklearn.utils.trie import Trie
from gklearn.utils.knn import knn_cv, knn_classification
