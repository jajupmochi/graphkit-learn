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
from gklearn.utils.utils import SpecialLabel, dummy_node, undefined_node, dummy_edge
from gklearn.utils.utils import normalize_gram_matrix, compute_distance_matrix
from gklearn.utils.utils import check_json_serializable, is_basic_python_type
from gklearn.utils.trie import Trie
from gklearn.utils.knn import knn_cv, knn_classification
from gklearn.utils.model_selection_precomputed import model_selection_for_precomputed_kernel
from gklearn.utils.iters import get_iters
