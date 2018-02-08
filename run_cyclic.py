import sys
sys.path.insert(0, "../")
from pygraph.utils.utils import kernel_train_test
from pygraph.kernels.cyclicPatternKernel import cyclicpatternkernel

import numpy as np

datafile = '../../../../datasets/NCI-HIV/AIDO99SD.sdf'
datafile_y = '../../../../datasets/NCI-HIV/aids_conc_may04.txt'
kernel_file_path = 'kernelmatrices_path_acyclic/'

kernel_para = dict(node_label = 'atom', edge_label = 'bond_type', labeled = True)

kernel_train_test(datafile, kernel_file_path, cyclicpatternkernel, kernel_para, \
    hyper_name = 'cycle_bound', hyper_range = np.linspace(0, 1000, 21), normalize = False, \
    datafile_y = datafile_y, model_type = 'classification')
