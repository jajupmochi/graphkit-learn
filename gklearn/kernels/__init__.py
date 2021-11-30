# -*-coding:utf-8 -*-
"""gklearn - graph kernels module
"""

# info
__version__ = "0.1"
__author__ = "Linlin Jia"
__date__ = "November 2018"


from gklearn.kernels.graph_kernel import GraphKernel
from gklearn.kernels.common_walk import CommonWalk
from gklearn.kernels.marginalized import Marginalized
from gklearn.kernels.random_walk_meta import RandomWalkMeta
from gklearn.kernels.sylvester_equation import SylvesterEquation
from gklearn.kernels.conjugate_gradient import ConjugateGradient
from gklearn.kernels.fixed_point import FixedPoint
from gklearn.kernels.spectral_decomposition import SpectralDecomposition
from gklearn.kernels.random_walk import RandomWalk
from gklearn.kernels.shortest_path import ShortestPath
from gklearn.kernels.structural_sp import StructuralSP
from gklearn.kernels.path_up_to_h import PathUpToH
from gklearn.kernels.treelet import Treelet
from gklearn.kernels.weisfeiler_lehman import WeisfeilerLehman, WLSubtree

from gklearn.kernels.metadata import GRAPH_KERNELS, list_of_graph_kernels

# old version.
from gklearn.kernels.commonWalkKernel import commonwalkkernel
from gklearn.kernels.marginalizedKernel import marginalizedkernel
from gklearn.kernels.randomWalkKernel import randomwalkkernel
from gklearn.kernels.spKernel import spkernel
from gklearn.kernels.structuralspKernel import structuralspkernel
from gklearn.kernels.untilHPathKernel import untilhpathkernel
from gklearn.kernels.treeletKernel import treeletkernel
from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
