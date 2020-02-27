import sys
import pathlib
sys.path.insert(0, "../")

import numpy as np

from gklearn.utils.model_selection_precomputed import model_selection_for_precomputed_kernel
from datasets.ds import dslist