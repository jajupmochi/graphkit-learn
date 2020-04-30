#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:17:38 2020

@author: ljia
"""

from enum import Enum, unique

@unique
class AlgorithmState(Enum):
    """can be used to specify the state of an algorithm.
    """
    CALLED = 1 # The algorithm has been called.
    INITIALIZED = 2 # The algorithm has been initialized.
    CONVERGED = 3 # The algorithm has converged.
    TERMINATED = 4 # The algorithm has terminated.