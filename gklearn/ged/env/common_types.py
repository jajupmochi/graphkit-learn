#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:17:38 2020

@author: ljia
"""

from enum import Enum, auto

class AlgorithmState(Enum):
    """can be used to specify the state of an algorithm.
    """
    CALLED = auto # The algorithm has been called.
    INITIALIZED = auto # The algorithm has been initialized.
    CONVERGED = auto # The algorithm has converged.
    TERMINATED = auto # The algorithm has terminated.