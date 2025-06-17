#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:21:25 2019

@author: ljia
"""

import sys
import time

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log." + str(time.time()) + ".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()