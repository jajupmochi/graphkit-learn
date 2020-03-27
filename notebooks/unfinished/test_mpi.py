#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Message Passing Interface for cluster paralleling.
Created on Wed Nov  7 17:26:40 2018

@author: ljia
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
import time
size = comm.Get_size()
numDataPerRank = 10
data = None
if rank == 0:
    data = np.linspace(1, size * numDataPerRank, size * numDataPerRank)
    
recvbuf = np.empty(numDataPerRank, dtype='d')
comm.Scatter(data, recvbuf, root=0)
recvbuf += 1
print('Rank: ', rank, ', recvbuf received: ', recvbuf, ', size: ', size, ', time: ', time.time())

#if rank == 0:
#    data = {'key1' : [1,2, 3],
#            'key2' : ( 'abc', 'xyz')}
#else:
#    data = None
#
#data = comm.bcast(data, root=0)
#print('Rank: ',rank,', data: ' ,data)

#if rank == 0:
#    data = {'a': 7, 'b': 3.14}
#    comm.send(data, dest=1)
#elif rank == 1:
#    data = comm.recv(source=0)
#    print('On process 1, data is ', data)

#print('My rank is ', rank)

#for i in range(0, 100000000):
#    print(i)