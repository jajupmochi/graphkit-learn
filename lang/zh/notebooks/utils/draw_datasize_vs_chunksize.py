#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw data size v.s. chunksize.
Created on Mon Oct  8 12:00:45 2018

@author: ljia
"""
import numpy as np
import matplotlib.pyplot as plt

def loadmin(file):
    result = np.load(file)
    return np.argmin(result), np.min(result), result

def idx2chunksize(idx):
    if idx < 9:
        return idx * 10 + 10
    elif idx < 18:
        return (idx - 8) * 100
    elif idx < 27:
        return (idx - 17) * 1000
    else:
        return (idx - 26) * 10000
    
def idx2chunksize2(idx):
    if idx < 5:
        return idx * 20 + 10
    elif idx < 10:
        return (idx - 5) * 200 + 100
    elif idx < 15:
        return (idx - 10) * 2000 + 1000
    else:
        return (idx - 15) * 20000 * 10000

idx, mrlt, rlt = loadmin('../test_parallel/myria/ENZYMES.npy')
csize = idx2chunksize2(idx)

#dsize = np.array([183, 150, 68, 94, 188, 2250, 600])
dsize = np.array([183, 150, 68, 94, 188, 2250])
dsize = dsize * (dsize + 1) / 2
#cs_sp_laptop = [900, 400, 70, 900, 2000, 8000, 300]
#cs_sp_myria = [900, 500, 500, 300, 400, 4000, 300]
cs_sp_laptop = [900, 400, 70, 900, 2000, 8000]
cs_sp_myria = [900, 500, 500, 300, 400, 4000]
cd_ssp_laptop = [500, 700, 500, 70, 3000, 3000]
cd_ssp_myria = [100, 90, 700, 30, 3000, 5000]

cs_sp_laptop = [x for _, x in sorted(zip(dsize, cs_sp_laptop))]
cs_sp_myria = [x for _, x in sorted(zip(dsize, cs_sp_myria))]
cd_ssp_laptop = [x for _, x in sorted(zip(dsize[0:6], cd_ssp_laptop))]
cd_ssp_myria = [x for _, x in sorted(zip(dsize[0:6], cd_ssp_myria))]
dsize.sort()
cd_mean = np.mean([cs_sp_laptop[0:6], cs_sp_myria[0:6], cd_ssp_laptop, cd_ssp_myria], 
                  axis=0)
#np.append(cd_mean, [6000])

fig, ax = plt.subplots()
##p1 = ax.plot(dsize, cs_sp_laptop, 'o-', label='sp laptop')
#p2 = ax.plot(dsize, cs_sp_myria, 'o-', label='sp CRIANN')
##p3 = ax.plot(dsize[0:6], cd_ssp_laptop, 'o-', label='ssp laptop')
#p4 = ax.plot(dsize[0:6], cd_ssp_myria, 'o-', label='ssp CRIANN')
#p5 = ax.plot(dsize[0:6], cd_mean, 'o-', label='mean')

#p1 = ax.plot(dsize[0:5], cs_sp_laptop[0:5], 'o-', label='sp laptop')
p2 = ax.plot(dsize[0:5], cs_sp_myria[0:5], 'o-', label='sp CRIANN')
#p3 = ax.plot(dsize[0:5], cd_ssp_laptop[0:5], 'o-', label='ssp laptop')
p4 = ax.plot(dsize[0:5], cd_ssp_myria[0:5], 'o-', label='ssp CRIANN')
p5 = ax.plot(dsize[0:5], cd_mean[0:5], 'o-', label='mean')


#ax.set_xscale('log', nonposx='clip')
#ax.set_yscale('log', nonposy='clip')
ax.set_xlabel('data sizes')
ax.set_ylabel('chunksizes for the fastest computation')
ax.legend(loc='upper left')
plt.show()