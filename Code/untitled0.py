# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:37:42 2021

@author: phumt
"""

import numpy as np
import matplotlib.pyplot as plt

def calc_PV(hb, )


def wave_deconstruct(hb):
    t = np.linspace(0, 2*np.pi, len(hb), endpoint=True)
    n = len(t)
    
    W = np.zeros((n, 5))
    wave_params=None
    if wave_params==None: # wave_params=None
        # Define params for R_peaks as a tuple
        wave_params = [init_rpeak(hb, t)]
        # Initialize other params with zero
        for i in range(4):
            wave_params.append((0, 0, 0, 0, 0))
            

    for i in range(5):
        if i == 0:
            W[:, i] = sim_hb(wave_params[i], t)
        else:
            W[:, i] = sim_hb(wave_params[i], t)
            wave_params[i] = FMM(hb - W.sum(axis=1))
            
            RSS = (hb - W.sum(axis=1))
            
            if RSS_t < RSS:
                RSS = W.sum(axis=1)

    
wave_params
W
plt.plot(hb)
plt.plot(W)

while k <= 20:
    print(k)
    k += 1
