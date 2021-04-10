# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:18:48 2021

@author: phumt
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def plot_12leads(ecg, save_fig=False):
    # Create an array for x
    x = np.arange(len(ecg))
    # Plot all the 12 leads
    fig, ax = plt.subplots(12, 1, figsize=(6.4*2, 4.8*2), dpi=300)
    for i in range(12):
        ax[i].plot(x, ecg[:, i])
        ax[i].get_yaxis().set_visible(False)
        #ax[i].xaxis.set_minor_locator(MultipleLocator(5))
    plt.show()
    # Save the plots
    if save_fig == True:
        plt.savefig('ecg.jpg')



