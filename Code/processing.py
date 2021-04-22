# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import biosppy.signals.ecg as bse

def plot_12leads(ecg, save_fig=False):
    '''Plot 12 ECG-leads from a patient'''
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
        
        
def plot_lead(ecg):
    '''Plots a single ECG lead'''
    x = np.arange(len(ecg))
    fig, ax = plt.subplots()
    ax.plot(x, ecg)
    ax.get_yaxis().set_visible(False)
    plt.show()
    

def plot_heartbeats(hbs, diff_plots=False):
    '''Plot each heartbeat from a single ECG lead. If diff_plots is False, 
    then we plot all heartbeats on the same plot'''
    if diff_plots:
        fig, ax =  plt.subplots(len(hbs), 1)
        for i, hb in enumerate(hbs):
            x = np.arange(0, len(hb))
            ax[i].plot(x, hb)
        plt.show()
    else:
        fig, ax = plt.subplots()
        for i, hb in enumerate(hbs):
            x = np.arange(0, len(hb))
            ax.plot(x, hb)
        plt.show()
    
### ======================= Data processing
# Plot 12 leads from patient zero in the training set
p0_all_leads = X_train[0, :, :]
plot_12leads(p0_all_leads)

# Plot only the first lead from patient zero
p0_0_lead = p0_all_leads[:, 1]
plot_lead(p0_0_lead)

# To extract a single heartbeat, we begin by identifying 
# the location of R-peaks. Do we actually need to correct the R-peaks?
r_locs = bse.christov_segmenter(signal=p0_0_lead, sampling_rate=100)[0]
r_locs = bse.correct_rpeaks(signal=p0_0_lead, 
                            rpeaks=r_locs, 
                            sampling_rate=100, 
                            tol=0.05)[0]


## ========================= NOTE
# Do we need to correct the locations using bse.correct_rpea'ks?'?
'''foo = bse.correct_rpeaks(signal=p0_0_lead, 
                     rpeaks=r_locs, 
                     sampling_rate=100, 
                     tol=0.05)[0]'''
    
## ==============================
hbs = bse.extract_heartbeats(signal=p0_0_lead, 
                             rpeaks=r_locs, 
                             sampling_rate=100, 
                             before=0.2, 
                             after=0.4)[0]

hb =  hbs[0]