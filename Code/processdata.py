import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp
import wfdb
import ast
import ecg_plot
from sklearn.decomposition import PCA
from scipy.signal import resample

# filtering - remove baseline wander + bandpass filter (butterworth)
def filter(ECGdata,samplingrate=100, remove_wandering=False, bandpass=True):
    if remove_wandering==True:
        ECG1 = hp.filtering.remove_baseline_wander(ECGdata,sample_rate=samplingrate,cutoff=0.03)
    else:
        ECG1 = ECGdata
    if bandpass==True:
        ECG2 = hp.filtering.filter_signal(ECG1.T,sample_rate=samplingrate,cutoff=[0.4,20],order=3, filtertype='bandpass')
        ECG2 = ECG2.T
    else:
        ECG2 = ECG1
    return ECG2
        # filter signal method 2

# plot ECG
def plotECG(ECGdata, samplingrate=100):
    nrows,ncols = ECGdata.shape
    if nrows >= ncols:
        ECGdata = ECGdata.T
    return ecg_plot.plot(ECGdata, sample_rate=samplingrate, title="12-lead ECG")

# extract ecg features
    
# PROCESSING DATA with PCA and then HP for measures/features
def extract_features(ECGdata, samplingrate=100, expandtrace=True, pca=True, lead=2):
    if pca==True:
        clf = PCA(n_components=1).fit(ECGdata)
        ECG_lean = clf.transform(ECGdata)
    else:
        nrows,ncols = ECGdata.shape
        if ncols >= nrows:
            ECGdata = ECGdata.T
        ECG_lean = ECGdata[:,lead]
    if expandtrace==True:
        resampled_signal = resample(ECG_lean, len(ECG_lean)*4)
        workingdata, measure = hp.process(hp.scale_data(resampled_signal.ravel()), 100 * 4)
    else:
        workingdata, measure = hp.process(ECG_lean.ravel(), sample_rate=100)
    
    features = {'heartrate':measure['bpm'], 'RRinterval':measure['ibi'], 'RRsd': measure['sdnn'], 
                'pNN20':measure['pnn20'], 'pNN50':measure['pnn50'],'RRmad': measure['hr_mad']}
    return features, workingdata, measure

    '''
    heartpy output Measures
    beats per minute (BPM)
    interbeat interval (IBI)
    standard deviation of RR intervals (SDNN)
    standard deviation of successive differences (SDSD)
    root mean square of successive differences (RMSSD)
    proportion of successive differences above 20ms (pNN20)
    proportion of successive differences above 50ms (pNN50)
    median absolute deviation of RR intervals (MAD)
    '''

