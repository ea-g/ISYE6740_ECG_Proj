"""
Quick example for how to access results
"""
import os
import numpy as np
import pandas as pd

results_folder = os.path.abspath(r'..\Results')

# to get the scores data frame
scores_df = pd.read_hdf(os.path.join(results_folder, 'scores.h5'), key='scores')

# to get all multilabel confusion matrices for plotting
cnf_matrices = np.load(os.path.join(results_folder, 'cnf_matrices.npy'), allow_pickle=True).item()

# to get all classification reports
clf_reports = np.load(os.path.join(results_folder, 'clf_reports.npy'), allow_pickle=True).item()

# both clf_reports and cnf_matrices can be accessed like nested dictionaries
# use the convention clf_reports['data-stream']['modelname'] to get data
