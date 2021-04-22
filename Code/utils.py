# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:18:48 2021

@author: phumt, eric
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

output_folder =os.path.abspath('..\Output')

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
        plt.savefig(os.path.join(output_folder, 'ecg.jpg'))

def matrix_2_df(matrix, column_prefix='lead_'):
    '''
    Converts 3-D numpy matrix to dataframe for input to sktime models.
    '''
    from collections import defaultdict
    from pandas import Series, DataFrame
    output = defaultdict(list)
    for i in matrix[:, :, :]:
        for c in range(matrix.shape[-1]):
            output[column_prefix+str(c+1)].append(Series(i[:, c:c+1].flatten()))            
    return DataFrame(data=output)

# dictionary of dictionaries of model paramenters (follow formatting if edit)
model_params = {'svc':{'svc__C':np.logspace(-3, 1, 5), 'svc__kernel':['scale', 'auto']}, 
               'logisticregression':{'logisticregression__C':np.logspace(-3, 1, 5)}, 
               'kneighborsclassifier': {'kneighborsclassifier__k':[1, 5, 7]}, 
               'randomforestclassifier': {'randomforestclassifier__n_estimators':[50, 100, 200]},
               'ridgeclassifier':{'ridgeclassifier__alpha': np.logspace(-2, 2, 5)},
               'adaboostclassifier':{'adaboostclassifier__n_estimators':[50, 100]}}

def make_gridcv(classifier):
    '''
    Given an sklearn classifier (e.g. SVC()), builds a pipeline and parameter grid based on model_params dictionary and 
    default estimators (scaling, pca). 
    
    Returns a GridSearchCV object ready to be fit to data.
    '''
    #==================================================================================
    # We will need to edit the evaluation critereon of grid cv to a different metric
    # for multilabel classification. See:
    # https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
    # 
    # Multimetric is also possible:
    # https://scikit-learn.org/stable/modules/grid_search.html#multimetric-grid-search
    #==================================================================================
    pipe = make_pipeline(StandardScaler(), PCA(), classifier)
    default_params = dict(pca=['passthrough', PCA(.80, svd_solver='full'), PCA(.90, svd_solver='full'), PCA(.95, svd_solver='full')])
    
    # update parameters with model's parameters
    default_params.update(model_params[list(pipe.named_steps.keys())[-1]])
    
    return GridSearchCV(pipe, default_params)


    

