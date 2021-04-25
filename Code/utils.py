# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:18:48 2021

@author: phumt, eric
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import biosppy.signals.ecg as bse
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

output_folder = os.path.abspath('..\Output')


def plot_12leads(ecg, save_fig=False):
    # Create an array for x
    x = np.arange(len(ecg))
    # Plot all the 12 leads
    fig, ax = plt.subplots(12, 1, figsize=(6.4 * 2, 4.8 * 2), dpi=300)
    for i in range(12):
        ax[i].plot(x, ecg[:, i])
        ax[i].get_yaxis().set_visible(False)
        # ax[i].xaxis.set_minor_locator(MultipleLocator(5))
    plt.show()
    # Save the plots
    if save_fig:
        plt.savefig(os.path.join(output_folder, 'ecg.jpg'))


def matrix_2_df(matrix, column_prefix='lead_'):
    """
    Converts 3-D numpy matrix to dataframe for input to sktime models.
    """
    from collections import defaultdict
    from pandas import Series, DataFrame
    output = defaultdict(list)
    for i in matrix[:, :, :]:
        for c in range(matrix.shape[-1]):
            output[column_prefix + str(c + 1)].append(Series(i[:, c:c + 1].flatten()))
    return DataFrame(data=output)


# dictionary of dictionaries of model paramenters (follow formatting if edit)
model_params = {'svc': {'svc__C': np.logspace(-3, 1, 5), 'svc__kernel': ['rbf'], 'svc__gamma': ['auto', 'scale']},
                'sgdclassifier': {'sgdclassifier__alpha': np.logspace(-4, -1, 5)},
                'linearsvc': {'linearsvc__C': np.logspace(-3, 1, 5)},
                'logisticregression': {'logisticregression__C': np.logspace(-3, 1, 5)},
                'kneighborsclassifier': {'kneighborsclassifier__k': [1, 5, 7]},
                'randomforestclassifier': {'randomforestclassifier__n_estimators': [50, 100, 200]},
                'ridgeclassifier': {'ridgeclassifier__alpha': np.logspace(-2, 2, 5)},
                'adaboostclassifier': {'adaboostclassifier__n_estimators': [50, 100]}}


def make_gridcv(classifier, multi_label=False, **kwargs):
    """
    Given an sklearn classifier (e.g. SVC()), builds a pipeline and parameter grid based on model_params dictionary and
    default estimators (scaling, pca).

    Returns a GridSearchCV object ready to be fit to data.

    **kwargs from GridSearchCV can be passed in as well
    Useful kwargs:
        - scoring = 'roc_auc_ovr' (makes scoring based on roc_auc for onevsrest multilabel classification)
        - n_jobs = int (runs jobs in parallel processes, maybe speeding things up but be careful and check docs!)

    for available scoring metrics see:
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """
    # ==================================================================================
    # We will need to edit the evaluation criterion of grid cv to a different metric
    # for multilabel classification. See:
    # https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
    # 
    # Multimetric is also possible:
    # https://scikit-learn.org/stable/modules/grid_search.html#multimetric-grid-search
    # ==================================================================================

    clf_key = str(type(classifier)).split('.')[-1][:-2].lower()

    default_params = dict(
        pca=['passthrough', PCA(.80, svd_solver='full'), PCA(.90, svd_solver='full'), PCA(.95, svd_solver='full')])

    if multi_label:
        pipe = make_pipeline(StandardScaler(), PCA(), OneVsRestClassifier(classifier))
        # update parameters with the chosen model's, formatting for OneVsRest
        if clf_key in model_params.keys():
            multilabel_params = {'onevsrestclassifier__estimator__{}'.format(i.split('_')[-1]): j for i, j in
                                 model_params[clf_key].items()}
            default_params.update(multilabel_params)


    else:
        pipe = make_pipeline(StandardScaler(), PCA(), classifier)
        # update parameters with model's parameters
        if clf_key in model_params.keys():
            default_params.update(model_params[clf_key])

    return GridSearchCV(pipe, default_params, **kwargs)


def extract_hbs(lead):
    """ Return a list of heartbeats from a single ECG lead
    """
    # To extract a single heartbeat, we begin by identifying 
    # the location of R-peaks. Do we actually need to correct the R-peaks?
    r_locs = bse.christov_segmenter(signal=lead, sampling_rate=100)[0]
    r_locs = bse.correct_rpeaks(signal=lead,
                                rpeaks=r_locs,
                                sampling_rate=100,
                                tol=0.05)[0]

    hbs = bse.extract_heartbeats(signal=lead,
                                 rpeaks=r_locs,
                                 sampling_rate=100,
                                 before=0.2,
                                 after=0.4)[0]
    return hbs


def plot_heartbeats(hbs, diff_plots=False):
    """Plot heartbeats from a single ECG lead. If diff_plots is False,
    plot all heartbeats on the same plot, else plot many subplots"""
    if diff_plots:
        fig, ax = plt.subplots(len(hbs), 1)
        for i, hb in enumerate(hbs):
            x = np.arange(0, len(hb))
            ax[i].plot(x, hb)
        ax.set_title('Number of heartbeats: {}'.format(len(hbs)))
        plt.show()
    else:
        fig, ax = plt.subplots()
        for i, hb in enumerate(hbs):
            x = np.arange(0, len(hb))
            ax.plot(x, hb)
        ax.set_title('Number of heartbeats: {}'.format(len(hbs)))
        plt.show()
