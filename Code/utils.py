# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:18:48 2021

@author: phumt, eric, ojas
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import biosppy.signals.ecg as bse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from collections import defaultdict
from processdata import filter, extract_features
from joblib import dump

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
            multilabel_params = {'onevsrestclassifier__estimator__{}'.format(i.split('__')[-1]): j for i, j in
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


def filter_all(ecg_data, **kwargs):
    """
    Filters data matrix ecg_data using Ojas's filter function

    Returns filtered data matrix of shape (m x time_points (1000) x leads (12))

    **kwargs:

    - samplingrate = 100 (default)
    - remove_wandering = False (default)
    - bandpass = True (default) : applies butterworth bandpass filter
    """
    return np.array([filter(ecg, **kwargs) for ecg in ecg_data])


def gridcv_all(clf, column_names, categorical=None, **kwargs):
    """
    Prepares a GridSearchCV object for input data with ALL features (including categorical)
    Input data must be a dataframe when fitting!

    *args:
    - clf:
        sklearn classifier to use
    - column_names:
        list, set, pandas object of column names of input data
    - categorical:
        list of columns containing categorical data
        default: None (assumes all columns numeric)

    **kwargs from GridSearchCV can be passed in as well
    Useful kwargs:
        - scoring = 'roc_auc_ovr' (makes scoring based on roc_auc for onevsrest multilabel classification)
        - n_jobs = int (runs jobs in parallel processes, maybe speeding things up but be careful and check docs!)

    for available scoring metrics see:
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """

    if categorical:
        numeric_features = list(set(column_names) - set(categorical))

        # transformers for different data types
        numeric_transformer = make_pipeline(StandardScaler(), PCA())
        categorical_transformer = make_pipeline('passthrough')

        # combining the above
        preprocessing = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                        ('cat', categorical_transformer, categorical)])

        # combine preprocessing and classifier
        pipe = make_pipeline(preprocessing, OneVsRestClassifier(clf))

        # default parameters
        default_params = {'columntransformer__num__pca': ['passthrough', PCA(.80, svd_solver='full'),
                                                      PCA(.90, svd_solver='full'), PCA(.95, svd_solver='full')]}

    else:
        preprocessing = make_pipeline(StandardScaler(), PCA())

        # combine preprocessing and classifier
        pipe = make_pipeline(preprocessing, OneVsRestClassifier(clf))

        # default parameters
        default_params = {'pipeline__pca': ['passthrough', PCA(.80, svd_solver='full'),
                                                      PCA(.90, svd_solver='full'), PCA(.95, svd_solver='full')]}

    clf_key = str(type(clf)).split('.')[-1][:-2].lower()

    if clf_key in model_params.keys():
        # update parameters with those for chosen model
        multilabel_params = {'onevsrestclassifier__estimator__{}'.format(i.split('__')[-1]): j for i, j in
                             model_params[clf_key].items()}
        default_params.update(multilabel_params)

    return GridSearchCV(pipe, default_params, **kwargs)


def model_wrapper(estimator_list, x_train, y_train, cat=None, prefix='save-file-label', **kwargs):
    """
    Builds, fits, and saves models in estimator_list with gridcv_all function

    *args
    - estimator_list: list of sklearn classification model objects
    - x_train: dataframe of training data
    - y_train: dataframe of target classes for training data
    - cat: list of categorical features (column labels in dataframe)
    - prefix: string prefix for output file

    **kwargs from GridSearchCV
    Useful kwargs:
        - scoring = 'roc_auc_ovr' (makes scoring based on roc_auc for onevsrest multilabel classification)
        - n_jobs = int (runs jobs in parallel processes, maybe speeding things up but be careful and check docs!)

    for available scoring metrics see:
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """

    fit_models = defaultdict(GridSearchCV)
    for model in estimator_list:
        label = str(type(model)).split('.')[-1][:-2]
        fit_models[label] = gridcv_all(model, x_train.columns, categorical=cat, **kwargs).fit(x_train, y_train)
        dump(fit_models[label], os.path.join(output_folder, '{}_{}.joblib'.format(prefix, label)))

    return fit_models


def extract_all_features(ecgdata, **kwargs):
    """
    Extract ecg descriptors (extract_features function)
    -- Note this function performs PCA prior to feature extraction

    Returns a n x 6 array (n patient ECGs, 6 descriptors)

    **kwargs:

    samplingrate=100
    expandtrace=True
    pca=True
    lead=2

    """
    return np.array([list(extract_features(ecgdata[i, :, :], kwargs)[0].values()) for i in range(ecgdata.shape[0])])

# will add this stuff to separate function for a classification report wrapper to load our models from saves instead
# if report:
#     for key, m in fit_models.items():
#         report = classification_report(y_val, m.predict(x_val))
#         val_reports['key'] = report

