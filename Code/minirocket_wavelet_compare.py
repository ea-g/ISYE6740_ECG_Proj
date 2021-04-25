# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:52:18 2021

@author: Eric
"""

from sklearn.model_selection import GridSearchCV
import get_data
from sktime.transformations.panel.rocket import MiniRocketMultivariate 
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from wavelet_features import get_ecg_features
from utils import matrix_2_df, make_gridcv, output_folder
from collections import defaultdict
from joblib import dump, load
import os
import matplotlib.pyplot as plt
import numpy as np

# prepare data for MiniRocket and sktime models
X_train_df = matrix_2_df(get_data.X_train)
X_test_df = matrix_2_df(get_data.X_test)

# extract features with MiniRocket
mini_mv = MiniRocketMultivariate().fit(X_train_df)
X_train_mr = mini_mv.transform(X_train_df)
X_test_mr = mini_mv.transform(X_test_df)

# extract features with wavelet transform
X_train_wv = get_ecg_features(get_data.X_train)
X_test_wv = get_ecg_features(get_data.X_test)

# list of models, feel free to add more (for parameters see model_params dict 
# in utils.py)
models = [SGDClassifier(max_iter=2000), LogisticRegression(max_iter=300), GaussianNB(),
          KNeighborsClassifier(), RandomForestClassifier(), RidgeClassifier(),
          AdaBoostClassifier()]

unfit_models = defaultdict(GridSearchCV)

# instantiate models
for m in models[4:5]:
    label = str(type(m)).split('.')[-1][:-2]
    unfit_models[label] = make_gridcv(m, True, scoring='roc_auc_ovr', n_jobs=2)

# fit models using minirocket and wavelet transform
fit_models = defaultdict(GridSearchCV)
for name, clf in unfit_models.items():
    for training, label in zip([X_train_mr, X_train_wv], ['MiniR-multi', 'Wav-multi']):
        fit_models[name+'_'+label] = clf.fit(training, get_data.y_train_multi)
        dump(clf, os.path.join(output_folder, '{}_{}_gridcv.joblib'.format(name, label)))

# # below here is loading in saved models for comparison
# sgd_minir_clf = load(os.path.join(output_folder, 'SGDClassifier_MiniR_gridcv.joblib'))
# sgd_wav_clf = load(os.path.join(output_folder, 'SGDClassifier_Wav_gridcv.joblib'))
#
# test_scores = [sgd_minir_clf.score(X_test_mr, get_data.y_test), sgd_wav_clf.score(X_test_wv, get_data.y_test)]
# training_score = [sgd_minir_clf.best_score_, sgd_wav_clf.best_score_]
#
# classifiers = ['SVM using MiniRocket', 'SVM using Wavelet']
#
#
# x = np.arange(len(classifiers))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, test_scores, width, label='test_set')
# rects2 = ax.bar(x + width/2, training_score, width, label='training_set')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy Scores')
# ax.set_xticks(x)
# ax.set_xticklabels(classifiers)
# ax.legend()
# fig.tight_layout()
# plt.show()
