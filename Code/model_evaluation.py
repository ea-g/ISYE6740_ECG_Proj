# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:59:11 2021

@author: phumt
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scikitplot as skplt
from collections import defaultdict
from joblib import dump, load

##################

def plot_multi_ROC(models, X_test, y_test):
    '''Plot the multiple ROC curves for each model
    from https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python'''
    for model in models:
        y_proba = model.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, y_proba)
        plt.show()


def plot_scores_CV(df):
    '''Return a barplot comparing model performance'''
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        x = 'classifier',
        y = 'best_score',
        data = df,
        kind = 'bar',
        hue = 'data_stream')
    g.set_axis_labels('', 'Best score')
    g.despine(left=True)
    plt.xticks(rotation=15)
    plt.show()
    
    
def plot_scores_test(df, col_name='ROC_AUC'):  set
    '''Return a barplot comparing model performance'''
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        x = 'classifier',
        y = 'ROC_AUC',
        data = df,
        kind = 'bar',
        hue = 'data_stream')
    g.set_axis_labels('', 'Best score')
    g.despine(left=True)
    plt.xticks(rotation=15)
    plt.show()


def get_performance(outputs):   
    '''Return a dataframe containing the optimal model info'''
    frames = []
    models = []
    for entry in outputs:
        model_name = entry.name.split('.')[0]
        model_output = load(entry)
        temp_dict = {
            'output_name': model_name,
            'best_score': model_output.best_score_,
            'best_parameters': model_output.best_params_
            }
        frames.append(pd.DataFrame(temp_dict))
        models.append(model_output)
    model_info_df = pd.concat(frames, ignore_index=True)
    
    # Label each row with the corresponding classifier
    # and data stream
    model_info_df[['data_stream', 'classifier']] = model_info_df['output_name']\
        .str.split('_', 1, expand=True)
    
    return model_info_df


################## 

# The current working directory should be "..\ISYE6740_ECG_Proj"
cur_dir = os.getcwd()
output_folder = cur_dir + '\Output'

# Get a list of model output files
outputs = [entry for entry in os.scandir(output_folder) \
           if not entry.name.startswith('.') and entry.is_file()]

model_info_df = get_performance(outputs)

plot_scores(model_info_df)

################## Need the model input to plot ROC

plot_multi_ROC(models, X_test_meta, y_test)

# model = models[3]
# y_proba = model.predict_proba(X_val_wv)
# skplt.metrics.plot_roc_curve(y_train_meta, y_proba)
# plt.show()
