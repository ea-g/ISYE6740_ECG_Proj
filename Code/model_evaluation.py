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
from run_models import X_train, X_test

##################

def plot_multi_ROC(models, X_test, y_test):
    '''Plot the multiple ROC curves for each model
    from https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python'''

    for model in models:
        y_proba = model.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, y_proba)
        plt.show()


def plot_scores(df):
    fig, ax = plt.subplots()
    ax = sns.barplot(
        x = 'output_name',
        y = 'best_score',
        data = df)
    ax.set_title('Model scores')
    ax.set_xlabel('')
    plt.xticks(rotation=90)
    plt.show()


# The current working directory should be "..\ISYE6740_ECG_Proj"
cur_dir = os.getcwd()
output_folder = cur_dir + '\Output'

# Get a list of model output files
outputs = [entry for entry in os.scandir(output_folder) \
           if not entry.name.startswith('.') and entry.is_file()]

# Create a dataframe containing the optimal model info
frames = []
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
df = pd.concat(frames, ignore_index=True)


################## Need the model input to plot ROC

plot_multi_ROC(models[0], X_train, y_train)

model = models[0]
y_proba = model.predict_proba(X_train)
skplt.metrics.plot_roc_curve(y_train_meta, y_proba)
plt.show()
