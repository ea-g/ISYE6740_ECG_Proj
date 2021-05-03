import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


# assumes current working directory is ..\Code (where this file is)
data_folder = os.path.abspath(r'..\Data')
output_folder = os.path.abspath(r'..\Output')
results_folder = os.path.abspath(r'..\Results')
docs_folder = os.path.abspath(r'..\Docs')

# get confusion matrices
cnf_matrices = np.load(os.path.join(results_folder, 'cnf_matrices.npy'), allow_pickle=True).item()

# get scores df
scores_df = pd.read_csv(os.path.join(results_folder, 'scores.csv'), index_col=0)


def plot_ml_cnf(cnf_matrix, labels, title=None, save=None, **kwargs):
    """
    Plots a multi-label confusion matrix for our problem given a numpy array of a multi-label confusion matrix

    *args:
        - cnf_matrix - np.array of a multi-label confusion matrix
        - labels - list of class labels
    **optional
        - title - string of title for the plot
        - save - string of save file name, if specified then the plot is saved in the Results folder as a jpg

    **kwargs:
        matplot lib kwargs, we'll use cmap
        - cmap = 'Blues'

    code adapted from:
    https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
    """
    fig, axs = plt.subplots(1, cnf_matrix.shape[0], figsize=(10, 2.5))
    for m in range(cnf_matrix.shape[0]):
        disp = ConfusionMatrixDisplay(cnf_matrix[m])
        disp.plot(ax=axs[m], **kwargs)
        disp.ax_.set_title(labels[m])
        if m > 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.tight_layout(pad=2.5)

    fig.colorbar(disp.im_, ax=axs)
    if title:
        plt.suptitle(title, y=.98)

    if save:
        save_folder = os.path.join(docs_folder, 'Confusions')
        plt.savefig(os.path.join(save_folder, save+'.jpg'))

    plt.show()


# ================================================================================================
# getting some confusion matrices below here

top_models = scores_df.sort_values('ROC_AUC', ascending=False).head(6).reset_index(drop=True)
data_keys = {'raw02-MR': 'MiniRocket features from raw ECG signals',
             'raw02-MR-meta-ecg': 'MiniRocket features from raw ECG signals, patient meta-data, and ECG descriptors',
             'fil02-MR': 'MiniRocket features from filtered ECG signals',
             'fil02-MR-meta-ecg': 'MiniRocket features from filtered ECG signals, '
                                  'patient meta-data, and ECG descriptors'}
labels = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
for i, row in top_models.iterrows():
    cnf = cnf_matrices[row.data_stream][row.classifier]
    title = '{} with {} \nusing ROC AUC as CV scoring'.format(row.classifier,
                                                                                 data_keys[row.data_stream])
    plot_ml_cnf(cnf, labels, title=title, save=row.data_stream + '-' + row.classifier, cmap='Blues')
