import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from joblib import load
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp


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
        plt.savefig(os.path.join(save_folder, save + '.jpg'))

    plt.show()


def multilabel_roc(y_true, y_prob, title, labels, save=None):
    """
    multi-label ROC curve from :
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    n_classes = y_true.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for ii in range(n_classes):
        fpr[ii], tpr[ii], _ = roc_curve(y_true[:, ii], y_prob[:, ii])
        roc_auc[ii] = auc(fpr[ii], tpr[ii])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[ii] for ii in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for ii in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[ii], tpr[ii])

    lw = 2

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for ii, color, lab in zip(range(n_classes), colors, labels):
        plt.plot(fpr[ii], tpr[ii], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                       ''.format(lab, roc_auc[ii]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save:
        save_folder = os.path.join(docs_folder, 'ROC-plots')
        plt.savefig(os.path.join(save_folder, save + '.jpg'))
    plt.show()


# ================================================================================================
# getting some confusion matrices below here
top_models = scores_df.sort_values('ROC_AUC', ascending=False).head(6).reset_index(drop=True)
data_keys = {'raw02-MR': 'MiniRocket features from raw ECG signals',
             'raw02-MR-meta-ecg': 'MiniRocket features from raw ECG signals,\n patient meta-data, and ECG descriptors',
             'fil02-MR': 'MiniRocket features from filtered ECG signals',
             'fil02-MR-meta-ecg': 'MiniRocket features from filtered ECG signals,\n '
                                  'patient meta-data, and ECG descriptors'}
labels = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
# for i, row in top_models.iterrows():
#     cnf = cnf_matrices[row.data_stream][row.classifier]
#     title = '{} with {} \nusing ROC AUC as CV scoring'.format(row.classifier,
#                                                               data_keys[row.data_stream])
#     plot_ml_cnf(cnf, labels, title=title, save=row.data_stream + '-' + row.classifier, cmap='Blues')

# ===============================================================================================
# getting corresponding ROC plots

# y_true = np.load(os.path.join(data_folder, 'y_test-final.npy'))
# for i, row in top_models.iterrows():
#     model_path = os.path.join(output_folder, row.data_stream + '_' + row.classifier + '.joblib')
#     model = load(model_path)
#     test_path = os.path.join(data_folder, row.data_stream + '-test.h5')
#     test_data = pd.read_hdf(test_path, key='test')
#     y_prob = model.predict_proba(test_data)
#     title = '{} with {} \nusing ROC AUC as CV scoring'.format(row.classifier,
#                                                               data_keys[row.data_stream])
#     multilabel_roc(y_true, y_prob, title, labels, save=row.data_stream + '-' + row.classifier)

# ===================================================================================================
# getting boxplots for ECG descriptors
ecg_path = os.path.join(data_folder, 'ecg_descriptors_clean.csv')
ecg_des = pd.read_csv(ecg_path, index_col=0)
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.ravel()
color = sns.color_palette("husl", 6)
for ax, col, c in zip(range(6), ecg_des.columns, color):
    sns.set_style("whitegrid")
    sns.violinplot(data=ecg_des[[col]], ax=axs[ax], color=c)
plt.suptitle('Violin Plots of ECG Descriptors', y=.98)
plt.savefig(os.path.join(docs_folder, 'ECG_descriptors_violin.jpg'))
plt.show()
