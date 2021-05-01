import os
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_auc_score, hamming_loss
from joblib import load
from collections import defaultdict

# folder paths
data_folder = os.path.abspath(r'..\Data')
output_folder = os.path.abspath(r'..\Output')
results_folder = os.path.abspath(r'..\Results')

# file names
outputs = os.listdir(output_folder)
test_data = [i for i in os.listdir(data_folder) if 'test.h5' in i]

# true labels
y_true = np.load(os.path.join(data_folder, 'y_test-final.npy'))

data_model = []
for t in test_data:
    pref = '-'.join(t.split('-')[:-1])
    for mod in outputs:
        mod_pref = mod.split('_')[0]
        if mod_pref == pref:
            data_model.append((pref, mod))

# read in all test data streams
x_tests = {'-'.join(t.split('-')[:-1]): pd.read_hdf(os.path.join(data_folder, t), key='test') for t in test_data}

cnf_matrices = defaultdict(dict)
scores = defaultdict(list)
clf_reports = defaultdict(dict)
for dm in data_model:
    model = load(os.path.join(output_folder, dm[1]))
    y_pred = model.predict(x_tests[dm[0]])

    # add in data
    scores['classifier'].append(dm[1].split('_')[-1].split('.')[0])
    scores['data_stream'].append(dm[0])

    if 'SGDClassifier' not in dm[1]:
        y_prob = model.predict_proba(x_tests[dm[0]])
        scores['ROC_AUC'].append(roc_auc_score(y_true, y_prob))
    else:
        scores['ROC_AUC'].append(np.nan)

    scores['hamming_loss'].append(hamming_loss(y_true, y_pred))
    scores['best_params'].append(model.best_params_)

    # get confusion matrix
    cnf_matrices[dm[0]][dm[1].split('_')[-1].split('.')[0]] = multilabel_confusion_matrix(y_true, y_pred)

    # get clf report
    clf_reports[dm[0]][dm[1].split('_')[-1].split('.')[0]] = classification_report(y_true, y_pred,
                                                                                   output_dict=True,
                                                                                   target_names=['CD', 'HYP', 'MI',
                                                                                                 'NORM', 'STTC'])

# save scores dataframe
pd.DataFrame(data=scores).to_hdf(os.path.join(results_folder, 'scores.h5'), key='scores')

# save clf_reports
np.save(os.path.join(results_folder, 'clf_reports.npy'), clf_reports)

# save confusion matrices
np.save(os.path.join(results_folder, 'cnf_matrices.npy'), cnf_matrices)



