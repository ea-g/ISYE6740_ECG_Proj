import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from utils import matrix_2_df
import time
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_auc_score, hamming_loss
from more_plots import plot_ml_cnf, multilabel_roc

results_folder = os.path.abspath(r'..\Results')
data_path = os.path.abspath(r'..\Data')

timings = {}
start_time = time.perf_counter()

sampling_rate = 100


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


# load and convert annotation data
Y = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, data_path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# remove unlabeled data points
Y['label_len'] = Y.diagnostic_superclass.apply(lambda x: len(x))
X = X[Y.label_len > 0]
Y = Y[Y.label_len > 0]

# reset index
Y.reset_index(inplace=True)

# one-hot code diagnostic superclasses for multilabel problem
hot = MultiLabelBinarizer()
y_multi = hot.fit_transform(Y.diagnostic_superclass.values)

# train, test splits for data and multilabel response
x_train = X[~(Y.strat_fold == 10)]
y_train = y_multi[~(Y.strat_fold == 10)]
x_test = X[Y.strat_fold == 10]
y_test = y_multi[Y.strat_fold == 10]

timings['data_import'] = time.perf_counter() - start_time
current_time = time.perf_counter()

# prepare data for MiniRocket and sktime models
x_train_df = matrix_2_df(x_train)
x_test_df = matrix_2_df(x_test)

# extract features with MiniRocket
mini_mv = MiniRocketMultivariate().fit(x_train_df)
x_train_mr = mini_mv.transform(x_train_df)
x_test_mr = mini_mv.transform(x_test_df)

timings['MiniRocket_transform'] = time.perf_counter() - current_time
current_time = time.perf_counter()

# scale
scale = StandardScaler().fit(x_train_mr)
x_train = scale.transform(x_train_mr)
x_test = scale.transform(x_test_mr)

# run model
clf = OneVsRestClassifier(LogisticRegression(C=0.001, solver='saga', tol=1e-3, max_iter=3000), n_jobs=-2)
clf.fit(x_train, y_train)

timings['training_time'] = time.perf_counter() - current_time
current_time = time.perf_counter()

# predict
y_pred_train = clf.predict(x_train)
y_prob_train = clf.predict_proba(x_train)
y_pred_test = clf.predict(x_test)
y_prob_test = clf.predict_proba(x_test)

timings['prediction_time'] = time.perf_counter() - current_time
timings['total_time'] = time.perf_counter() - start_time

# score the model
labels = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
full_scores = {'ROC_AUC_train': roc_auc_score(y_train, y_prob_train),
               'ROC_AUC_test': roc_auc_score(y_test, y_prob_test),
               'Hamming_train': hamming_loss(y_train, y_pred_train),
               'Hamming_test': hamming_loss(y_test, y_pred_test),
               'parameters': clf.get_params()}

clf_report = classification_report(y_test, y_pred_test, output_dict=True, target_names=labels)
cnf_matrix = multilabel_confusion_matrix(y_test, y_pred_test)

# plots and tables
timing_df = pd.DataFrame(data=timings)
timing_df.to_csv(os.path.join(results_folder, 'timing.csv'))

all_data_score = pd.DataFrame(data=full_scores)
all_data_score.to_csv(os.path.join(results_folder, 'all_data_score.csv'))

clf_report_df = pd.DataFrame(data=clf_report)
clf_report_df.to_csv(os.path.join(results_folder, 'all_data_clf_report.csv'))

plot_ml_cnf(cnf_matrix, labels, title='Full Test Results', save='all_data_cnf_matrix', cmap='Blues')
multilabel_roc(y_test, y_prob_test, 'Full test ROC', labels, save='all_data_ROC')
