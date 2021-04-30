import pandas as pd
import get_data
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from wavelet_features import get_ecg_features
from utils import matrix_2_df, model_wrapper, filter_all, extract_all_features, output_folder
import os

to_filter = False

# load in data to local pointers
x_train = get_data.X_train
x_train_meta = get_data.X_train_meta
x_val = get_data.X_test
x_val_meta = get_data.X_test_meta

if to_filter:
    x_train = filter_all(x_train)
    x_val = filter_all(x_val)

#     # # extract Dr.Ojas features
#     # X_train_ecg = extract_all_features(x_train)
#     # X_val_ecg = extract_all_features(x_val)
#
# else:
#     # filter only for ECG descriptors
#     X_train_ecg = extract_all_features(filter_all(x_train))
#     X_val_ecg = extract_all_features(filter_all(x_val))

# extract features with wavelet transform
X_train_wv = pd.DataFrame(data=get_ecg_features(x_train))
X_val_wv = pd.DataFrame(data=get_ecg_features(x_val))

# prepare data for MiniRocket and sktime models
X_train_df = matrix_2_df(x_train)
X_val_df = matrix_2_df(x_val)

# extract features with MiniRocket
mini_mv = MiniRocketMultivariate().fit(X_train_df)
X_train_mr = mini_mv.transform(X_train_df)
X_val_mr = mini_mv.transform(X_val_df)

# rename columns
X_train_mr.columns = ['f_' + str(i) for i in X_train_mr.columns]
X_val_mr.columns = ['f_' + str(i) for i in X_val_mr.columns]
X_train_wv.columns = ['f_' + str(i) for i in X_train_wv.columns]
X_val_wv.columns = ['f_' + str(i) for i in X_val_wv.columns]

feature_data = {'MR': {'train': X_train_mr, 'test': X_val_mr}, 'wavelet': {'train': X_train_wv, 'test': X_val_mr},
                'meta': {'train': x_train_meta, 'test': x_val_meta}}


def feature_mix(to_mix):
    """
    Combines data from different feature extractions into a single dataframe given a list of keys to combine from the
    feature_data dictionary.

    *args
    - to_mix:
        list of keys from the feature_data dictionary to combine

    Returns:
        train (dataframe of combined features for training), test (dataframe of combined features for testing)
    """
    train = pd.concat([feature_data[i]['train'] for i in to_mix], axis=1)
    test = pd.concat([feature_data[i]['test'] for i in to_mix], axis=1)
    return train, test


# list of models, feel free to add more (for parameters see model_params dict
# in utils.py), //note : LogisticRegression did not converge, needs more iter
models = [SGDClassifier(loss='log', max_iter=4000), SGDClassifier() GaussianNB(),
          KNeighborsClassifier(), RandomForestClassifier(),
          AdaBoostClassifier()]

# set up models below here ============================================================================================
mixes = [['MR', 'meta'], ['wavelet', 'meta']]
fit_models = {}
pref = 'raw01-'

for mix in mixes:
    X_train, X_test = feature_mix(mix)
    out_loc = os.path.join(output_folder, pref + '-'.join(mix))
    X_train.to_hdf(out_loc + '-train.h5', key='train', mode='w')
    X_test.to_hdf(out_loc + '-test.h5', key='test', mode='w')
    fit_models['-'.join(mix)] = model_wrapper(models, X_train, get_data.y_train_multi, cat=['sex'],
                                              prefix=pref + '-'.join(mix), scoring='roc_auc_ovr',
                                              n_jobs=-3, cv=5)

# # concat patient meta-data features with each of the above
# X_train_metmr = pd.concat([X_train_mr, x_train_meta], axis=1)
# X_val_metmr = pd.concat([X_val_mr, x_val_meta], axis=1)
# X_train_metwv = pd.concat([X_train_wv, x_train_meta], axis=1)
# X_val_metwv = pd.concat([X_val_wv, x_val_meta], axis=1)
#
# # trials
# raw_MR_meta = model_wrapper(models[2:5], X_train_metmr, get_data.y_train_multi, cat=['sex'], prefix='raw-MR-meta',
#                             scoring="roc_auc_ovr", n_jobs=3)
#
# raw_wv_meta = model_wrapper(models[2:5], X_train_metwv, get_data.y_train_multi, cat=['sex'], prefix='raw-wv-meta',
#                             scoring="roc_auc_ovr", n_jobs=3)
#
# raw_MR = model_wrapper(models[2:5], X_train_mr, get_data.y_train_multi, prefix='raw-MR',
#                        scoring="roc_auc_ovr", n_jobs=3)

# test = gridcv_all(SGDClassifier(max_iter=2000), X_train_metmr.columns, categorical=['sex'])

# loaded_model = load(path)

# y_pred = loaded_model.best_estimator_.predict_proba_(X_val_metmr)

# classification_report(get_data.y_test_multi, y_pred)
