import pandas as pd
import get_data
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from wavelet_features import get_ecg_features
from utils import matrix_2_df, model_wrapper, filter_all

to_filter = False

if get_data.reduced:
    x_train = get_data.X_train
    x_train_meta = get_data.X_train_meta
    x_val = get_data.X_test
    x_val_meta = get_data.X_test_meta

else:
    x_train = get_data.X_train
    x_train_meta = get_data.X_train_meta
    x_val = get_data.X_val
    x_val_meta = get_data.X_val_meta

if to_filter:
    x_train = filter_all(x_train)
    x_val = filter_all(x_val)

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

# concat patient meta-data features with each of the above
X_train_metmr = pd.concat([X_train_mr, x_train_meta], axis=1)
X_val_metmr = pd.concat([X_val_mr, x_val_meta], axis=1)
X_train_metwv = pd.concat([X_train_wv, x_train_meta], axis=1)
X_val_metwv = pd.concat([X_val_wv, x_val_meta], axis=1)

# list of models, feel free to add more (for parameters see model_params dict
# in utils.py), //note : LogisticRegression did not converge, needs more iter
models = [SGDClassifier(max_iter=2000), LogisticRegression(max_iter=1000), GaussianNB(),
          KNeighborsClassifier(), RandomForestClassifier(), RidgeClassifier(),
          AdaBoostClassifier()]

# trial with Randomforest
raw_MR_meta = model_wrapper(models[4:5], X_train_metmr, get_data.y_train_multi, cat=['sex'], prefix='raw-MR-meta',
                            scoring="roc_auc_ovr", n_jobs=3)

#test = gridcv_all(SGDClassifier(max_iter=2000), X_train_metmr.columns, categorical=['sex'])