import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from utils import output_folder

# set this to retrieve all data (set as False) or only fold 9 (set as True)
reduced = False

data_path = os.path.abspath(r'..\Data')
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

# reduce data set to only the 9th fold if reduced
if reduced:
    Y = Y[Y.strat_fold == 9]

else:
    # reduce data to folds 9 and 10 only
    Y = Y[Y.strat_fold.isin([9, 10])]

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

# remove records with no age (11 records in fold 9)
X = X[~Y.age.isna()]
Y = Y[~Y.age.isna()]

# reset index
Y.reset_index(inplace=True)

if not reduced:
    # load in ecg_descriptors
    ecg_des = pd.read_csv(os.path.join(data_path, 'ecg_descriptors.csv'), index_col=0)

    # drop points with missing data
    Y = Y[~ecg_des.heartrate.isna()]
    X = X[~ecg_des.heartrate.isna()]
    ecg_des = ecg_des[~ecg_des.heartrate.isna()]
    Y = Y[~ecg_des.pNN20.isna()]
    X = X[~ecg_des.pNN20.isna()]
    ecg_des = ecg_des[~ecg_des.pNN20.isna()]

    # drop points with high heart rate
    Y = Y[ecg_des.heartrate < 300].reset_index(drop=True)
    X = X[ecg_des.heartrate < 300]
    ecg_des = ecg_des[ecg_des.heartrate < 300].reset_index(drop=True)

# one-hot code diagnostic superclasses for multilabel problem
hot = MultiLabelBinarizer()
y_multi = hot.fit_transform(Y.diagnostic_superclass.values)

if reduced:
    # label normal true/false
    Y['normal'] = Y.diagnostic_superclass.apply(lambda x: 'NORM' in x)
    y = Y['normal']

    # Split data into train and test
    y_train, y_test = train_test_split(y, test_size=.20, random_state=5)
    X_train_meta, X_test_meta = train_test_split(Y[['age', 'sex']], test_size=.20, random_state=5)

else:
    X_train_meta, X_test_meta, X_train_ecg, X_test_ecg = train_test_split(Y[['age', 'sex']], ecg_des,
                                                                          test_size=.20, random_state=5)
    X_train_ecg.reset_index(drop=True, inplace=True)
    X_test_ecg.reset_index(drop=True, inplace=True)

# train, test splits for data and multilabel response, meta data, ecg descriptors
X_train, X_test, y_train_multi, y_test_multi = train_test_split(X, y_multi, test_size=.20, random_state=5)


# save targets (only once)
# out_path = os.path.join(data_path, 'y_')
# np.save(out_path + 'train-final', y_train_multi)
# np.save(out_path + 'test-final', y_test_multi)
X_train_meta.reset_index(drop=True, inplace=True)
X_test_meta.reset_index(drop=True, inplace=True)

