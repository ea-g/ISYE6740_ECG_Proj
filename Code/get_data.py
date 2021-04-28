import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# set this to retrieve all data or only fold 9
reduced = True

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

# one-hot code diagnostic superclasses for multilabel problem
hot = MultiLabelBinarizer()
y_multi = hot.fit_transform(Y.diagnostic_superclass.values)

if reduced:
    # label normal true/false
    Y['normal'] = Y.diagnostic_superclass.apply(lambda x: 'NORM' in x)
    y = Y['normal']

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=5)

    # Same train test split but for multilabel problem
    X_train_meta, X_test_meta, y_train_multi, y_test_multi = train_test_split(Y[['age', 'sex']], y_multi,
                                                                              test_size=.20, random_state=5)
    X_train_meta.reset_index(drop=True, inplace=True)
    X_test_meta.reset_index(drop=True, inplace=True)

else:
    # train, validation, test splits for full data
    X_test = X[Y.strat_fold == 10]
    X_val = X[Y.strat_fold == 9]
    X_train = X[Y.strat_fold.isin(range(1, 9))]

    y_test = y_multi[Y.strat_fold == 10]
    y_val = y_multi[Y.strat_fold == 9]
    y_train = y_multi[Y.strat_fold.isin(range(1, 9))]

    X_test_meta = Y[Y.strat_fold == 10][['age', 'sex']].reset_index(drop=True)
    X_val_meta = Y[Y.strat_fold == 9][['age', 'sex']].reset_index(drop=True)
    X_train_meta = Y[Y.strat_fold.isin(range(1, 9))][['age', 'sex']].reset_index(drop=True)




