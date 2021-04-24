import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = os.path.join(r'..\Data', '')
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# reduce data set to only the 9th fold
Y = Y[Y.strat_fold == 9]

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
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

# one-hot code diagnostic superclasses for multilabel problem
hot = MultiLabelBinarizer()
y_multi = hot.fit_transform(Y.diagnostic_superclass.values)

# label normal true/false
Y['normal'] = Y.diagnostic_superclass.apply(lambda x: 'NORM' in x)
y = Y['normal']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=5)

# Same train test split but for multilabel problem
y_train_multi, y_test_multi = train_test_split(y_multi, test_size=.20, random_state=5)
