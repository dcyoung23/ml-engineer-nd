import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


# List for keys that will be used to exclude in column renaming
KEYS = ['SK_ID_CURR', 'SK_BUREAU_ID']


# Create binary from flag = Y
def convert_flag(df, flag_dict):
    for k, v in flag_dict.items():
        try:
            df[v] = df[k].apply(lambda x: 1 if x == 'Y' else 0)
        except KeyError:
            print('Key Error creating flag field')


# Create a function to drop columns from a list
def drop_cols(df, col_list):
    for column in col_list:
        try:
            df.drop(column, axis=1, inplace=True)
        except ValueError:
            continue


# Create dummy features function
def create_dummy(df, cols):
    # Dummy df
    dummy = pd.get_dummies(df[cols])

    # Dummy column names to list
    dummy_cols = list(dummy)

    # To avoid concat duplicate dummy columns drop all dummy columns first
    drop_cols(df, dummy_cols)

    # Add new dummy features
    df_out = pd.concat([df, dummy], axis=1)

    print 'Dummy columns created!'

    return df_out, dummy_cols


# Rename df columns
def rename_cols_df(df, cols, suffix):
    # Do not rename columns that are keys
    cols = [c for c in cols if c not in KEYS]

    # If a suffix is provided
    if suffix:
        rename_dict = dict((k, k.replace(' ', '_').upper() + '_' +
                           suffix.upper()) for k in cols)
    # Else clean spaces and convert to upper
    else:
        rename_dict = dict((k, k.replace(' ', '_').upper()) for k in cols)

    # Rename columns
    df.rename(columns=rename_dict, inplace=True)

    return df


# Custom function for different aggregates and appending suffix to column names
def perform_aggregate(source, group, agg):
    if agg == 'sum':
        output = pd.DataFrame(source.groupby(group).sum()).reset_index()
    elif agg == 'avg':
        output = pd.DataFrame(source.groupby(group).mean()).reset_index()
    elif agg == 'min':
        output = pd.DataFrame(source.groupby(group).min()).reset_index()
    elif agg == 'max':
        output = pd.DataFrame(source.groupby(group).max()).reset_index()

    # Get columns in output excluding groupby columns
    cols = [c for c in list(output) if c not in group]

    # Fill aggregate NaN with zero
    output[cols] = output[cols].fillna(0)

    # Rename columns
    output = rename_cols_df(output, cols, agg)

    return output


# Custom function to merge multiple dfs in one action
def merge_df(merge_dict):
    # Store output dfs in list for multiple merges
    df_list = []
    # Loop through dict and merge dfs
    for i in range(len(merge_dict)):
        if i > 0:
            # if more than 1 item in merge_dict then left is the prior output
            output = df_list[i - 1].merge(merge_dict[i]['right'].
                                          reset_index(drop=True),
                                          how=merge_dict[i]['how'],
                                          on=merge_dict[i]['on'])
        else:
            output = merge_dict[i]['left'].merge(merge_dict[i]['right'].
                                                 reset_index(drop=True),
                                                 how=merge_dict[i]['how'],
                                                 on=merge_dict[i]['on'])

        # Fill right NaN columns with zero
        output[list(merge_dict[i]['right'])] = output[list(merge_dict[i]['right'])].fillna(0)

        # Append to list
        df_list.append(output)

    return output


def predict_missing_ext(column):
    ext_drop_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                     'SK_ID_CURR', 'TARGET', 'SOURCE']

    # Create training for rows that are not null for column
    features_train_ext = features_transformed[features_transformed[column].notnull()].copy()

    # Create labels
    labels_train_ext = features_train_ext[column]

    # Drop columns
    drop_cols(features_train_ext, ext_drop_cols)

    # Create RandomForestRegressor
    regr = RandomForestRegressor(n_estimators=10, max_depth=5,
                                 max_features='sqrt', random_state=SEED)

    # Fit model
    regr.fit(features_train_ext, labels_train_ext)

    # Create test for rows that are null for column
    features_test_ext = features_transformed[features_transformed[column].isnull()].copy()

    # Set aside SK_ID_CURR for mapping on prediction updates
    id_ext = features_test_ext['SK_ID_CURR']

    # Drop columns
    drop_cols(features_test_ext, ext_drop_cols)

    # Make predictions
    pred_ext = regr.predict(features_test_ext)

    # Create update df
    update_ext = pd.DataFrame({'SK_ID_CURR': id_ext, column + '_PRED': pred_ext})

    return update_ext


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):

    results = {}

    # Get start time
    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    # Get end time
    end = time()

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train)
    # Get start time
    start = time()
    predictions_test = learner.predict_proba(X_test)
    predictions_train = learner.predict_proba(X_train[:300])
    # Get end time
    end = time()

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute ROC AUC Score on the the first 300 training samples
    results['roc_auc_score_train'] = roc_auc_score(y_train[:300],
                                                   predictions_train[:, 1])

    # Compute ROC AUC Score on the test set which is y_test
    results['roc_auc_score_test'] = roc_auc_score(y_test,
                                                  predictions_test[:, 1])

    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__,
                                             sample_size))

    # Return the results
    return results


def cv_result_analysis(input_params, train, param, values, nfold=5,
                       num_boost_round=5, early_stopping_rounds=None,
                       verbose_eval=True):
    # Loop through values
    for i in values:
        xgb_params = input_params
        xgb_params[param] = i
        print(param + ' cv results for: ')
        print(xgb_params)
        cv_result = xgb.cv(xgb_params, train, nfold=nfold, metrics='auc',
                           num_boost_round=num_boost_round,
                           early_stopping_rounds=early_stopping_rounds,
                           verbose_eval=verbose_eval)
    return cv_result


# Create Submission function
def create_submission(id_test, y_scores):

    # Create submission df
    submission = pd.DataFrame({'SK_ID_CURR': id_test, 'TARGET': y_scores})

    return submission


# Create csv output function
def create_output(submission, output):

    # Create submission csv
    submission.to_csv(output, index=False)
