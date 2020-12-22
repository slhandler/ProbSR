import pandas as pd
import numpy as np
import os
import joblib

from ProbSR.ml_lib.core.sample_methods import SampleTechniques
from ProbSR.ml_lib.core.split_data import DataSplitter
from ProbSR.ml_lib.utils.utils_funcs import (
    splitTargetFromLabels, removePredictor)

# config file storing all necessary inputs for script
import probsr_ml_config as config

# scikit learn imports
from skopt import BayesSearchCV
from skopt import gp_minimize, space
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from functools import partial
from sklearn.model_selection import cross_val_score
#----------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------#


def optimize_skopt(params, param_names, examples, targets, splits=3, scoring_func='roc_auc'):

    # make dict using two lists
    params = dict(zip(param_names, params))

    # define model
    model = RandomForestClassifier(criterion='entropy', class_weight='balanced',
                                   random_state=42, **params)

    # perform cross validation over my splits
    cv_scores = cross_val_score(
        model, examples, targets, cv=splits, scoring=scoring_func)

    # return the mean score
    return -1.0*np.mean(cv_scores)
#----------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------#


def preprocessDF(data0):

    # convert units of brightness temp to C from K
    if ('sat_irbt' in data0.columns):
        data0['sat_irbt'] -= 273.15

    # drop some columns if in DataFrame
    cols_to_drop = ['prev_RT', 'obs_subsfc_rt', 'precip', 'veg_type', 'time_day',
                    'Unnamed: 0.1.1', 'Unnamed: 0.1', 'Unnamed: 0']

    for column in cols_to_drop:
        if (column in data0.columns):
            data0, _ = removePredictor(data0, column, 1)

    # drop September
    data0 = data0[data0['month'] != 9]
    data0 = data0.reset_index(drop=True, inplace=False)

    # compute difference in soilT between 2 top layers
    data0['soilT_diff'] = data0['soilT_0_01m'] - data0['soilT_0_1m']

    # change was_precipitating field to be based on 1HRAD instead of 3HRAD
    data0['was_precipitating'] = 0
    iprecip = np.where(data0['1HRAD'] > 0.0)[0]
    data0.loc[iprecip, 'was_precipitating'] = 1

    # drop rows with nans
    data0 = data0.dropna(axis=0, how='any')
    data0 = data0.reset_index(drop=True, inplace=False)

    data0['cat_rt'].astype('int')

    return data0
#----------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------#
# get config params
predictor_columns = config.PREDICTOR_COLUMNS
target_column = config.TARGET_COLUMN
train_data_path = config.TRAINING_DATA_PATH
cal_data_path = config.CALIBRATION_DATA_PATH
test_data_path = config.TESTING_DATA_PATH
scoring_func = config.SCORING_FUNC
hyperparm_n_iter = config.HYPER_PARAM_N_ITER
output_path_file = config.RF_OUTPUT_PATH
column_dtypes = config.COLUMN_DTYPES
#----------------------------------------------------------------------------------------#

if __name__ == "__main__":

    print("Reading in training DataFrame")
    train_data = pd.read_csv(train_data_path, usecols=list(
        column_dtypes.keys()), dtype=column_dtypes)
    print(train_data.shape)

    # do a couple preprocess routines
    train_data = preprocessDF(train_data)
    print(train_data.shape)

    # make an instance of sampleTechniques class
    samplingObj = SampleTechniques(train_data, target_column)

    # get daily random samples for tuning/training
    train_data = samplingObj.make_specific_range_samples(
        (-8, 5), feature_to_sample='sfc_temp')
    print(train_data.shape)

    # get 50/50 samples of when it was precipitating
    train_data = samplingObj.smart_sample_by_feature(
        feature='was_precipitating')
    print(train_data.shape)

    # split predictors from label
    train_predictors, train_targets = splitTargetFromLabels(train_data,
                                                            target_column=target_column)

    # make an instance of dataSplitter class
    splitData = DataSplitter(train_predictors, train_targets, target_column)

    # get monthly splits
    cv_dict = splitData.splitByMonth()
    cv_splits = list(zip(cv_dict['train_indices'], cv_dict['test_indices']))

    # tune RF using Bayes over monthly cv splits
    #----------------------------------------------------------------------------------------#
    rf_hyperparameter_space = [
        space.Integer(2, 500, name='min_samples_leaf'),
        space.Integer(100, 500, name='n_estimators'),
        space.Integer(3, 25, name='max_depth'),
        space.Real(0.01, 1, prior='uniform', name='max_features'),
        space.Real(0.01, 1, prior='uniform', name='ccp_alpha')
    ]

    param_names = ['min_samples_leaf', 'n_estimators',
                   'max_depth', 'max_features', 'ccp_alpha']

    optimization_function = partial(
        optimize_skopt,
        param_names=param_names,
        examples=train_predictors[predictor_columns],
        targets=train_targets,
        splits=cv_splits,
        scoring_func=scoring_func
    )

    result = gp_minimize(
        optimization_function,
        dimensions=rf_hyperparameter_space,
        n_calls=hyperparm_n_iter,
        n_initial_points=10,
        verbose=10,
        n_jobs=3
    )

    best_param_dict = dict(zip(param_names, result.x))

    #----------------------------------------------------------------------------------------#

    # build best model
    rf_model = RandomForestClassifier(criterion='entropy', class_weight='balanced',
                                      random_state=42, **best_param_dict)
    print(rf_model)

    # train on whole training set
    print("Training the final model...")
    rf_model.fit(train_predictors[predictor_columns], train_targets)

    print("Saving model...")
    joblib.dump(
        rf_model, '/data/ScikitModels/RoadTemp/current/RF_ProbSR_Model.pkl')

    print("Reading in calibration DataFrame")
    calibrate_data = pd.read_csv(cal_data_path, usecols=list(
        column_dtypes.keys()), dtype=column_dtypes)
    print(calibrate_data.shape)

    # preprocess calibration data again
    calibrate_data = preprocessDF(calibrate_data)

    # make an instance of sampleTechniques class
    samplingObj = SampleTechniques(calibrate_data, target_column)

    # get daily random samples for tuning/training
    calibrate_data = samplingObj.make_specific_range_samples(
        (-8, 5), feature_to_sample='sfc_temp')
    print(calibrate_data.shape)

    # get 50/50 samples of when it was precipitating
    calibrate_data = samplingObj.smart_sample_by_feature(
        feature='was_precipitating')
    print(calibrate_data.shape)

    # split predictors from labels for calibration set
    calibration_predictors, calibration_targets = splitTargetFromLabels(calibrate_data,
                                                                        target_column=target_column)

    # train isotonic regression model
    print("Training an isotonic regression model...")
    input_predictions = rf_model.predict_proba(
        calibration_predictors[predictor_columns])[:, 1]
    isotonic_model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
    isotonic_model.fit(input_predictions, calibration_targets)

    print("Saving calibration model...")
    joblib.dump(isotonic_model,
                '/data/ScikitModels/RoadTemp/current/RF_ProbSR_Model_Isotonic.pkl')

    print("Reading in final evaluation DataFrame")
    testing_data = pd.read_csv(test_data_path, usecols=list(
        column_dtypes.keys()), dtype=column_dtypes)
    print(testing_data.shape)

    # preprocess calibration data again
    testing_data = preprocessDF(testing_data)

    # make an instance of sampleTechniques class
    samplingObj = SampleTechniques(testing_data, target_column)

    # get daily random samples for tuning/training
    testing_data = samplingObj.make_specific_range_samples(
        (-8, 5), feature_to_sample='sfc_temp')
    print(testing_data.shape)

    # get 50/50 samples of when it was precipitating
    testing_data = samplingObj.smart_sample_by_feature(
        feature='was_precipitating')
    print(testing_data.shape)

    # split predictors from labels for testing set
    testing_predictors, testing_targets = splitTargetFromLabels(testing_data,
                                                                target_column=target_column)

    # evaluate final model using the test set
    print("Evaluating test set...")
    raw_probs = rf_model.predict_proba(
        testing_predictors[predictor_columns])[:, 1]
    calibrated_probs = isotonic_model.predict(raw_probs)

    # fix bad predictions
    infin = np.where(~np.isfinite(calibrated_probs))
    if (len(infin[0]) > 0):
        calibrated_probs[infin] = 0.0

    # assign probabilities to a csv file to pefrom future analysis with
    testing_data.loc[:, f'calibrated_prob_{target_column}'] = calibrated_probs
    testing_data.loc[:, f'prob_{target_column}'] = raw_probs

    # write out for graphical analysis
    testing_data.to_csv(output_path_file)
    print("DONE!")
