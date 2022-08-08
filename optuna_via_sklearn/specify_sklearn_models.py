import optuna
import pandas as pd
import faiss
import pickle
import time

import numpy as np
from joblib import dump, load
from optuna_via_sklearn.FrequentClassifier import FrequentClassifier
from optuna_via_sklearn.load_data import Dataset
from optuna_via_sklearn.load_data import fast_check_for_repeating_rows
from optuna_via_sklearn.RandomClassifier import RandomClassifier
from optuna_via_sklearn.WeightedRandomClassifier import WeightedRandomClassifier
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def define_model(model_name, params):
    if model_name =="GB":
        return GradientBoostingClassifier(**params)
    elif model_name == "SVC" or model_name == "SVC_balanced":

        if 'probability' in params and params['probability'] != True:
            raise Exception(f"The probability in {params=} for SVC is set to False!")
        if 'probability' in params:
            return SVC(**params)
        else:
            return SVC(probability = True,**params)
    elif model_name == "NN":
        return MLPClassifier(**params)
    elif model_name == "Elastic":
        return ElasticNet(**params)
    elif model_name == "Linear":
        return LinearRegression(**params)
    elif model_name == "KNN":
        return KNeighborsClassifier(**params)
    elif model_name == "Random":
        return RandomClassifier(**params)
    elif model_name == "WeightedRandom":
        return WeightedRandomClassifier(**params)
    elif model_name == "Frequent":
        return FrequentClassifier(**params)
    else: 
       raise Exception("Model name not valid.")

def train_model(model_name, parameters, train_feats, train_labs, save = False):
    '''
    Creates a model that's train using the inputted data
    input:
    model_name (str) - Name of the classifier to be trained
    parameters (dict: str -> data) - Dictionary that maps string representations of hyperperameter names to their values
    train_feats (np array) - numpy array of the training data
    train_labs (np array) - numpy array of the training labels
    save (bool) - Whether or not to save the trained model
    '''

    # if isinstance(train_feats, np.ndarray):
    #     raise Exception(f"train_feats must be of type numpy.ndarray but is instead {type(train_feats)}!")

    # if isinstance(train_labs, np.ndarray):
    #     raise Exception(f"train_labs must be of type numpy.ndarray but is instead {type(train_labs)}!")

    
    # Define model type
    classifier = define_model(model_name, parameters)
    # Train model
    classifier.fit(train_feats, train_labs)
    # Save model
    if save != False:
        dump(classifier, save)
    return classifier

def objective(trial, dataset, index_list, metric,  model_name, params = None):

    if params is not None and not isinstance(params, dict):
        raise Exception(f"params should be None or a dict but is instead {type(params)}!")

    fast_check_for_repeating_rows(dataset.features)

    # define model
    if model_name == "GB" and params is None:
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 1024, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 25), # make min larger 1--> 5?
            "random_state": 7
            }
                ## Unaltered default params
                    #loss='deviance', learning_rate=0.1, subsample=1.0, criterion='friedman_mse', min_weight_fraction_leaf=0.0,
                    #min_impurity_split=None, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0
                    #min_weight_fraction_leaf=0.0
    elif model_name == "SVC" and params is None:
        params = {
            "C" : trial.suggest_float("C", 1e-3, 1),
            "kernel" : trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "degree" : trial.suggest_int("degree", 1, 10),
            "gamma"  : trial.suggest_categorical("gamma", ["scale", "auto"]),
            "random_state": 7
        }
            ## Unaltered default params
                #degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)[source]¶
    elif model_name == "SVC_balanced" and params is None:
        params = {
            "C" : trial.suggest_float("C", 1e-3, 1),
            "kernel" : trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "degree" : trial.suggest_int("degree", 1, 10),
            "gamma"  : trial.suggest_categorical("gamma", ["scale", "auto"]),
    	    "random_state": 7,
            "class_weight": "balanced"
        }
    elif model_name == "NN" and params is None:
        params = {
            "hidden_layer_sizes" : (trial.suggest_int("hidden_layer_sizes", 100, 1000)),
            "activation": trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
            "alpha" : trial.suggest_float("alpha", 1e-6, 1e-0, log=True),
            "learning_rate" : trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
            "random_state": 7
        }
            ## Unaltered default params
                #activation='relu', solver='adam',, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000
    elif model_name == "Elastic" and params is None:
        params = {
            "l1_ratio":trial.suggest_float("l1_ratio", 0, 1),
            "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
            "random_state": 7
        }
    elif model_name == "Linear" and params is None:
        params = {
            
        }
    elif model_name == "KNN" and params is None:
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 11),
            "weights" : trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": trial.suggest_int("leaf_size", 1, 100),
            "p": trial.suggest_int("p", 1, 10)
        }
    elif model_name == "Random" and params is None:
        params = {
            "random_state": 7
        }
    elif model_name == "WeightedRandom" and params is None:
        params = {
            "random_state": 7
        }
    elif model_name == "Frequent" and params is None:
        params = {
        
        }
    elif params is None:
        raise Exception(f"Model name({model_name}) not valid.")
    
    score_list = []

    start = time.time()

    # train and evaluate models
    for i in range(len(index_list)):

        testing_index = index_list[i]
        training_index = []
        for j in range(len(index_list)):
            if j == i: continue
            training_index.extend(index_list[j])

        # Training and testing data should be distinct
        if len(set(training_index) & set(testing_index)) != 0:
            raise Exception(f"{set(training_index) & set(testing_index)} are in {training_index=} and {testing_index=}!")

        X_train = np.take(dataset.features, training_index, axis=0)
        y_train = np.take(dataset.labels, training_index, axis=0)

        X_test = np.take(dataset.features, testing_index, axis=0)
        y_test = np.take(dataset.labels, testing_index, axis=0)


        training_data = Dataset(input_df=None, features=X_train, labels=y_train)
        testing_data = Dataset(input_df=None, features=X_test, labels=y_test)

        fast_check_for_repeating_rows(X_train, X_test)




        score = train_and_score_model(params, training_data, testing_data, metric, model_name)
        score_list.append(score)

    end = time.time()
    return (np.mean(score_list), end - start)




def train_and_score_model(parameters, training_data, testing_data, metric, model_name):

    if training_data.labels.ndim != 1:
        raise Exception("The training data's labels should be 1D!")
    if testing_data.labels.ndim != 1:
        raise Exception("The testing data's labels should be 1D!")

    fast_check_for_repeating_rows(training_data.features, testing_data.features)

    if training_data.features.shape == testing_data.features.shape and np.amax(training_data.features - testing_data.features) == 0:
        raise Exception("Training data is the same as the testing {training_data=} {testing_data=}!")
    
    if metric == "f-measure":
        if 0 not in training_data.labels:
            raise Exception(f"No 0s in {training_data.labels}!")
        if 1 not in training_data.labels:
            raise Exception(f"No 1s in {training_data.labels}!")
        # We know 0 and 1 are in training_data.labels so if it has anything else, its size will be greater than 2
        if len(set(training_data.labels)) > 2:
            raise Exception(f"The training labels {set(training_data.labels)=} has something besides 0s and 1s!")
    
    # Generate prediction probs for test set
    classifier = train_model(model_name, parameters, training_data.features, training_data.labels)
    return score_model(classifier, testing_data, metric, model_name)



def score_model(classifier, testing_data, metric, model_name):
    if metric == "f-measure" or metric == "accuracy":
        # We must have 0 and 1 are out only outputs
        y_score = classifier.predict(testing_data.features)
        y_score = np.heaviside(y_score - 0.5, 0)
        y_score = np.round(y_score)
        y_score = y_score.astype(int)
    elif metric == "spearman":
        # We need the continous values for spearman
        if model_name == "Linear" or model_name == "Elastic":
            y_score = classifier.predict(testing_data.features)
        else:
            y_score = classifier.predict_proba(testing_data.features)[:,1]
    else:
    # In this case, the metric isn't f-measure or accuracy so we need the continous outputs of the classifier
        if model_name == "Linear" or model_name == "Elastic":
            y_score = classifier.predict(testing_data.features)
            y_score = np.maximum(y_score, 0)
            y_score = np.minimum(y_score, 1)
        else:
            y_score = classifier.predict_proba(testing_data.features)[:,1]


    if metric == "f-measure":
        score_set = set(y_score.flatten())
        if 0 not in score_set and 1 not in score_set:
            raise Exception(f"Invalid values in the labels: {score_set}!")
        # We know 0 and 1 are in score_set so if it has anything else, its size will be greater than 2
        if len(score_set) > 2:
            raise Exception(f"{score_set=} has something besides 0s and 1s!")
    
    # Generate scoring metric
    if metric == "auPRC": # Calculate auPRC
        precision, recall, thresholds = precision_recall_curve(testing_data.labels, y_score)
        score = auc(recall, precision)
    elif metric == "auROC": # Calculate auROC
        score = roc_auc_score(testing_data.labels, y_score)
    # elif metric == "auROC_bygene": # Calculate by-gene auROC
    #     input_df["ref"]["d1"].groupby('protein_id').apply(lambda x: roc_auc_score(x.label, y_score))
    elif metric == "accuracy": # Calculate mean accuracy
        score = accuracy_score(testing_data.labels, y_score)
    elif metric == "f-measure":
        score = f1_score(testing_data.labels, y_score)
    elif metric == "spearman":
        # print(f"{testing_data.labels=}, {y_score=}")
        # print(f"{stats.rankdata(testing_data.labels)=} , {stats.rankdata(y_score)=}")
        score = abs(stats.spearmanr( stats.rankdata(testing_data.labels) , stats.rankdata(y_score) ).correlation)
    else:
        raise Exception(f"Unknown metric: {metric}! ")
    return(score)

