"""Optuna optimization of hyperparameters.

Implements optuna optimization algorithms using sklearn ML algorithms (currently,
GradientBoostingClassifier.) auROC, auPRC or accuracy between datasets can be
used as the optimization objective. Returns an optuna study class.

  Typical usage examples:
    python3 run_ML_diff_dataset.py
    python3 run_ML_diff_dataset.py --scoring_metric ROC
"""
from joblib import dump, load
import optuna
import pandas as pd
import faiss
import pickle

from iterative_stratification import StratifiedGroupKFold
from optuna_via_sklearn.config import *
from optuna_via_sklearn.generate_prediction_probs import *
from optuna_via_sklearn.load_data import *
from optuna_via_sklearn.RandomClassifier import RandomClassifier
from optuna_via_sklearn.specify_sklearn_models import train_model, train_and_score_model, score_model
from optuna_via_sklearn.WeightedRandomClassifier import WeightedRandomClassifier
from run_ML_same_dataset import optimize_hyperparams
from os import system
from os.path import exists
from pprint import pprint
from run_ML_args import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import itertools
import optuna_via_sklearn.specify_sklearn_models
import warnings



def main():
    args = parse_run_optuna_args()
    run_experiment(args)


def run_experiment(args):

    verify_optuna_args(args)
    
    # Define prefix for all files produced by run
    # Check if optuna-trained model already exists
    if args.feature_alias is not None:
        data_model_path =  f"{args.results_folder}/{args.data_alias}_{args.feature_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.scoring_metric}_model.joblib"
        train_model_path =  f"{args.results_folder}/{args.training_alias}_{args.feature_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.train_scoring_metric}_model.joblib"
    else:
        data_model_path =  f"{args.results_folder}/{args.data_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.scoring_metric}_model.joblib"
        train_model_path =  f"{args.results_folder}/{args.training_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.train_scoring_metric}_model.joblib"
    datasets = None
    
    if exists(train_model_path):
        # Load model
        print("Loading model at: " + train_model_path)
        try:
            (best_classifier) = load(train_model_path)
        except:
            if os.path.exists(train_model_path):
                os.remove(train_model_path)
                raise Exception(f"Deleting {train_model_path} since it could not be loaded!")
            else:
                raise Exception(f"{train_model_path} does not exist!")
    else:
        # Load training/testing data
        config = DataSpecification(args)

        if exists(data_model_path):
            args.data_path = None

        datasets, metadata = load_data(config)

        # Optimize hyperparameters with optuna
        if exists(data_model_path):
            print("Loading model at: " + data_model_path)
            print(load(data_model_path))
            (best_params, score_list) = load(data_model_path)
        else:
            print("Creating model at: " + data_model_path)
            print("Running optuna optimization.")
            fast_check_for_repeating_rows(datasets["training"].features)
            (best_params, score_list) = optimize_hyperparams(datasets["data"], args.scoring_metric, args.n, args.model_name, n_splits=5, n_jobs=args.num_jobs)
            print(f"{(best_params, score_list)=}")
            dump((best_params, score_list), data_model_path)
        best_index = np.argmin(score_list)
        best_param = best_params[best_index]

        print(f"Training on {args.training_alias} and saving to {train_model_path}")
        best_classifier = train_model(args.model_name, best_param, datasets["training"].features, datasets["training"].labels, save = train_model_path)


    # print(f"The best parameters are {best_params}")

    if not os.path.exists(args.result_file):
        result_dict = dict()
    else:
        with open(args.result_file, 'rb') as f:
            result_dict = pickle.load(f)

    result_key = (args.training_alias, args.testing_alias, args.model_name, args.train_scoring_metric, args.test_scoring_metric, args.feature_alias)
    
    if result_key not in result_dict:
        if datasets is None:
            if exists(data_model_path):
                args.data_path = None
            config = DataSpecification(args)
            datasets, metadata = load_data(config)

        if '_bg' in args.test_scoring_metric:
            score_list = train_and_score_model(args.model_name, best_classifier.get_params(), datasets["training"], datasets["testing"], args.test_scoring_metric)
            result_dict[result_key] = score_list
            final_score = score_list
        else:
            final_score = score_model(best_classifier, datasets["testing"], args.test_scoring_metric, args.model_name)
            result_dict[result_key] = final_score
        
        with open(args.result_file, 'wb') as f:
            pickle.dump(result_dict, f)
    else:
        final_score = result_dict[result_key]

    print(f"Final {args.test_scoring_metric} is {final_score=}")

    # print(f"{result_dict=}")
    print()
    print()
    print()

if __name__ == "__main__":
    main()
    # import cProfile
    # import pstats

    # with cProfile.Profile() as pr:
    #     main()

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # # stats.print_stats()
    # stats.dump_stats(filename='needs_profiling.prof')
