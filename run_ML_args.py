"""Optuna optimization of hyperparameters: argument processing module

Processes arguments for each script in the optuna_training library. Returns an
object of class argparse.

  Typical usage examples:
    from run_optuna_args import *
    args = parse_run_optuna_args()
"""
import argparse
import os
from pathlib import Path


def parse_run_optuna_args():
    parser = argparse.ArgumentParser(description="Optuna optimization of hyperparameters.")
    # Define output folder
    parser.add_argument("--results-folder", type=str, default = "optuna_via_sklearn/results/",
                            help="Write path to folder of results. Must end in '/'. ")

    # Optuna parameters
    parser.add_argument("--model-name", type = str, default = "GB",
                        choices = ["GB", "SVC", "SVC_balanced", "NN", "KNN", "GNB", "DT", "Logistic", "Random", "WeightedRandom", "Frequent"],
                        help="Name of Machine Learning algorithm.")
    parser.add_argument("--scoring-metric", type=str, default=None,
                        choices = ["auPRC", "auROC", "accuracy", "f-measure", "spearman"],
                        help="Metric used to score the models. ROC, PR, accuracy, f-measure or spearman.")
    parser.add_argument("--prediction-col", type=str, default="label",
                        help="The column in the datasets that we need to predict. By default, it's set to 'label'")
    
    parser.add_argument("--train-prediction-col", type=str, default="label",
                        help="The column in the training dataset that we need to predict. By default, it's set to 'label'")
    parser.add_argument("--train-scoring-metric", type=str, default= "auROC",
                        choices = ["auPRC", "auROC", "accuracy", "f-measure", "spearman"],
                        help="Metric used to score the models on the training data. ROC, PR, accuracy, f-measure or spearman.")
    parser.add_argument("--test-prediction-col", type=str, default="label",
                        help="The column in the testing dataset that we need to predict. By default, it's set to 'label'")
    parser.add_argument("--test-scoring-metric", type=str, default= "auROC",
                        choices = ["auPRC", "auPRC_bg", "auROC", "auROC_bg", "accuracy", "f-measure", "f-measure_bg", "spearman"],
                        help="Metric used to score the models on the testing data. ROC, PR, accuracy, f-measure or spearman.")

    parser.add_argument("--n", type=int, default=200, help="Number of models for oputuna to train.")
    parser.add_argument("--num-jobs", "-j", type=int, default=1,
                        help="Number of parallel jobs that you want Optuna to run while hyperparameter searching")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Number of seconds that an optuna study can run before it is killed. Must be a non-negative number")
    
    # Data specification parameters
    parser.add_argument("--training-path", type=str, help="Path to training data.")
    parser.add_argument("--training-alias", type=str, help="Name to give to training data.")
    parser.add_argument("--training-start", type=int, help="Index of column containing first feature.")

    parser.add_argument("--testing-path", type=str, help="Path to testing data.")
    parser.add_argument("--testing-alias", type=str, help="Name to give to testing data.")
    parser.add_argument("--testing-start", type=int, help="Index of column containing first feature.")

    parser.add_argument("--data-path", type=str, help="Path to data.")
    parser.add_argument("--data-alias", type=str, help="Name to give to data.")
    parser.add_argument("--data-start", type=int, help="Index of column containing first feature.")

    parser.add_argument("--result-file", type=str, help="Name of the result file", default='result.pkl')

    parser.add_argument("--feature-list", nargs='*', help="The columns from the features that we'll take. Uses 0-based indices and is optional.")
    parser.add_argument("--feature-alias", type=str, default=None, help="The name for the selected features")

    parser.add_argument("--feature-type", type=str, default= "mut",
                        help="Mapping of aa representation between mutant and reference.")
    parser.add_argument("--lang_model_type", type=str, default = "lang_model_type", #choices = ["UniRep", "Rostlab_Bert", "other"],
                        help="Type of language model underlying features.")
    parser.add_argument("--pca-key", type = str, default = "None", help="PCA matrix specified by key in pca_mats. See config file for further specifications.")
#    parser.add_argument("--split", type = str, default = "None", help="Number of folds to split crossvalidation data into.")

    
    parser.add_argument('--pkl', action=argparse.BooleanOptionalAction)
    parser.set_defaults(pkl=False)
    args = parser.parse_args()
    return(args)


def verify_optuna_args(args):

    if args.data_path is None and args.training_path is None:
        raise Exception(f"Either the --data-path or --training-path need to be set!")

    if args.training_path is not None and args.testing_path is None:
        raise Exception(f"The --testing-path needs to be set if --training-path is specified!")

    if args.testing_path is not None and args.training_path is None:
        raise Exception(f"The --training-path needs to be set if --testing-path is specified!")

    # By default training/testing alias is the same as the path
    if args.training_alias is None and args.training_path is not None:
        args.training_alias = args.training_path

    if args.data_alias is None and args.data_path is not None:
        args.data_alias = args.data_path

    # Verify the existance of the training/testing files
    if args.data_path is not None and not os.path.exists(args.data_path):
        raise Exception(f"The training file {args.data_path} does not exist!")

    if args.training_path is not None and not os.path.exists(args.training_path):
        raise Exception(f"The training file {args.training_path} does not exist!")

    if args.num_jobs < 1 and args.num_jobs != -1:
        raise Exception(f"The number of jobs({args.num_jobs}) must be -1 or a positive integer!")

    if args.timeout is not None and args.timeout <= 0:
        raise Exception(f"The timeout is {args.timeout} but must be positive!")

    filen, file_ext = os.path.splitext(args.result_file)
    if file_ext != '.pkl':
        raise Exception(f"The result file({args.result_file}) must be a .pkl file!")

    # Create the results folder if it doesn't exist
    Path(args.results_folder).mkdir(parents=True, exist_ok=True)

    if args.prediction_col is None:
        raise Exception(f"prediction-col is not specified! Specify it using the prediction-col flag")


    if args.scoring_metric is None:
        if args.train_scoring_metric is not None:
            args.scoring_metric = args.train_scoring_metric
        elif args.test_scoring_metric is not None:
            args.scoring_metric = args.test_scoring_metric
        else:
            raise Exception(f"scoring-metric is not specified! Specify it using the scoring-metric flag")

    if args.feature_list is not None:
        fl = args.feature_list

        col_list = []
        for f in fl:
            if '-' in f:
                split_f = f.split('-')
                try:
                    a = int(split_f[0])
                    b = int(split_f[1])
                except Exception as e:
                    raise Exception(f"Invalid format {f}! Must be in the form a-b where a and b are non-negative integers and a <= b")
                if a > b:
                    raise Exception(f"Invalid format {f}! Must be in the form a-b where a and b are non-negative integers and a <= b")
                col_list.extend([i for i in range(a, b + 1)])
            elif ',' in f:
                split_f = f.split(',')
                try:
                    for i in split_f:
                        int_i = int(i)
                        if int_i < 0:
                            raise Exception()
                        col_list.append(int_i)
                except Exception as e:
                    raise Exception(f"Invalid format {f}! Must be a list of non-negative integers separated by comma(s)")
            else:
                try:
                    int_i = int(f)
                    if int_i < 0:
                        raise Exception()
                    col_list.append(int_i)
                except Exception as e:
                    raise Exception(f"Invalid format {f}! Must be a non-negative integer")          

        args.feature_list = col_list


    
if __name__ == "__main__":

    args = parse_run_optuna_args()

    verify_optuna_args(args)

    print(args)
    
