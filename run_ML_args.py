"""Optuna optimization of hyperparameters: argument processing module

Processes arguments for each script in the optuna_training library. Returns an
object of class argparse.

  Typical usage examples:
    from run_optuna_args import *
    args = parse_run_optuna_args()
"""
import argparse
import os

def parse_run_optuna_args():
    parser = argparse.ArgumentParser(description="Optuna optimization of hyperparameters.")
    # Define output folder
    parser.add_argument("--results_folder", type=str, default = "optuna_via_sklearn/results/",
                            help="Write path to folder of results. Must end in '/'. ")

    # Optuna parameters
    parser.add_argument("--model_name", type = str, default = "GB",
                        choices = ["GB", "SVC", "SVC_balanced", "NN", "KNN", "Elastic", "Linear", "Random", "WeightedRandom", "Frequent"],
                        help="Name of Machine Learning algorithm.")
    parser.add_argument("--scoring_metric", type=str, default=None,
                        choices = ["auPRC", "auROC", "accuracy", "f-measure", "spearman"],
                        help="Metric used to score the models. ROC, PR, accuracy, f-measure or spearman.")
    parser.add_argument("--prediction-col", type=str, default=None,
                        help="The column in the datasets that we need to predict. By default, it's set to 'label'")
    
    parser.add_argument("--train-prediction-col", type=str, default="label",
                        help="The column in the training dataset that we need to predict. By default, it's set to 'label'")
    parser.add_argument("--train-scoring-metric", type=str, default= "auROC",
                        choices = ["auPRC", "auROC", "accuracy", "f-measure", "spearman"],
                        help="Metric used to score the models on the training data. ROC, PR, accuracy, f-measure or spearman.")
    parser.add_argument("--test-prediction-col", type=str, default="label",
                        help="The column in the testing dataset that we need to predict. By default, it's set to 'label'")
    parser.add_argument("--test-scoring-metric", type=str, default= "auROC",
                        choices = ["auPRC", "auROC", "accuracy", "f-measure", "spearman"],
                        help="Metric used to score the models on the testing data. ROC, PR, accuracy, f-measure or spearman.")

    parser.add_argument("--n", type=int, default=200, help="Number of models for oputuna to train.")
    parser.add_argument("--num-jobs", "-j", type=int, default=1,
                        help="Number of parallel jobs that you want Optuna to run while hyperparameter searching")
    
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

    parser.add_argument("--feature-list", type=int, nargs='*', help="The columns from the features that we'll take. Uses 0-based indices and is optional.")
    parser.add_argument("--feature-alias", type=str, default=None, help="The name for the selected features")

    parser.add_argument("--feature_type", type=str, default= "mut",
                        help="Mapping of aa representation between mutant and reference.")
    parser.add_argument("--lang_model_type", type=str, default = "lang_model_type", #choices = ["UniRep", "Rostlab_Bert", "other"],
                        help="Type of language model underlying features.")
    parser.add_argument("--pca_key", type = str, default = "None", help="PCA matrix specified by key in pca_mats. See config file for further specifications.")
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

    filen, file_ext = os.path.splitext(args.result_file)
    if file_ext != '.pkl':
        raise Exception(f"The result file({args.result_file}) must be a .pkl file!")

    if args.prediction_col is not None:
        args.train_prediction_col = args.prediction_col
        args.test_prediction_col = args.prediction_col

    if args.train_prediction_col is None:
        raise Exception(f"train-prediction-col is not specified! Specify it using the train-prediction-col flag or the prediction-col flag if it's the same as test-prediction-col")
    if args.test_prediction_col is None:
        raise Exception(f"test-prediction-col is not specified! Specify it using the test-prediction-col flag or the prediction-col flag if it's the same as train-prediction-col")


    if args.scoring_metric is not None:
        args.train_scoring_metric = args.scoring_metric
        args.test_scoring_metric = args.scoring_metric

    if args.train_scoring_metric is None:
        raise Exception(f"train-scoring-metric is not specified! Specify it using the train-scoring-metric flag or the scoring-metric flag if it's the same as test-scoring-metric")
    if args.test_scoring_metric is None:
        raise Exception(f"test-scoring-metric is not specified! Specify it using the test-scoring-metric flag or the scoring-metric flag if it's the same as train-scoring-metric")




    
if __name__ == "__main__":

    args = parse_run_optuna_args()

    verify_optuna_args(args)

    print(args)
    