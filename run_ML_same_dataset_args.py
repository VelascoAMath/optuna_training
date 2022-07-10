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
                        choices = ["GB", "SVC", "SVC_balanced", "NN", "KNN", "Elastic", "Linear", "Random", "WeightedRandom"],
                        help="Name of Machine Learning algorithm.")
    parser.add_argument("--scoring_metric", type=str, default="auROC",
                        choices = ["auPRC", "auROC", "accuracy", "f-measure", "spearman"],
                        help="Metric used to score the models. ROC, PR, accuracy, f-measure or spearman.")
    parser.add_argument("--prediction-col", type=str, default="label",
                        help="The column in the datasets that we need to predict. By default, it's set to 'label'")
    parser.add_argument("--n", type=int, default=200, help="Number of models for oputuna to train.")
    parser.add_argument("--num-jobs", "-j", type=int, default=1,
                        help="Number of parallel jobs that you want Optuna to run while hyperparameter searching")
    
    # Data specification parameters
    parser.add_argument("--data-path", type=str, help="Path to data.", required=True)
    parser.add_argument("--data-alias", type=str, help="Name to give to data.")
    parser.add_argument("--data-start", type=int, help="Index of column containing first feature.")

    parser.add_argument("--result-file", type=str, help="Name of the result file", default='result.pkl')

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

    # By default training/testing alias is the same as the path
    if args.data_alias is None:
        args.data_alias = args.data_path

    # Verify the existance of the training/testing files
    if not os.path.exists(args.data_path):
        raise Exception(f"The training file {args.data_path} does not exist!")

    if args.num_jobs < 1 and args.num_jobs != -1:
        raise Exception(f"The number of jobs({args.num_jobs}) must be -1 or a positive integer!")

    filen, file_ext = os.path.splitext(args.result_file)
    if file_ext != '.pkl':
        raise Exception(f"The result file({args.result_file}) must be a .pkl file!")
    
if __name__ == "__main__":

    args = parse_run_optuna_args()

    verify_optuna_args(args)

    print(args)
    
