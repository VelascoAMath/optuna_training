"""Optuna optimization of hyperparameters.

Implements optuna optimization algorithms using sklearn ML algorithms (currently,
GradientBoostingClassifier.) auROC, auPRC or accuracy between datasets can be
used as the optimization objective. Returns an optuna study class.

  Typical usage examples:
    python3 run_ML_same_dataset.py
    python3 run_ML_same_dataset.py --scoring_metric ROC
"""
import itertools
from multiprocessing import Pool
from os.path import exists

import optuna
from joblib import dump, load
from tqdm import tqdm

import optuna_via_sklearn.specify_sklearn_models
from iterative_stratification import StratifiedGroupKFold
from optuna_via_sklearn.config import *
from optuna_via_sklearn.load_data import *
from run_ML_args import *


def fill_objective(dataset, index_list, metric, model_name, param=None):
    def filled_obj(trial):
        return optuna_via_sklearn.specify_sklearn_models.objective(trial, dataset, index_list, metric, model_name,
                                                                   param)

    return filled_obj


# Since Pool can only take in one parameter, we need this wrapper function
# It's defined at the top-level because Pool requires it (https://stackoverflow.com/a/8805244)
def objective_star(x):
    return optuna_via_sklearn.specify_sklearn_models.objective(*x)


def optimize_hyperparams(training_data, train_metric, n_trials, model_name, n_splits=5, n_jobs=1):
    if not isinstance(n_splits, int):
        raise Exception(f"n_splits is not an int but instead {type(n_splits)}")

    if n_splits < 3:
        raise Exception(f"The number of splits({n_splits}) must be greater than 2!")

    if not isinstance(n_jobs, int):
        raise Exception(f"n_jobs is not an int but instead {type(n_jobs)}")

    if not isinstance(n_trials, int):
        raise Exception(f"n_trials is not an int but instead {type(n_trials)}")

    # Rand 5 Training Testing Split
    gss = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    split = gss.split(training_data.features, training_data.labels, groups=training_data.input_df["protein_id"])

    # fast_check_for_repeating_rows(training_data.features)

    # print(training_data.features.shape)
    testing_index_list = []
    i = 1
    for train, test in split:
        # print(f"Split {i} {train=} {test=}")
        # print(f"Split {i} {len(train)=} {len(test)=}")
        i += 1
        testing_index_list.append(test)
        # print(training_data.input_df["label"][np.r_[train]].value_counts())
        # print(training_data.input_df["label"][np.r_[test]].value_counts())

    # Verify that the proteins stay within the same indices
    for i, j in itertools.combinations(range(n_splits), 2):
        protein_i_set = set(training_data.input_df["protein_id"].to_numpy()[testing_index_list[i]])
        protein_j_set = set(training_data.input_df["protein_id"].to_numpy()[testing_index_list[j]])
        protein_inter_set = protein_i_set & protein_j_set
        if len(protein_inter_set) != 0:
            raise Exception(
                f"{protein_inter_set} are in {protein_i_set} and {protein_j_set} in indices {testing_index_list[i]} and {testing_index_list[i]}!")

    param_list = []
    score_list = []
    for i in tqdm(range(n_splits), desc='Fold loop'):
        training_index_list = []
        for k in range(n_splits):
            if k != i: training_index_list.append(testing_index_list[k])

        specified_objective = fill_objective(training_data, training_index_list, train_metric, model_name, param=None)
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(directions=["maximize", "minimize"])
        study.optimize(specified_objective, n_trials=n_trials, n_jobs=n_jobs)
        study_list = study.best_trials

        best_study = sorted(study_list, key=lambda x: (x.values[0], -x.values[1]))[-1]

        # print(f"{study_list=}")
        print(f"{best_study=}")
        param_list.append(best_study.params)

        print("training_index_list=", hashlib.sha256(bytes(f"{training_index_list}", "utf-8")).hexdigest())
        # print(f"{list(itertools.chain.from_iterable(training_index_list))=}")
        X_train = np.take(training_data.features, list(itertools.chain.from_iterable(training_index_list)), axis=0)
        y_train = np.take(training_data.labels, list(itertools.chain.from_iterable(training_index_list)), axis=0)

        X_test = np.take(training_data.features, testing_index_list[i], axis=0)
        y_test = np.take(training_data.labels, testing_index_list[i], axis=0)

    return param_list


def optimize_hyperparams2(training_data, test_metric, n_trials, model_name, param_list, n_splits=5, n_jobs=1):
    if not isinstance(n_splits, int):
        raise Exception(f"n_splits is not an int but instead {type(n_splits)}")

    if n_splits < 3:
        raise Exception(f"The number of splits({n_splits}) must be greater than 2!")

    if not isinstance(n_jobs, int):
        raise Exception(f"n_jobs is not an int but instead {type(n_jobs)}")

    if not isinstance(n_trials, int):
        raise Exception(f"n_trials is not an int but instead {type(n_trials)}")

    # Rand 5 Training Testing Split
    gss = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    split = gss.split(training_data.features, training_data.labels, groups=training_data.input_df["protein_id"])

    # fast_check_for_repeating_rows(training_data.features)

    # print(training_data.features.shape)
    testing_index_list = []
    i = 1
    for train, test in split:
        i += 1
        testing_index_list.append(test)

    # Verify that the proteins stay within the same indices
    for i, j in itertools.combinations(range(n_splits), 2):
        protein_i_set = set(training_data.input_df["protein_id"].to_numpy()[testing_index_list[i]])
        protein_j_set = set(training_data.input_df["protein_id"].to_numpy()[testing_index_list[j]])
        protein_inter_set = protein_i_set & protein_j_set
        if len(protein_inter_set) != 0:
            raise Exception(
                f"{protein_inter_set} are in {protein_i_set} and {protein_j_set} in indices {testing_index_list[i]} and {testing_index_list[i]}!")

    score_list = []
    l = []

    # Now, we are taking the average CV score of a parameter instead of just seeing how it did on its fold
    for param in tqdm(param_list):
        x = (None, training_data, testing_index_list, test_metric, model_name, param)
        l.append(x)

    with tqdm(total=n_splits) as pbar:
        num_pool_workers = None
        if n_jobs is None or n_jobs == -1:
            num_pool_workers = None
        else:
            num_pool_workers = args.n_jobs

        with Pool(num_pool_workers) as p:
            i = 0
            for score in p.imap(objective_star, l):
                param = param_list[i]
                print(f"{param=} {score=}")
                score_list.append(score[0])
                pbar.update(1)
                i += 1

    return (param_list, score_list)


def main():
    args = parse_run_optuna_args()
    run_experiment(args)


def run_experiment(args):
    verify_optuna_args(args)

    # Define prefix for all files produced by run

    # Check if optuna-trained model already exists
    if args.feature_alias is not None:
        train_path = f"{args.results_folder}/{args.data_alias}_{args.feature_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.train_scoring_metric}_model.joblib"
        model_path = f"{args.results_folder}/{args.data_alias}_{args.feature_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.train_scoring_metric}_{args.test_scoring_metric}_model.joblib"
    else:
        train_path = f"{args.results_folder}/{args.data_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.train_scoring_metric}_model.joblib"
        model_path = f"{args.results_folder}/{args.data_alias}_{args.lang_model_type}_{args.pca_key}_{args.model_name}_{args.train_scoring_metric}_{args.test_scoring_metric}_model.joblib"

    if exists(model_path):
        # Load model
        print("Loading model at: " + model_path)
        try:
            (best_params, score_list) = load(model_path)
        except:
            os.remove(model_path)
            raise Exception(f"Can't load {model_path} so we're deleting it")
    else:
        # Load training/testing data
        config = DataSpecification(args)
        datasets, metadata = load_data(config)
        # Optimize hyperparameters with optuna
        print("Creating model at: " + model_path)
        print("Running optuna optimization.")
        # fast_check_for_repeating_rows(datasets["data"].features)
        if exists(train_path):
            print("Loading train model at: " + train_path)
            try:
                (best_params) = load(train_path)
            except:
                os.remove(train_path)
                raise Exception(f"Can't load {train_path} so we're deleting it")
        else:
            print("Creating model at: " + train_path)
            (best_params) = optimize_hyperparams(datasets["data"], args.train_scoring_metric, args.n, args.model_name,
                                                 n_splits=5, n_jobs=args.num_jobs)
            dump((best_params), train_path)
        (best_params, score_list) = optimize_hyperparams2(datasets["data"], args.test_scoring_metric, args.n,
                                                          args.model_name, best_params, n_splits=5,
                                                          n_jobs=args.num_jobs)
        dump((best_params, score_list), model_path)

    # print(f"The best parameters are {best_params}")
    final_score = np.mean(score_list)
    final_score_std = np.std(score_list)
    print(f"Final {args.test_scoring_metric} is mean({score_list})={final_score} std={final_score_std}")

    if not os.path.exists(args.result_file):
        result_dict = dict()
    else:
        with open(args.result_file, 'rb') as f:
            result_dict = pickle.load(f)

    obtained_new_results = False
    for i, score in enumerate(score_list):
        result_header = (
        args.data_alias, args.model_name, args.train_scoring_metric, args.test_scoring_metric, args.feature_alias, i)

        if result_header not in result_dict:
            obtained_new_results = True
            result_dict[result_header] = score

    if obtained_new_results:
        with open(args.result_file, 'wb') as f:
            pickle.dump(result_dict, f)

    # print(f"{result_dict=}")
    print()
    print()
    print()


if __name__ == "__main__":
    main()
