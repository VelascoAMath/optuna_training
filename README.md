# Optuna hyperparamater tunning, model training, and testing.

Implements optuna optimization algorithms using sklearn ML algorithms (currently,
GradientBoostingClassifier, SVC, and Neural Nets.) auROC, auPRC or accuracy between datasets can be
used as the optimization objective.

Datasets (i.e. the crossvalidation, training, and testing set(s)) are user-specified according to ```config_function.py```.

Typical usage examples:
```
    python3 run_optuna_training_testing.py --lang_model_type Rostlab_Bert
    python3 run_optuna_training_testing.py --model_name SVC --scoring_metric auROC --lang_model_type Rostlab_Bert
```

To find all possible arguments to the main function ```run_optuna_training_testing.py```, please see ```run_optuna_args.py```.
