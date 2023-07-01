'''
This program only exists to create pickled versions of datasets

'''


import argparse
import os

from optuna_via_sklearn.load_data import process_data


def parse_run_optuna_args():
    parser = argparse.ArgumentParser(description="Optuna optimization of hyperparameters.")
    # Optuna parameters
    parser.add_argument("--prediction-col", type=str, default='label',
                        help="The column in the datasets that we need to predict. By default, it's set to 'label'")

    # Data specification parameters
    parser.add_argument("--data-path", type=str, help="Path to data.", required=True)
    parser.add_argument("--data-start", type=int, help="Index of column containing first feature.", required=True)

    parser.add_argument("--feature-list", type=int, nargs='*', help="The columns from the features that we'll take. Uses 0-based indices and is optional.")

    args = parser.parse_args()
    return(args)


def verify_optuna_args(args):

    # Verify the existance of the training/testing files
    if not os.path.exists(args.data_path):
        raise Exception(f"The file {args.data_path} does not exist!")

    filen, file_ext = os.path.splitext(args.data_path)
    if file_ext != '.tsv':
        raise Exception(f"The result file({args.data_path}) must be a .tsv file!")





def main(args):
	exclude = {
            "mcf10A":[], #["ENSP00000361021", "ENSP00000483066"], #PTEN
            "maveDB": ["ENSP00000312236", "ENSP00000350283", "ENSP00000418960", "ENSP00000417148", "ENSP00000496570", "ENSP00000465818", "ENSP00000467329", "ENSP00000465347", "ENSP00000418775", "ENSP00000418548", "ENSP00000420705", "ENSP00000419481", "ENSP00000420412", "ENSP00000418819", "ENSP00000418212", "ENSP00000417241", "ENSP00000326002", "ENSP00000489431", "ENSP00000397145", "ENSP00000419274", "ENSP00000498906",
                       "ENSP00000419988", "ENSP00000420253", "ENSP00000418986", "ENSP00000419103", "ENSP00000420201", "ENSP00000495897", "ENSP00000417554", "ENSP00000417988", "ENSP00000420781", "ENSP00000494614", "ENSP00000478114"] #BRCA1
        }
	process_data(args.data_path, args.data_start, exclude, convert_to_pickle=True, prediction_column=args.prediction_col)

if __name__ == '__main__':
	args = parse_run_optuna_args()
	main(args)