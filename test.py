from dataclasses import dataclass
from tqdm import tqdm
import glob
import itertools
import os
import random
import run_ML_diff_dataset
import run_ML_same_dataset
import subprocess





@dataclass
class Arguments(object):
	results_folder       : str = "optuna_via_sklearn/results/"
	model_name           : str = "GB"
	prediction_col       : str = None
	scoring_metric       : str = None
	train_prediction_col : str = "label"
	train_scoring_metric : str = None
	test_prediction_col  : str = "label"
	test_scoring_metric  : str = None
	n                    : int = 200
	num_jobs             : int = -1
	data_path            : str = None
	data_alias           : str = None
	training_path        : str = None
	training_alias       : str = None
	training_start       : int = None
	testing_path         : str = None
	testing_alias        : str = None
	testing_start        : int = None
	result_file          : str = "result.pkl"
	feature_type         : str = "mut"
	lang_model_type      : str = "lang_model_type"
	pca_key              : str = "None"
	pkl                  : bool = False
	feature_list         : list[int] = None
	feature_alias        : str = None

	def __contains__(self, value):
		return value in self.__dict__ and self.__dict__[value] is not None



def DRGN():
	dataset_dir = 'datasets/'


	alias_list = ['DRGN_BERT_Intersect', 'DRGN_PhysChem_Intersect', 'DRGN_DMSK_Intersect', 'DRGN_PhysChem_Intersect_No_Con']

	clf_to_num_test = {
		'Linear': 1,
		'Elastic': 40,
		'GB': 40,
		'Random': 1,
		'WeightedRandom': 1,
		'Frequent': 1,
		# 'KNN', 20,
		# 'SVC', 200,
		# 'SVC_balanced', 200,
		# 'NN', 20,
	}

	# metric_list = ['auPRC', 'auROC', 'f-measure', 'accuracy']
	metric_list = ['auPRC', 'auROC', 'f-measure']


	pkl_command_list = []
	command_list = []
	for file in alias_list:
		training_alias = f"{file}"
		training_name = f"{dataset_dir}/{training_alias}.pkl"


		if not os.path.exists(training_name):
			pkl_command_list.append(f"python run_ML_same_dataset.py --prediction-col label --scoring_metric auROC --model-name Linear --n 1\
				--data-path {dataset_dir}/{training_alias}.tsv --data-alias {training_alias} --data-start 5\
				--lang_model_type Rostlab_Bert --num-jobs -1 --pkl")

		for metric in metric_list:
			for clf, n_tests in clf_to_num_test.items():
				args = Arguments()
				args.prediction_col = "label"
				args.model_name = clf
				args.n = n_tests
				args.data_path = training_name
				args.data_alias = training_alias
				args.data_start = 5
				args.lang_model_type = "Rostlab_Bert"
				args.num_jobs = -1
				args.scoring_metric = metric
				command_list.append( (f"python run_ML_same_dataset.py --prediction-col label --model-name {clf} --n {n_tests}\
					--data-path {training_name} --data-alias {training_alias} --data-start 5 --lang_model_type Rostlab_Bert\
					--num-jobs -1 --scoring_metric {metric}", args))


	for command in pkl_command_list:
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

	random.shuffle(command_list)
	for command, args in tqdm(command_list, smoothing=0):
		print(command)
		run_ML_same_dataset.run_experiment(args)
 
def mmc2():
	dataset_dir = 'datasets/'

	training_list = ['DRGN_minus_mmc2_BERT_Intersect', 'DRGN_minus_mmc2_DMSK_Intersect', 'DRGN_minus_mmc2_PhysChem_Intersect', 'DRGN_minus_mmc2_PhysChem_Intersect_No_Con']
	testing_list  = [           'mmc2_BERT_Intersect',            'mmc2_DMSK_Intersect',            'mmc2_PhysChem_Intersect',            'mmc2_PhysChem_3_pos_neg_No_Con']

	clf_to_num_test = {
		'Linear': 1,
		'Elastic': 40,
		'GB': 40,
		'Random': 1,
		'WeightedRandom': 1,
		'Frequent': 1,
		# 'KNN', 20,
		# 'SVC', 200,
		# 'SVC_balanced', 200,
		# 'NN', 20,
	}

	train_metric_list = ['auPRC', 'auROC', 'f-measure']
	test_metric_list  = ['auPRC', 'auROC', 'f-measure']

	pkl_command_list = []
	command_list = []
	for i in range(len(training_list)):

		training_alias = training_list[i]
		training_name = f"{dataset_dir}/{training_alias}.pkl"
		testing_alias = testing_list[i]
		testing_name = f"{dataset_dir}/{testing_alias}.pkl"


		if not os.path.exists(training_name):
			pkl_command_list.append(f"python run_ML_diff_dataset.py --model-name Linear --n 1\
				--training-path {dataset_dir}/{training_alias}.tsv --training-alias {training_alias} --training-start 5\
				--testing-path {dataset_dir}/{testing_alias}.tsv --testing-alias {testing_alias} --testing-start 5\
				--lang_model_type Rostlab_Bert --num-jobs -1 --pkl")
			pkl_command_list.append(f"python run_ML_diff_dataset.py --model-name Linear --n 1\
				--testing-path {dataset_dir}/{testing_alias}.tsv --testing-alias {testing_alias} --testing-start 5\
				--training-path {dataset_dir}/{training_alias}.tsv --training-alias {training_alias} --training-start 5\
				--lang_model_type Rostlab_Bert --num-jobs -1 --pkl")

		for train_metric, test_metric in itertools.product(train_metric_list, test_metric_list):
			for clf, n_tests in clf_to_num_test.items():
				args = Arguments()
				args.model_name = clf
				args.n = n_tests
				args.training_path = training_name
				args.training_alias = training_alias
				args.training_start = 5
				args.testing_path = f"{dataset_dir}/{testing_alias}.pkl"
				args.testing_alias = testing_alias
				args.testing_start = 5
				args.lang_model_type = "Rostlab_Bert"
				args.num_jobs = -1
				args.train_scoring_metric = train_metric
				args.test_scoring_metric = test_metric
				command_list.append( (f"python run_ML_diff_dataset.py --model-name {clf} --n {n_tests} \
					--training-path {training_name} --training-alias {training_alias} --train-scoring-metric {train_metric} --training-start 5 \
					--testing-path {args.testing_path} --testing-alias {testing_alias} --test-scoring-metric {test_metric} --testing-start 5 \
					--lang_model_type Rostlab_Bert --num-jobs -1", args) ) 


	for command in pkl_command_list:
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

	random.shuffle(command_list)
	for command, args in tqdm(command_list, smoothing=0):
		print(command)
		run_ML_diff_dataset.run_experiment(args)

def maveDB():
	dataset_dir = 'datasets/'

	training_list = [ 'DRGN_minus_mavedb_PhysChem_Intersect',  'DRGN_minus_mavedb_PhysChem_No_Con_Intersect', 'DRGN_minus_mavedb_BERT_Intersect']
	testing_list  = [                  'mavedb_mut_PhysChem',                   'mavedb_mut_PhysChem_No_Con',                  'mavedb_mut_BERT']
	training_list.extend(['DRGN_minus_mavedb_PhysChem_No_Con_GB'])
	testing_list.extend ([       'mavedb_mut_PhysChem_No_Con_GB'])
	training_list.extend(['DRGN_minus_mavedb_DMSK_Intersect'])
	testing_list.extend ([                 'mavedb_mut_DMSK'])

	clf_to_num_test = {
		'Linear': 1,
		'Elastic': 40,
		'GB': 40,
		'Random': 1,
		'WeightedRandom': 1,
		'Frequent': 1,
		# 'KNN', 20,
		# 'SVC', 200,
		# 'SVC_balanced', 200,
		# 'NN', 20,
	}


	metric_list = ['auPRC', 'auROC', 'f-measure']

	pkl_command_list = []
	command_list = []
	for i in range(len(training_list)):

		training_alias = training_list[i]
		training_name = f"{dataset_dir}/{training_alias}.pkl"
		testing_alias_base = testing_list[i]

		print()

		if not os.path.exists(training_name):
			pkl_command_list.append(f"python run_ML_same_dataset.py --model-name Linear --n 1 --prediction-col label --scoring_metric auROC\
				--data-path {dataset_dir}/{training_alias}.tsv --data-alias {training_alias} --data-start 5\
				--lang_model_type Rostlab_Bert --num-jobs -1 --pkl")

		for training_name, testing_name in list(itertools.product([training_name], glob.glob(f"{dataset_dir}/{testing_alias_base}_experiment*tsv"))):
			testing_alias = os.path.splitext(os.path.basename(testing_name))[0]
			for train_metric in metric_list:
				for clf, n_tests in clf_to_num_test.items():
					args = Arguments()
					args.model_name = clf
					args.n = n_tests
					args.training_path = training_name
					args.training_alias = training_alias
					args.training_start = 5
					args.train_prediction_col = "label"
					args.testing_alias = testing_alias
					args.testing_start = 6
					args.test_prediction_col = "score"
					args.lang_model_type = "Rostlab_Bert"
					args.num_jobs = -1
					args.train_scoring_metric = train_metric
					args.test_scoring_metric = "spearman"
					if 'GB' in training_alias:
						args.feature_alias = 'GB_auROC'
						args.feature_list = '0  1  2  5  6  7  9 61 86 92'
					if os.path.exists(f"{dataset_dir}/{testing_alias}.pkl"):
						args.testing_path = f"{dataset_dir}/{testing_alias}.pkl"
					else:
						args.testing_path = f"{dataset_dir}/{testing_alias}.tsv"

					if args.feature_alias is None:
						command_list.append((f"python run_ML_diff_dataset.py --model-name {clf} --n {n_tests}\
						 --training-path {training_name} --training-alias {training_alias} --training-start 5 --train-prediction-col label\
						 --testing-path {args.testing_path} --testing-alias {testing_alias} --testing-start 6 --test-prediction-col score\
						 --lang_model_type Rostlab_Bert --num-jobs -1 --train-scoring-metric {train_metric} --test-scoring-metric spearman", args))
					else:
						command_list.append((f"python run_ML_diff_dataset.py --model-name {clf} --n {n_tests}\
						 --training-path {training_name} --training-alias {training_alias} --training-start 5 --train-prediction-col label\
						 --testing-path {args.testing_path} --testing-alias {testing_alias} --testing-start 6 --test-prediction-col score\
						 --lang_model_type Rostlab_Bert --num-jobs -1 --train-scoring-metric {train_metric} --test-scoring-metric spearman\
						 --feature-list {args.feature_list} --feature-alias {args.feature_alias}", args))


	for command in pkl_command_list:
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

	for command, args in tqdm(command_list, smoothing=0):
		print(command)
		run_ML_diff_dataset.run_experiment(args)





if __name__ == '__main__':
	DRGN()
	mmc2()
	maveDB()

