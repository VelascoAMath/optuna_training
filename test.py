from tqdm import tqdm
import glob
import itertools
import os
import random
import subprocess





def DRGN():
	dataset_dir = 'datasets/'


	alias_list = ['DRGN_BERT_Intersect', 'DRGN_PhysChem_Intersect', 'DRGN_PhysChem_Intersect_No_Con',
				'mmc2_BERT_Intersect', 'mmc2_PhysChem_Intersect', 'mmc2_PhysChem_Intersect_No_Con']

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
	metric_list = ['auPRC', 'auROC']


	pkl_command_list = []
	command_list = []
	for file in alias_list:
		training_alias = f"{file}"
		training_name = f"{dataset_dir}/{training_alias}.pkl"


		if not os.path.exists(training_name):
			pkl_command_list.append(f"python run_ML_same_dataset.py --prediction-col label --model_name Linear --n 1 --data-path {dataset_dir}/{training_alias}.tsv --data-alias {training_alias} --data-start 5 --lang_model_type Rostlab_Bert --num-jobs -1 --pkl")

		for metric in metric_list:
			for clf, n_tests in clf_to_num_test.items():
				command_list.append(f"python run_ML_same_dataset.py --prediction-col label --model_name {clf} --n {n_tests} --data-path {training_name} --data-alias {training_alias} --data-start 5 --lang_model_type Rostlab_Bert --num-jobs -1 --scoring_metric {metric}")


	for command in pkl_command_list:
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

	random.shuffle(command_list)
	for command in tqdm(command_list, smoothing=0):
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

def mmc2():
	dataset_dir = 'datasets/'

	training_list = ['DRGN_minus_mmc2_BERT_Intersect', 'DRGN_minus_mmc2_PhysChem_Intersect', 'DRGN_minus_mmc2_PhysChem_Intersect_No_Con']
	testing_list  = [           'mmc2_BERT_Intersect',            'mmc2_PhysChem_Intersect',            'mmc2_PhysChem_3_pos_neg_No_Con']

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
	metric_list = ['auPRC', 'auROC']

	pkl_command_list = []
	command_list = []
	for i in range(len(training_list)):

		training_alias = training_list[i]
		training_name = f"{dataset_dir}/{training_alias}.pkl"
		testing_alias = testing_list[i]
		testing_name = f"{dataset_dir}/{testing_alias}.pkl"


		if not os.path.exists(training_name):
			pkl_command_list.append(f"python run_ML_diff_dataset.py --model_name Linear --n 1 --training-path {dataset_dir}/{training_alias}.tsv --training-alias {training_alias} --training-start 5 --testing-path {dataset_dir}/{testing_alias}.tsv --testing-alias {testing_alias} --testing-start 5 --lang_model_type Rostlab_Bert --num-jobs -1 --pkl")

		for metric in metric_list:
			for clf, n_tests in clf_to_num_test.items():
				command_list.append(f"python run_ML_diff_dataset.py --model_name {clf} --n {n_tests} --training-path {training_name} --training-alias {training_alias} --training-start 5 --testing-path {dataset_dir}/{testing_alias}.tsv --testing-alias {testing_alias} --testing-start 5 --lang_model_type Rostlab_Bert --num-jobs -1 --scoring_metric {metric}")


	for command in pkl_command_list:
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

	random.shuffle(command_list)
	for command in tqdm(command_list, smoothing=0):
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

def maveDB():
	dataset_dir = 'datasets/'

	training_list = [ 'DRGN_minus_mavedb_PhysChem_Intersect',  'DRGN_minus_mavedb_PhysChem_No_Con_Intersect', 'DRGN_minus_mavedb_BERT_Intersect']
	testing_list  = [                  'mavedb_mut_PhysChem',                   'mavedb_mut_PhysChem_No_Con',                  'mavedb_mut_BERT']
	# training_list = [ 'DRGN_minus_mavedb_PhysChem_Intersect',  'DRGN_minus_mavedb_PhysChem_No_Con_Intersect']
	# testing_list  = [                  'mavedb_mut_PhysChem',                   'mavedb_mut_PhysChem_No_Con']

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


	metric_list = ['auPRC', 'auROC']

	pkl_command_list = []
	command_list = []
	for i in range(len(training_list)):

		training_alias = training_list[i]
		training_name = f"{dataset_dir}/{training_alias}.pkl"
		testing_alias_base = testing_list[i]

		print()

		if not os.path.exists(training_name):
			pkl_command_list.append(f"python run_ML_diff_dataset.py --model_name Linear --n 1 --training-path {dataset_dir}/{training_alias}.tsv --training-alias {training_alias} --training-start 5 --testing-path {dataset_dir}/{testing_alias}.tsv --testing-alias {testing_alias} --testing-start 6 --lang_model_type Rostlab_Bert --num-jobs -1 --pkl")

		for training_name, testing_name in list(itertools.product([training_name], glob.glob(f"{dataset_dir}/{testing_alias_base}_experiment*"))):
			testing_alias = os.path.splitext(os.path.basename(testing_name))[0]
			testing_alias = testing_alias.replace(' ', '\ ')
			for metric in metric_list:
				for clf, n_tests in clf_to_num_test.items():
					if os.path.exists(f"{dataset_dir}/{testing_alias}.pkl"):
						command_list.append(f"python run_ML_diff_dataset.py --model_name {clf} --n {n_tests}\
						 --training-path {training_name} --training-alias {training_alias} --training-start 5 --train-prediction-col label\
						 --testing-path {dataset_dir}/{testing_alias}.pkl --testing-alias {testing_alias} --testing-start 6 --test-prediction-col score\
						 --lang_model_type Rostlab_Bert --num-jobs -1 --train-scoring-metric {metric} --test-scoring-metric spearman")
					else:
						command_list.append(f"python run_ML_diff_dataset.py --model_name {clf} --n {n_tests}\
						 --training-path {training_name} --training-alias {training_alias} --training-start 5 --train-prediction-col label\
						 --testing-path {dataset_dir}/{testing_alias}.tsv --testing-alias {testing_alias} --testing-start 6 --test-prediction-col score\
						 --lang_model_type Rostlab_Bert --num-jobs -1 --train-scoring-metric {metric} --test-scoring-metric spearman")


	for command in pkl_command_list:
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")

	for command in tqdm(command_list, smoothing=0):
		print(command)
		code = os.system(command)
		if code != 0:
			raise Exception(f"'{command}' returned {code}")





if __name__ == '__main__':
	DRGN()
	mmc2()
	maveDB()
