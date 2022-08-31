"""

A program to generate plots that compare differences between the datasets

"""



from pprint import pprint
from vel_data_structures import AVL
from vel_data_structures import AVL_Dict
from vel_data_structures import AVL_Set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns


def DRGN():
	'''
	Generate the plots for the DRGN dataset
	'''
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 5 and 'DRGN' in key[0] and 'minus' not in key[0]]
	columns = ['Dataset', 'Model', 'Metric', 'Feature', 'Trial', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df = df[df['Dataset'].str.contains('DRGN')]
	columns = ['Dataset', 'Model', 'Metric', 'Trial', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)

	# print(df)

	# Create a plot for every model
	g = sns.catplot(x="Metric", y="Score", hue="Model", kind="bar", data=df, col="Dataset", ci="sd")
	plt.ylim(0, 1)
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_title(f'Scores of DRGN with {["BERT", "DMSK", "PhysChem", "PhysChem without Conservation"][i] }')

	plt.savefig(f"plots/DRGN_dataset.png")
	# plt.show()
	plt.close()

	# Create a plot for every model and metric
	model_set = list(AVL_Set(df['Model']))
	metric_set  = list(AVL_Set(df['Metric']))

	fig, axes = plt.subplots(len(metric_set), len(model_set), sharex=True, figsize=(20,8))
	fig.suptitle('Average results of a specified metric on DRGN vs a specified model')
	for i, metric in enumerate(metric_set):
		for j, model in enumerate(model_set):
			h = df[df['Model'] == model]
			h = h[h['Metric'] == metric]
			h.reset_index(drop=True, inplace=True)
			sns.barplot(ax=axes[i, j], x="Metric", y="Score", hue="Dataset", data=h, ci="sd")
			print(h)
			# print(f"{i=} {j=} {model} {metric}")
			axes[i, j].set_ylabel(f'')
			axes[-1, j].set_xlabel(f'Trained on {model}')
			axes[i, j].set_ylim([0, 1])
			axes[i, j].get_legend().remove()
			
		axes[i, 0].set_ylabel(f'Score when tested on\n{metric}')
		axes[i, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

	for i, metric in enumerate(metric_set):
		for j, model in enumerate(model_set):
			axes[i, j].set_xlabel(f'')

	plt.savefig("plots/DRGN_model.png", bbox_inches='tight')
	# plt.show()
	plt.close()

def docm():
	'''
	Generate the plots for the docm dataset
	'''
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 5 and 'docm' in key[0] and 'minus' not in key[0]]
	columns = ['Dataset', 'Model', 'Metric', 'Feature', 'Trial', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df = df[df['Dataset'].str.contains('docm')]
	columns = ['Dataset', 'Model', 'Metric', 'Trial', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)

	# print(df)

	# Create a plot for every model
	g = sns.catplot(x="Metric", y="Score", hue="Model", kind="bar", data=df, col="Dataset", ci="sd")
	plt.ylim(0, 1)
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_title(f'Scores of docm with {["BERT", "DMSK", "PhysChem", "PhysChem without Conservation"][i] }')

	plt.savefig(f"plots/docm_dataset.png")
	# plt.show()
	plt.close()

	# Create a plot for every dataset
	g = sns.catplot(x="Metric", y="Score", hue="Dataset", kind="bar", data=df, col="Model", ci="sd")
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_title(f'F-Measure of {sorted(set(df["Model"]))[i]} classifier')

	plt.savefig("plots/docm_model.png")
	# plt.show()
	plt.close()

def mmc2():
	'''
	Generate the plots for the mmc2 dataset
	'''
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	# pprint(result_dict)

	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 6 and 'mmc2' in key[1] and key[5] is None and key[3] == key[4]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	df = df[columns]
	df.sort_values(by=['Test_dataset', 'Train_dataset'], inplace=True)
	df.reset_index(drop=True, inplace=True)

	print(df)


	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_xlabel('Metric')
		ax.set_title(f'Scores from using {["BERT", "DMSK", "PhysChem without Conservation", "PhysChem"][i // 2] } features\n and training on {["DRGN", "docm"][i % 2] } and testing on mmc2')
	plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"plots/mmc2.png", bbox_inches='tight')
	plt.close()

	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 6 and 'DRGN' in key[0] and 'mmc2' in key[1] and key[5] is None and '_bg' not in key[4]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	df = df[columns]
	df['Train_dataset'] = df['Train_dataset'].replace({'DRGN_minus_mmc2_BERT_Intersect': 'BERT',
		'DRGN_minus_mmc2_DMSK_Intersect': 'DMSK',
		'DRGN_minus_mmc2_PhysChem_Intersect': 'PhysChem',
		'DRGN_minus_mmc2_PhysChem_Intersect_No_Con': 'PhysChem\nWithout Conservation'})
	df.sort_values(by=columns, inplace=True)
	df.reset_index(drop=True, inplace=True)


	train_set = list(AVL_Set(df['Train_dataset']))
	metric_set  = list(AVL_Set(df['Train_metric']))
	fig, axes = plt.subplots(len(metric_set), len(metric_set), sharex=True, figsize=(20,8))
	fig.suptitle('MMC2 when using different metrics to train and test')
	for i, test_metric in enumerate(metric_set):
		for j, train_metric in enumerate(metric_set):
			h = df[df['Train_metric'] == train_metric]
			h = h[h['Test_metric'] == test_metric]
			h.reset_index(drop=True, inplace=True)
			sns.barplot(ax=axes[i, j], x="Train_dataset", y="Score", hue="Model", data=h, ci="sd")
			print(h)
			# print(f"{i=} {j=} {train_metric} {test_metric}")
			axes[i, j].set_xlabel(f'')
			axes[-1, j].set_xlabel(f'Trained on {train_metric}')
			axes[i, j].set_ylim([0, 1])
			axes[i, j].get_legend().remove()
			# axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=10, ha='right')
			
		axes[i, 0].set_ylabel(f'Score when tested on\n{test_metric}')
		axes[i, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))


	# plt.show()
	plt.savefig(f"plots/mmc2_metrics.png", bbox_inches="tight")


def maveDB():
	'''
	Generate the plots for the maveDB datasets
	'''
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)


	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 6 and 'maved' in key[1] and key[5] is None]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Score']
	df = df[columns]
	df.sort_values(by=['Test_dataset', 'Train_dataset'], inplace=True)
	df.reset_index(drop=True, inplace=True)

	# print(df)


	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.axhline(0)
		ax.set_xlabel('Metric for training on DRGN')
		ax.set_title(f'Scores from using\n{["BERT", "DMSK", "PhysChem without Conservation", "PhysChem"][i // 2] } features and training on\n{["DRGN", "docm"][i % 2] } and testing on maveDB')
	plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"plots/maveDB.png", bbox_inches='tight')
	plt.close()


def maveDB_GB():
	'''
	Generate the plots for the maveDDDB datasets when using GB features
	'''
	'''
	Generate the plots for the maveDB datasets
	'''
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)


	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 6 and 'maved' in key[1] and key[5] == 'GB_auROC']
	data_list.extend([(*key, val) for key,val in result_dict.items() if len(key) == 6 and 'DRGN_minus_mavedb_PhysChem_No_Con_Intersect' in key[0]])
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)
	df.reset_index(drop=True, inplace=True)

	print(df)


	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.axhline(0)
		ax.set_xlabel(f'Metric for training on DRGN using \n{["GB PhysChem without Conservation", "All PhysChem without Conservation" ][i]} features')
		ax.set_title(f'Scores from using \n{["GB PhysChem without Conservation", "All PhysChem without Conservation" ][i] } features\n and testing on maveDB')
	plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"plots/maveDB_GB.png", bbox_inches='tight')
	plt.close()

if __name__ == '__main__':
	DRGN()
	docm()
	mmc2()
	maveDB()
	maveDB_GB()
