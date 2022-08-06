"""

A program to generate plots that compare differences between the datasets

"""



from pprint import pprint
from vel_data_structures import AVL
from vel_data_structures import AVL_Dict
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

	# Create a plot for every dataset
	g = sns.catplot(x="Metric", y="Score", hue="Dataset", kind="bar", data=df, col="Model", ci="sd")
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_title(f'F-Measure of {sorted(set(df["Model"]))[i]} classifier')

	plt.savefig("plots/DRGN_model.png")
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

	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 6 and 'mmc2' in key[1] and key[5] is None]
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
		ax.set_xlabel('Metric')
		ax.set_title(f'Scores from using {["BERT", "DMSK", "PhysChem", "PhysChem without Conservation"][i] } features\n and testing on mmc2')
	plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"plots/mmc2.png", bbox_inches='tight')
	plt.close()



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
	df.sort_values(by=columns, inplace=True)
	df.reset_index(drop=True, inplace=True)

	# print(df)


	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.axhline(0)
		ax.set_xlabel('Metric for training on DRGN')
		ax.set_title(f'Scores from using {["BERT", "DMSK", "PhysChem", "PhysChem without Conservation"][i] } features\n and testing on maveDB')
	plt.ylim(-1,1)

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
		ax.set_xlabel('Metric for training on DRGN')
		ax.set_title(f'Scores from using {["PhysChem without Conservation"][i] } features\n and testing on maveDB')
	plt.ylim(-1,1)

	# plt.show()
	plt.savefig(f"plots/maveDB_GB.png", bbox_inches='tight')
	plt.close()

if __name__ == '__main__':
	DRGN()
	mmc2()
	maveDB()
	maveDB_GB()
