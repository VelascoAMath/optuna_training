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
	data_list = [(*key, *val) for key,val in result_dict.items() if len(key) == 4]
	columns = ['Dataset', 'Model', 'label', 'Metric', 'score', 'std']
	df = pd.DataFrame(data_list, columns=columns)
	df.dropna(inplace=True)
	df = df[df['Dataset'].str.contains('DRGN')]
	columns = ['Dataset', 'Model', 'Metric', 'score', 'std']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)

	# print(df)

	# Create a plot for every model
	g = sns.catplot(x="Metric", y="score", hue="Model", kind="bar", data=df, col="Dataset", ci="sd")
	plt.ylim(0, 1)
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_title(f'Scores of DRGN with {["BERT", "PhysChem", "PhysChem without Conservation"][i] }')

	plt.savefig(f"plots/DRGN_dataset.png")
	# plt.show()
	plt.close()

	# Create a plot for every dataset
	g = sns.catplot(x="Metric", y="score", hue="Dataset", kind="bar", data=df, col="Model", ci="sd")
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

	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 5 and 'mmc2' in key[1]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df.dropna(inplace=True)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)
	df.reset_index(drop=True, inplace=True)

	# print(df)


	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_xlabel('Metric')
		ax.set_title(f'Scores from using {["BERT", "PhysChem", "PhysChem without Conservation"][i] } features\n and testing on mmc2')
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
	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 5 and 'maved' in key[1]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df.dropna(inplace=True)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)

	# print(df)


	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.axhline(0)
		ax.set_xlabel('Metric for training on DRGN')
		ax.set_title(f'Scores from using {["BERT", "PhysChem", "PhysChem without Conservation"][i] } features\n and testing on maveDB')
	plt.ylim(-1,1)

	# plt.show()
	plt.savefig(f"plots/maveDB.png", bbox_inches='tight')
	plt.close()


if __name__ == '__main__':
	DRGN()
	mmc2()
	maveDB()
