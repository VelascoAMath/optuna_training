"""

A program to generate plots that compare differences between the datasets

"""
import itertools
import pickle as pkl
import sqlite3
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score
from vel_data_structures import AVL_Set

model_order = ['Logistic', 'GB', 'GNB', 'DT', 'Random', 'WeightedRandom', 'Frequent']

plot_location = "../Thesis/plots/"

def train_and_test_metrics():
	result_file_name = 'training_results.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key, val in result_dict.items() if 'BERT' not in key[0] or key[3] == 'BERT_1']
	columns = ['Dataset', 'Model', 'Train Metric', 'Test Metric', 'Feature', 'Trial', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df = df[df['Model'] != 'Random']
	df = df[df['Model'] != 'WeightedRandom']
	df = df[df['Model'] != 'Frequent']
	df['Feature'] = df['Dataset'].replace(
		{
			'DRGN_BERT': 'BERT',
			'DRGN_DMSK': 'DMSK',
			'DRGN_PhysChem_No_Con': 'PhysChem\nNo Cons',
			'DRGN_PhysChem': 'PhysChem'
		}
	)
	df['Dataset'] = df['Dataset'].apply(lambda x: x[:4])
	columns = ['Dataset', 'Model', 'Train Metric', 'Test Metric', 'Trial', 'Score']
	df.sort_values(by=columns, inplace=True)
	group_cols = ['Dataset', 'Model', 'Test Metric', 'Feature', 'Trial']
	df['Score'] = df[['Score']] - df.groupby(by=group_cols).transform('mean')[['Score']]

	print(df)
	for dataset in AVL_Set(df['Dataset']):
		for model in AVL_Set(df['Model']):
			plt.title(f"Mean-centered scores of {model} on {dataset}")
			sns.boxplot(x="Test Metric", y="Score", hue="Train Metric",
						data=df[(df['Dataset'] == dataset) & (df['Model'] == model)])
			plt.savefig(f"{plot_location}/training_{dataset}_{model}.png")
			# plt.show()
			plt.close()
			print(f"{plot_location}/training_{dataset}_{model}.png")


def DRGN():
	'''
	Generate the plots for the DRGN dataset
	'''
	result_file_name = 'training_results.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key, val in result_dict.items() if 'BERT' not in key[0] or key[4] == 'BERT_1']
	# Let's read in everything as a panda dataframe
	columns = ['Alias', 'Model', 'Train Metric', 'Test Metric', 'Feature', 'Trial', 'Score']
	master_df = pd.DataFrame(data_list, columns=columns)
	master_df['Feature'] = master_df['Alias'].replace(
		{
			'DRGN_BERT': 'BERT',
			'DRGN_DMSK': 'DMSK',
			'DRGN_PhysChem_No_Con': 'PhysChem\nNo Cons',
			'DRGN_PhysChem': 'PhysChem',
			'DRGN_BD': 'BERT+DMSK',
			'DRGN_BN': 'BERT+No PhysChem',
			'DRGN_BP': 'BERT+PhysChem',
			'docm_BERT': 'BERT',
			'docm_DMSK': 'DMSK',
			'docm_PhysChem_No_Con': 'PhysChem\nNo Cons',
			'docm_PhysChem': 'PhysChem',
			'docm_BD': 'BERT+DMSK',
			'docm_BN': 'BERT+No PhysChem',
			'docm_BP': 'BERT+PhysChem',
		}
	)
	master_df['Dataset'] = master_df['Alias'].replace(
		{
			'DRGN_BERT': 'DRGN',
			'DRGN_DMSK': 'DRGN',
			'DRGN_PhysChem_No_Con': 'DRGN',
			'DRGN_PhysChem': 'DRGN',
			'DRGN_BD': 'DRGN',
			'DRGN_BN': 'DRGN',
			'DRGN_BP': 'DRGN',
			'docm_BERT': 'docm',
			'docm_DMSK': 'docm',
			'docm_PhysChem_No_Con': 'docm',
			'docm_PhysChem': 'docm',
			'docm_BD': 'docm',
			'docm_BN': 'docm',
			'docm_BP': 'docm',
		}
	)
	columns = ['Alias', 'Dataset', 'Feature', 'Model', 'Train Metric', 'Test Metric', 'Trial', 'Score']
	master_df = master_df[columns]
	master_df.sort_values(by=columns, inplace=True)
	master_df.reset_index(inplace=True, drop=True)
	print(master_df)

	df = master_df[master_df['Dataset'] == 'DRGN']
	df = df[df['Train Metric'] == 'auROC']
	df = df[df['Test Metric'] == df['Train Metric']]
	pprint(df)

	# Create a plot for every model
	sns.set_theme(font_scale=2, palette=sns.color_palette())
	sns.catplot(x="Train Metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, col="Feature",
				errorbar="sd")
	plt.ylim(0, 1)
	plt.ylabel("auROC", fontdict={"fontsize": "large"})

	plt.savefig(f"{plot_location}/DRGN_dataset.png")
	# plt.show()
	plt.close()

	df = master_df[master_df['Dataset'] == 'DRGN']
	df = df[df['Train Metric'] == 'auROC']
	df = df[df['Test Metric'] == 'auROC']
	df = df[df['Model'] == 'GB']
	df.sort_values(by=columns, inplace=True)

	pprint(df)
	# print(data_list)

	# Create a plot for every model
	sns.set_theme(font_scale=2, palette=sns.color_palette())
	sns.catplot(x="Model", y="Score", hue="Feature", kind="bar", col="Dataset", data=df, errorbar="sd")
	# plt.ylim(0, 1)
	plt.ylabel("auROC", fontdict={"fontsize": "large"})

	plt.savefig(f"{plot_location}/DRGN_GB.png")
	print(AVL_Set(df['Feature']))
	# plt.show()
	plt.close()

	df = master_df[master_df['Dataset'] == 'DRGN']
	df = df[df['Train Metric'] == 'auROC']
	df = df[df['Test Metric'] == 'auROC']
	pprint(df)

	# Create a plot for every model
	sns.set_theme(font_scale=2, palette=sns.color_palette())
	sns.catplot(x="Test Metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, col="Dataset",
				errorbar="sd")
	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	# plt.ylim(0, 1)

	plt.savefig(f"{plot_location}/DRGN_Model.png")
	# plt.show()
	plt.close()

	# Create a plot for every model
	sns.set_theme(font_scale=2, palette=sns.color_palette())
	sns.catplot(x="Test Metric", y="Score", hue="Model", hue_order=model_order, kind="bar",
				data=df[df['Feature'] == 'BERT'], col="Dataset", errorbar="sd")

	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.savefig(f"{plot_location}/DRGN_BERT_Model.png")
	plt.close()

	# Create a plot for every model
	sns.set_theme(font_scale=2, palette=sns.color_palette())
	sns.catplot(x="Test Metric", y="Score", hue="Model", hue_order=model_order, kind="bar",
				data=df[df['Feature'] == 'BERT+DMSK'], col="Dataset", errorbar="sd")

	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.savefig(f"{plot_location}/DRGN_BERT+DMSK_Model.png")
	plt.close()


def docm():
	"""
	Generate the plots for the docm dataset
	"""
	result_file_name = 'training_results.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key, val in result_dict.items() if 'BERT' not in key[0] or key[4] == 'BERT_1']
	# Let's read in everything as a panda dataframe
	columns = ['Alias', 'Model', 'Train Metric', 'Test Metric', 'Feature', 'Trial', 'Score']
	master_df = pd.DataFrame(data_list, columns=columns)
	master_df['Feature'] = master_df['Alias'].replace(
		{
			'DRGN_BERT': 'BERT',
			'DRGN_DMSK': 'DMSK',
			'DRGN_PhysChem_No_Con': 'PhysChem\nNo Cons',
			'DRGN_PhysChem': 'PhysChem',
			'DRGN_BD': 'BERT+DMSK',
			'DRGN_BN': 'BERT+No PhysChem',
			'DRGN_BP': 'BERT+PhysChem',
			'docm_BERT': 'BERT',
			'docm_DMSK': 'DMSK',
			'docm_PhysChem_No_Con': 'PhysChem\nNo Cons',
			'docm_PhysChem': 'PhysChem',
			'docm_BD': 'BERT+DMSK',
			'docm_BN': 'BERT+No PhysChem',
			'docm_BP': 'BERT+PhysChem',
		}
	)
	master_df['Dataset'] = master_df['Alias'].replace(
		{
			'DRGN_BERT': 'DRGN',
			'DRGN_DMSK': 'DRGN',
			'DRGN_PhysChem_No_Con': 'DRGN',
			'DRGN_PhysChem': 'DRGN',
			'DRGN_BD': 'DRGN',
			'DRGN_BN': 'DRGN',
			'DRGN_BP': 'DRGN',
			'docm_BERT': 'docm',
			'docm_DMSK': 'docm',
			'docm_PhysChem_No_Con': 'docm',
			'docm_PhysChem': 'docm',
			'docm_BD': 'docm',
			'docm_BN': 'docm',
			'docm_BP': 'docm',
		}
	)
	columns = ['Alias', 'Dataset', 'Feature', 'Model', 'Train Metric', 'Test Metric', 'Trial', 'Score']
	master_df = master_df[columns]
	master_df.sort_values(by=columns, inplace=True)
	master_df.reset_index(inplace=True, drop=True)
	print(master_df)

	df = master_df[master_df['Dataset'].str.contains('docm')]
	df = df[df['Train Metric'] == 'auROC']
	df = df[df['Test Metric'] == 'auROC']

	# Create a plot for every model
	sns.catplot(x="Test Metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, col="Feature",
				errorbar="sd")
	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.ylim(0, 1)

	sns.set_theme(font_scale=2, palette=sns.color_palette())
	plt.savefig(f"{plot_location}/docm_dataset.png")
	# plt.show()
	plt.close()

	df = master_df[master_df['Dataset'].str.contains('docm')]
	df = df[df['Train Metric'] == 'auROC']
	df = df[df['Test Metric'] == 'auROC']
	df = df[df['Model'] == 'GB']
	pprint(df)

	# Create a plot for every model
	sns.catplot(x="Model", y="Score", hue="Feature", kind="bar", col="Dataset", data=df, errorbar="sd")

	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	sns.set_theme(font_scale=2, palette=sns.color_palette())
	plt.savefig(f"{plot_location}/docm_GB.png")
	# plt.show()
	plt.close()

	df = master_df[master_df['Dataset'] == 'docm']
	df = df[df['Train Metric'] == 'auROC']
	df = df[df['Test Metric'] == 'auROC']

	# Create a plot for every model
	sns.catplot(x="Test Metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, col="Dataset",
				errorbar="sd")

	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	sns.set_theme(font_scale=2, palette=sns.color_palette())
	plt.savefig(f"{plot_location}/docm_Model.png")
	# plt.show()
	plt.close()


def mmc2():
	"""
	Generate the plots for the mmc2 dataset
	"""
	result_file_name = 'mmc2_results.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	# pprint(list(result_dict.keys()))

	data_list = [(*key, val) for key, val in result_dict.items() if
				 ('BERT' not in key[0] or key[5] == 'BERT_1') and key[3] == key[4]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Feature', 'Score']
	master_df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	master_df = master_df[columns]
	master_df = master_df[master_df['Train_metric'] == 'auROC']
	# df = df[df['Model'] != 'Frequent']

	# pprint(data_list)
	master_df['Test_dataset'] = master_df['Test_dataset'].replace(
		{
			'mmc2_BERT': 'BERT',
			'mmc2_DMSK': 'DMSK',
			'mmc2_PhysChem_No_Con': 'PhysChem No Cons',
			'mmc2_PhysChem': 'PhysChem',
			'mmc2_BD': 'BERT+DMSK',
			'mmc2_BN': 'BERT+PhysChem\nNo Cons',
			'mmc2_BP': 'BERT+PhysChem',
		}
	)
	master_df['Source'] = master_df['Train_dataset']
	master_df['Source'] = master_df['Source'].apply(lambda x: 'DRGN' if 'DRGN' in x else 'docm')
	master_df.rename(columns={"Test_dataset": "Feature"}, inplace=True)
	master_df.sort_values(by=['Feature', 'Train_dataset'], inplace=True)

	master_df.reset_index(drop=True, inplace=True)
	# pprint(df)
	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=master_df,
					col="Feature", errorbar="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_xlabel('Metric')
	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.ylim(0, 1)

	# plt.show()
	plt.savefig(f"{plot_location}/mmc2.png", bbox_inches='tight')
	plt.close()

	data_list = [(*key, val) for key, val in result_dict.items() if
				 ('BERT' not in key[0] or key[5] == 'BERT_1') and key[3] == key[4]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Feature', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Source', 'Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	df['Source'] = df['Train_dataset']
	df['Source'] = df['Source'].apply(lambda x: 'DRGN' if 'DRGN' in x else 'docm')
	df = df[columns]
	df = df[df['Train_metric'] == 'auROC']

	df['Test_dataset'] = df['Test_dataset'].replace(
		{
			'mmc2_BERT': 'BERT',
			'mmc2_DMSK': 'DMSK',
			'mmc2_PhysChem_3_pos_neg_No_Con': 'PhysChem No Cons',
			'mmc2_PhysChem_No_Con': 'PhysChem No Cons',
			'mmc2_PhysChem': 'PhysChem'
		}
	)
	df.rename(columns={"Test_dataset": "Feature", "Train_metric": "Metric"}, inplace=True)
	df = df[df['Feature'] == 'BERT']
	columns = ['Source', 'Feature', 'Model', 'Metric', 'Score']
	df = df[columns]
	df.sort_values(by=['Source', 'Feature'], inplace=True)
	df.reset_index(drop=True, inplace=True)
	pprint(df)

	sns.catplot(x="Metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, errorbar="sd")

	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.savefig(f"{plot_location}/mmc2_models.png", bbox_inches='tight')
	plt.close()

	sns.catplot(x="Metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df[df['Source'] == 'DRGN'],
				errorbar="sd")
	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.savefig(f"{plot_location}/mmc2_DRGN_models.png", bbox_inches='tight')
	plt.close()

	sns.catplot(x="Metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df[df['Source'] == 'docm'],
				errorbar="sd")
	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.savefig(f"{plot_location}/mmc2_docm_models.png", bbox_inches='tight')
	plt.close()

	sns.catplot(x="Model", y="Score", hue="Feature", kind="bar", data=master_df[master_df['Model'] == "GB"],
				errorbar="sd")
	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.savefig(f"{plot_location}/mmc2_GB.png", bbox_inches='tight')
	plt.close()

	sns.catplot(x="Model", y="Score", hue="Source", kind="bar", col="Feature",
				data=master_df[(master_df['Model'] == "GB") & (master_df['Feature'] == 'BERT+PhysChem\nNo Cons')], errorbar="sd")
	plt.ylabel("auROC", fontdict={"fontsize": "large"})
	plt.savefig(f"{plot_location}/mmc2_training.png", bbox_inches='tight')
	plt.close()


# df = df[df['Model'] != 'Random']
# df = df[df['Model'] != 'WeightedRandom']
# df = df[df['Model'] != 'Frequent']
# df['Source'] = df['Train_dataset']
# df['Source'] = df['Source'].apply( lambda x:  'DRGN' if 'DRGN' in x else 'docm' )
# df = df[['Source', 'Feature', 'Model', 'Train_metric', 'Score']]
# df.rename(columns={"Train_metric": "Metric"}, inplace=True)
# print(df)

# sub_df_list = []
# model_set = list(AVL_Set(df['Model']))
# source_set = list(AVL_Set(df['Source']))
# for selected_model in model_set:
# 	for source in source_set:
# 		df_model = df[df['Model'] == selected_model]
# 		df_model = df_model[df_model['Source'] == source]
# 		# df_model['Score_Norm'] = 2 * (df_model['Score'] - df_model['Score'].min()) / ( df_model['Score'].max() - df_model['Score'].min() ) - 1
# 		df_model['Score_Norm'] = 2 * (df_model['Score'] - df_model['Score'].mean()) / ( df_model['Score'].std() )
# 		sub_df_list.append(df_model)
# df = pd.concat(sub_df_list, ignore_index=True, sort=True)


# sns.catplot(x="Metric", y="Score_Norm", hue="Feature", kind="bar", data=df, errorbar="sd")
# plt.title("Scores on the mmc2 dataset (normalized by model and source)")
# # plt.show()
# plt.close()
# plt.savefig(f"{plot_location}/mmc2_models_norm_by_model_and_source.png", bbox_inches="tight")


# sub_df_list = []
# model_set = list(AVL_Set(df['Model']))
# for selected_model in model_set:
# 	df_model = df[df['Model'] == selected_model]
# 	# df_model['Score_Norm'] = 2 * (df_model['Score'] - df_model['Score'].min()) / ( df_model['Score'].max() - df_model['Score'].min() ) - 1
# 	df_model['Score_Norm'] = 2 * (df_model['Score'] - df_model['Score'].mean()) / ( df_model['Score'].std() )
# 	sub_df_list.append(df_model)
# df = pd.concat(sub_df_list, ignore_index=True, sort=True)


# sns.catplot(x="Metric", y="Score_Norm", hue="Feature", kind="bar", data=df, errorbar="sd")
# plt.title("Scores on the mmc2 dataset (normalized by model)")
# # plt.show()
# plt.savefig(f"{plot_location}/mmc2_models_norm_by_model.png", bbox_inches="tight")
# plt.close()

#
# data_list = [(*key, val) for key,val in result_dict.items() if ('BERT' not in key[0] or key[5] == 'BERT_1') and '_bg' not in key[4]]
# columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
# df = pd.DataFrame(data_list, columns=columns)
# columns = ['Train_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
# df = df[columns]
# df['Train_dataset'] = df['Train_dataset'].replace({'DRGN_minus_mmc2_BERT_Intersect': 'BERT',
# 	'DRGN_minus_mmc2_DMSK_Intersect': 'DMSK',
# 	'DRGN_minus_mmc2_PhysChem_Intersect': 'PhysChem',
# 	'DRGN_minus_mmc2_PhysChem_Intersect_No_Con': 'PhysChem\nWithout Conservation'})
# df.sort_values(by=columns, inplace=True)
# df.reset_index(drop=True, inplace=True)
#
#
# train_set = list(AVL_Set(df['Train_dataset']))
# metric_set  = list(AVL_Set(df['Train_metric']))
# fig, axes = plt.subplots(len(metric_set), len(metric_set), sharex=True, figsize=(20,8))
# fig.suptitle('MMC2 when using different metrics to train and test')
# for i, test_metric in enumerate(metric_set):
# 	for j, train_metric in enumerate(metric_set):
# 		h = df[df['Train_metric'] == train_metric]
# 		h = h[h['Test_metric'] == test_metric]
# 		h.reset_index(drop=True, inplace=True)
# 		sns.barplot(ax=axes[i, j], x="Train_dataset", y="Score", hue="Model",  hue_order=model_order, data=h, errorbar="sd")
# 		# print(f"{i=} {j=} {train_metric} {test_metric}")
# 		axes[i, j].set_xlabel(f'')
# 		axes[-1, j].set_xlabel(f'Trained on {train_metric}')
# 		axes[i, j].set_ylim([0, 1])
# 		axes[i, j].get_legend().remove()
# 		# axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=10, ha='right')
#
# 	axes[i, 0].set_ylabel(f'Score when tested on\n{test_metric}')
# 	axes[i, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
#
# # plt.show()
# plt.savefig(f"{plot_location/mmc2_metrics.png", bbox_inches="tight")
# plt.close()


def maveDB():
	"""
	Generate the plots for the maveDB datasets
	"""
	result_file_name = 'maveDB_result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key, val in result_dict.items() if ('BERT' not in key[0] or key[5] == 'BERT_1')]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Score']
	df = df[columns]
	df['Features'] = df['Train_dataset'].replace({
		'DRGN_minus_mavedb_BERT': 'BERT',
		'docm_minus_mavedb_BERT': 'BERT',
		'DRGN_minus_mavedb_BD': 'BERT+DMSK',
		'docm_minus_mavedb_BD': 'BERT+DMSK',
		'DRGN_minus_mavedb_BP': 'BERT+PhysChem',
		'docm_minus_mavedb_BP': 'BERT+PhysChem',
		'DRGN_minus_mavedb_BN': 'BERT+PhysChem\nWithout Conservation',
		'docm_minus_mavedb_BN': 'BERT+PhysChem\nWithout Conservation',
		'DRGN_minus_mavedb_DMSK': 'DMSK',
		'docm_minus_mavedb_DMSK': 'DMSK',
		'DRGN_minus_mavedb_PhysChem': 'PhysChem',
		'docm_minus_mavedb_PhysChem': 'PhysChem',
		'DRGN_minus_mavedb_PhysChem_No_Con': 'PhysChem\nWithout Conservation',
		'docm_minus_mavedb_PhysChem_No_Con': 'PhysChem\nWithout Conservation',
	})


	df['Source'] = df['Train_dataset'].replace({
		'DRGN_minus_mavedb_BERT': 'DRGN',
		'docm_minus_mavedb_BERT': 'docm',
		'DRGN_minus_mavedb_BD': 'DRGN',
		'docm_minus_mavedb_BD': 'docm',
		'DRGN_minus_mavedb_BP': 'DRGN',
		'docm_minus_mavedb_BP': 'docm',
		'DRGN_minus_mavedb_BN': 'DRGN',
		'docm_minus_mavedb_BN': 'docm',
		'DRGN_minus_mavedb_DMSK': 'DRGN',
		'docm_minus_mavedb_DMSK': 'docm',
		'DRGN_minus_mavedb_PhysChem': 'DRGN',
		'docm_minus_mavedb_PhysChem': 'docm',
		'DRGN_minus_mavedb_PhysChem_No_Con': 'DRGN',
		'docm_minus_mavedb_PhysChem_No_Con': 'docm',
	})



	df['experiment'] = df['Test_dataset'].replace(
		dict((f"mavedb_mut_{feature}_experiment_{experiment}", experiment) for experiment, feature in
			 itertools.product(['ErbB2', 'MSH2', 'PTEN', 'PTEN VS', 'RAF', 'Src CD', 'Src SH4', 'CXCR4'],
							   ['BERT', 'BD', 'BN', 'BP', 'DMSK', 'PhysChem', 'PhysChem_No_Con']))
	)

	# df = df[ df['Train_metric'] == 'f-measure']
	df = df[df['Model'] != 'Random']
	df = df[df['Model'] != 'WeightedRandom']
	df = df[df['Model'] != 'Frequent']
	df = df[df['Train_metric'] == 'auROC']
	df = df[df['experiment'] != 'PTEN']
	df['Score'] = abs(df['Score'])
	# df = df[ (df['Test_dataset'] == 'mavedb_mut_BERT_experiment_ErbB2') | (df['Test_dataset'] == 'mavedb_mut_BERT_experiment_MSH2') | (df['Test_dataset'] == 'mavedb_mut_BERT_experiment_PTEN') | (df['Test_dataset'] == 'mavedb_mut_BERT_experiment_Src CD') | (df['Test_dataset'] == 'mavedb_mut_BERT_experiment_Src SH4')]
	df.sort_values(by=['Train_dataset', 'Test_dataset'], inplace=True)
	df.reset_index(drop=True, inplace=True)
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	pprint(df)

	# Create a plot for every dataset
	sns.catplot(x="Train_metric", y="Score", hue="experiment", kind="bar", data=df, col="Model", errorbar="sd")
	plt.ylabel("auROC", fontdict={"fontsize": "xx-large"})
	# plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"{plot_location}/maveDB.png", bbox_inches='tight')
	plt.close()

	sns.catplot(x="experiment", y="Score", hue="Model", kind="bar", data=df[(df['Model'] == 'GB') | (df['Model'] == 'Logistic')], errorbar="sd")
	plt.ylabel("Spearman", fontdict={"fontsize": "xx-large"})
	# sns.catplot(x="experiment", y="Score", hue="Model", kind="bar", data=df, errorbar="sd")
	plt.savefig(f"{plot_location}/maveDB_model.png", bbox_inches='tight')
	# plt.show()
	plt.close()

	sns.catplot(x="experiment", y="Score", hue="Features", kind="bar", col="Model", data=df[(df['Model'] == 'Logistic') & ((df['Features'] == 'BERT') | (df['Features'] == 'BERT+DMSK') | (df['Features'] == 'BERT+PhysChem') | (df['Features'] == 'DMSK'))].dropna(), errorbar="sd")
	plt.ylabel("Spearman", fontdict={"fontsize": "xx-large"})
	plt.savefig(f"{plot_location}/maveDB_feature.png", bbox_inches='tight')
	plt.close()

	sns.catplot(x="experiment", y="Score", hue="Source", col="Model", kind="bar", data=df[(df['Model'] == 'Logistic') & (df['Features'] == 'BERT+DMSK')], errorbar="sd")
	plt.ylabel("Spearman", fontdict={"fontsize": "xx-large"})
	plt.savefig(f"{plot_location}/maveDB_training.png", bbox_inches='tight')
	# plt.show()
	plt.close()



def maveDB_GB():
	"""
	Generate the plots for the maveDDDB datasets when using GB features
	"""
	'''
	Generate the plots for the maveDB datasets
	'''
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key, val in result_dict.items() if
				 len(key) == 6 and 'maved' in key[1] and key[5] == 'GB_auROC']
	data_list.extend([(*key, val) for key, val in result_dict.items() if
					  len(key) == 6 and 'DRGN_minus_mavedb_PhysChem_No_Con_Intersect' in key[0]])
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)
	df.reset_index(drop=True, inplace=True)

	print(df)

	# Create a plot for every dataset
	sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", errorbar="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.axhline(0)
		ax.set_xlabel(
			f'Metric for training on DRGN using \n{["GB PhysChem without Conservation", "All PhysChem without Conservation"][i]} features')
		ax.set_title(
			f'Scores from using \n{["GB PhysChem without Conservation", "All PhysChem without Conservation"][i]} features\n and testing on maveDB')
	plt.ylim(0, 1)

	# plt.show()
	plt.savefig(f"{plot_location}/maveDB_GB.png", bbox_inches='tight')
	plt.close()


def BERT_layers():
	"""
	Generate the plots for the BERT layers experiments
	"""
	result_file_name = 'BERT_layers.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	data_list = [(*key, val) for key, val in result_dict.items()]
	pprint(data_list)
	print(len(data_list))
	columns = ['Train_dataset', 'Model', 'Train Metric', 'Test Metric', 'Layers', 'Fold', 'Score']
	df = pd.DataFrame(data_list, columns=columns)

	df['Layers'] = df['Layers'].apply(lambda x: int(x[5:]))
	df['Dataset'] = df['Train_dataset'].apply(lambda x: x.replace('_BERT', ''))
	df['Metric'] = df['Train Metric']
	df = df[df['Model'] != 'Random']
	df = df[df['Model'] != 'WeightedRandom']
	df = df[df['Model'] != 'Frequent']

	columns = ['Dataset', 'Model', 'Metric', 'Layers', 'Fold', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)
	print(df)

	for metric in ['auROC']: #AVL_Set(df['Metric']):
		plt.title(f'DRGN {metric} vs the number of BERT layers selected')
		ax = sns.lineplot(x="Layers", y="Score", hue="Model", hue_order=model_order[:-3],
						  data=df[(df['Dataset'] == 'DRGN') & (df['Metric'] == metric)])
		sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
		plt.ylabel(metric, fontdict={"fontsize": "large"})
		plt.savefig(f"{plot_location}/DRGN_layers_{metric}.png", bbox_inches='tight')
		plt.close()

		plt.title(f'docm {metric} vs the number of BERT layers selected')
		ax = sns.lineplot(x="Layers", y="Score", hue="Model", hue_order=model_order[:-3],
						  data=df[(df['Dataset'] == 'docm') & (df['Metric'] == metric)])
		sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
		plt.ylabel(metric, fontdict={"fontsize": "large"})
		plt.savefig(f"{plot_location}/docm_layers_{metric}.png", bbox_inches='tight')
		plt.close()


def BERT_which_layer():
	"""
	Generate the plots for determining which
	BERT layer is optimal
	"""
	result_file_name = 'BERT_which_layer.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	data_list = [(*key, val) for key, val in result_dict.items()]
	pprint(data_list)
	print(len(data_list))
	columns = ['Train_dataset', 'Model', 'Train Metric', 'Test Metric', 'Layer', 'Fold', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df['Layer'] = df['Layer'].apply(lambda x: int(x[8:]))
	df['Dataset'] = df['Train_dataset'].apply(lambda x: x.replace('_BERT', ''))
	df['Metric'] = df['Train Metric']
	df = df[df['Model'] != 'Random']
	df = df[df['Model'] != 'WeightedRandom']
	df = df[df['Model'] != 'Frequent']

	columns = ['Dataset', 'Model', 'Metric', 'Layer', 'Fold', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True, ignore_index=True)
	pd.set_option('display.max_rows', None)
	print(df)

	for metric in AVL_Set(df['Metric']):
		plt.title(f'DRGN {metric} vs the the BERT layer selected')
		ax = sns.lineplot(x="Layer", y="Score", hue="Model", hue_order=model_order[:-3],
						  data=df[(df['Dataset'] == 'DRGN') & (df['Metric'] == metric)])
		sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
		# plt.show()
		plt.ylabel(metric, fontdict={"fontsize": "large"})
		plt.savefig(f"{plot_location}/DRGN_which_layer_{metric}.png", bbox_inches='tight')
		plt.close()

		plt.title(f'docm {metric} vs the the BERT layer selected')
		ax = sns.lineplot(x="Layer", y="Score", hue="Model", hue_order=model_order[:-3],
						  data=df[(df['Dataset'] == 'docm') & (df['Metric'] == metric)])
		sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
		# plt.show()
		plt.ylabel(metric, fontdict={"fontsize": "large"})
		plt.savefig(f"{plot_location}/docm_which_layer_{metric}.png", bbox_inches='tight')
		plt.close()


def mmc2_layers():
	"""
	Generate the plots for the BERT layers experiments
	"""
	result_file_name = 'mmc2_layers.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	data_list = [(*key, val) for key, val in result_dict.items()]
	pprint(data_list)
	print(len(data_list))
	columns = ['Dataset', 'Test_dataset', 'Model', 'Metric_train', 'Metric', 'Layers', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Dataset', 'Model', 'Metric', 'Layers', 'Score']
	df = df[columns]
	df.rename(columns={'Metric': 'Metric (auROC)'}, inplace=True)
	df = df[df['Metric (auROC)'] == 'auROC']
	df['Layers'] = df['Layers'].apply(lambda x: int(x[5:]))
	print(df)

	sns.catplot(x="Metric (auROC)", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, col="Layers",
				errorbar="sd")

	plt.savefig(f"{plot_location}/mmc2_layers.png", bbox_inches='tight')


# plt.show()
# plt.close()


def BERT_timeout():
	result_file_name = 'BERT_timeout.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	data_list = [(*key, val) for key, val in result_dict.items()]

	pprint(data_list)
	pd.set_option("display.max_rows", None, "display.max_columns", None)

	columns = ['Train_dataset', 'Model', 'Metric', 'Time (min)', 'Fold', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df = df[df['Metric'] == 'auROC']
	model_set = list(AVL_Set(df['Model']))
	for selected_model in model_set:
		df_model = df[df['Model'] == selected_model]
		df_model['Time (min)'] = df_model['Time (min)'].apply(lambda x: int(x.split('_')[2]))
		columns = ['Model', 'Metric', 'Time (min)', 'Fold', 'Score']
		df_model = df_model[columns]
		# df_model = df_model[df_model['Time (min)'] < 15]
		df_model['Time (min)'] = df_model['Time (min)'].replace({1440: 20})
		df_model.sort_values(by=columns, inplace=True, ignore_index=True)
		print(df_model)

		sns.lmplot(x="Time (min)", y="Score", hue="Metric", data=df_model, fit_reg=False, facet_kws={'sharex': True})
		sns.lineplot(x="Time (min)", y="Score", hue="Metric", data=df_model, legend=False)
		plt.title(f'Average fold score ({selected_model}) vs timeout')

		plt.savefig(f"{plot_location}/Timeout_{selected_model}.png", bbox_inches="tight")
	# plt.show()


def logistic_lc():
	conn = sqlite3.connect('learning_curve.db')
	c = conn.cursor()

	for model in ['logistic', 'gb']:
		Path(f"plots/{model}").mkdir(parents=True, exist_ok=True)

		# pd.set_option("display.max_rows", None, "display.max_columns", None, 'expand_frame_repr', False)

		if model == 'logistic':
			columns = ['Dataset', 'Metric', 'Penalty', 'L1 Ratio', 'C', 'Max iter', 'Score', 'Time', 'Computed']
			parameter_list = ['L1 Ratio', 'Log(C)', 'Log(Max iter)']
		elif model == 'gb':
			columns = ['Dataset', 'Metric', "max_depth", "n_estimators", "min_samples_split", "min_impurity_decrease",
					   "min_samples_leaf", 'Score', 'Time', 'Computed']
			parameter_list = ["Log(max_depth)", "Log(n_estimators)", "n_estimators", "min_samples_split",
							  "min_impurity_decrease", "min_samples_leaf"]
		command = f"""
		SELECT *
		FROM {model}
		"""
		c.execute(command)
		df = pd.DataFrame(c.fetchall(), columns=columns)
		df.sort_values(by=columns, ignore_index=True, inplace=True)
		print(df)

		if model == 'logistic':
			df['Log(Max iter)'] = np.log10(df['Max iter'])
			df['Log(C)'] = np.log10(df['C'])
		elif model == 'gb':
			df['Log(max_depth)'] = np.log2(df['max_depth'])
			df['Log(n_estimators)'] = np.log2(df['n_estimators'])

		for x in parameter_list:
			for y in ['Score', 'Time']:
				plt.title(f"{y} vs {x}")
				sns.relplot(x=x, y=y, kind="line", hue='Dataset', data=df, col='Metric', errorbar=("pi", 100))
				# sns.relplot(x=x, y=y, kind="scatter", hue='Dataset', data=df, col='Metric')
				# plt.show()
				plt.savefig(f"{plot_location}/{model}/{y}_vs_{x}")
				plt.close()

	conn.commit()
	conn.close()


def boostdm():
	results = pd.read_pickle('datasets/mmc2_DMSK.pkl')[
		['protein_id', 'reference_aa', 'protein_position', 'mutant_aa', 'label']].rename(
		columns={"reference_aa": "AA_ref", "mutant_aa": "AA_mut", "protein_position": "pos"})
	key_columns = ['protein_id', 'AA_ref', 'pos', 'AA_mut']

	conn = sqlite3.connect('diff_results.db')
	c = conn.cursor()
	command = """
	SELECT DISTINCT protein as protein_id, experiment, boostdm.AA_ref, boostdm.pos, boostdm.AA_mut, score as prob
	FROM boostdm
	NATURAL JOIN diff_results
	WHERE protein_id = protein;
	"""
	c.execute(command)
	chasm_df = pd.DataFrame(c.fetchall(), columns=['protein_id', 'experiment', 'AA_ref', 'pos', 'AA_mut', 'prob'])
	chasm_df['Model'] = 'boostdm'

	key_df = chasm_df[key_columns]
	df_list = [chasm_df]
	feature_list = ['DMSK', 'BD', 'BN']

	for train_dataset in ['DRGN', 'docm']:
		for model in ['GB']:
			for feature in feature_list:
				command = f"""
				SELECT protein as protein_id, AA_ref, pos, AA_mut, model, prob
				FROM diff_results
				WHERE train_metric = 'auROC' AND test_metric = 'auROC' AND model = '{model}' and train_alias LIKE '{train_dataset}_minus_mmc2_{feature}' AND test_alias LIKE 'mmc2_{feature}'
				"""
				c.execute(command)

				df = pd.DataFrame(c.fetchall(), columns=['protein_id', 'AA_ref', 'pos', 'AA_mut', 'Model', 'prob'])
				df['Feature'] = feature
				df['Train Dataset'] = train_dataset
				df_list.append(df)

	key_df = pd.merge(key_df, results,on=key_columns, how='inner')[key_columns]
	key_df = pd.merge(key_df, chasm_df,on=key_columns, how='inner')[key_columns]
	for df in df_list:
		key_df = pd.merge(key_df, df, on=key_columns, how='inner')[key_columns]

	chasm_df = pd.merge(chasm_df, key_df, on=key_columns, how='inner')
	chasm_df = pd.merge(chasm_df, results)
	for i, df in enumerate(df_list):
		df = pd.merge(df, key_df, on=key_columns, how='inner')
		df = df.merge(results, on=key_columns)
		df_list[i] = df

	master_df = pd.concat(df_list)
	#
	# print(f"{results=}")
	# print(f"{master_df=}")
	# print(f"{df=}")
	# print(f"{chasm_df=}")
	# print(f"{key_df=}")
	#
	protein_set = AVL_Set(master_df['protein_id'])
	print(set(master_df['Model']))
	model_set = AVL_Set(master_df['Model'])
	#
	for train_dataset in ['DRGN', 'docm']:
		for feature in feature_list:
			data_list = []
			for protein in protein_set:
				for model in model_set:
					sub_df = master_df[master_df['protein_id'] == protein]
					sub_df = sub_df[sub_df['Model'] == model]
					sub_df = sub_df[(sub_df['Feature'] == feature) | (sub_df['Model'] == 'boostdm')]
					sub_df = sub_df[(sub_df['Train Dataset'] == train_dataset) | (sub_df['Model'] == 'boostdm')]
					# print(f"{sub_df=}")
					data_list.append( (train_dataset, protein, model, roc_auc_score(sub_df['label'], sub_df['prob']) )  )
			df = pd.DataFrame(data_list, columns=['Training Dataset', 'protein', 'Model', 'auROC'])
			# print(df)

			sns.catplot(x="protein", y='auROC', hue="Model", col="Training Dataset", kind="bar", data=df, errorbar="sd")
			plt.ylabel("auROC", fontdict={"fontsize": "large"})
			# https://stackoverflow.com/a/61445717
			plt.xticks(
				rotation=45,
				horizontalalignment='right',
				fontweight='light',
				fontsize='x-large'
			)
			plt.savefig(f"{plot_location}/boostdm_{train_dataset}_mmc2_{feature}.png", bbox_inches="tight")
	pd.set_option('display.max_rows', None)
	# print(df.groupby(['Model', 'Train Dataset', 'Feature'])['auROC'].median())

	key_columns = ['protein_id', 'experiment', 'AA_ref', 'pos', 'AA_mut']

	command = """
	SELECT DISTINCT protein as protein_id, experiment, boostdm.AA_ref, boostdm.pos, boostdm.AA_mut, score as prob
	FROM boostdm
	NATURAL JOIN diff_results
	WHERE protein_id = protein;
	"""
	c.execute(command)
	results = pd.read_csv('datasets/mavedb_mut_DMSK.tsv', sep='\t')[
		['protein_id', 'experiment', 'reference_aa', 'protein_position', 'mutant_aa', 'score']].rename(
		columns={"reference_aa": "AA_ref", "mutant_aa": "AA_mut", "protein_position": "pos"})
	chasm_df = pd.DataFrame(c.fetchall(), columns=['protein_id', 'experiment', 'AA_ref', 'pos', 'AA_mut', 'prob'])
	chasm_df['Model'] = 'boostdm'

	key_df = chasm_df[key_columns]
	df_list = [chasm_df]
	# feature_list = ['DMSK', 'BD', 'BERT', 'PhysChem', 'PhysChem_No_Con', 'BP', 'BN']

	for train_dataset in ['DRGN', 'docm']:
		for model in ['Logistic']:
			for feature in feature_list:
				command = f"""
				SELECT protein as protein_id, experiment, AA_ref, pos, AA_mut, model, prob
				FROM diff_results
				WHERE train_metric = 'auROC' AND test_metric = 'spearman' AND model = '{model}' and train_alias LIKE '{train_dataset}_minus_mavedb_{feature}' AND test_alias LIKE 'mavedb_mut_{feature}%'
				"""
				c.execute(command)
				df = pd.DataFrame(c.fetchall(),
								  columns=['protein_id', 'experiment', 'AA_ref', 'pos', 'AA_mut', 'Model', 'prob'])
				df['Feature'] = feature
				df['Train Dataset'] = train_dataset
				print(command)
				# print(f"{train_dataset=} {model=} {df}")
				df_list.append(df)

	key_df = pd.merge(key_df, results, on=key_columns, how='inner')[key_columns]
	key_df = pd.merge(key_df, chasm_df, on=key_columns, how='inner')[key_columns]
	for df in df_list:
		key_df = pd.merge(key_df, df, on=key_columns, how='inner')[key_columns]
		key_df.drop_duplicates(inplace=True)

	chasm_df = pd.merge(chasm_df, key_df, on=key_columns, how='inner')
	chasm_df = pd.merge(chasm_df, results)
	for i, df in enumerate(df_list):
		df = pd.merge(df, key_df, on=key_columns, how='inner')
		df = df.merge(results, on=key_columns)
		df_list[i] = df

	master_df = pd.concat(df_list)
	master_df = master_df[master_df['experiment'] != 'PTEN']

	# print(f"{results=}")
	print(f"{master_df=}")
	# print(f"{df=}")
	# print(f"{chasm_df=}")
	# print(f"{key_df=}")

	experiment_set = AVL_Set(master_df['experiment'])
	model_set = AVL_Set(master_df['Model'])
	everything_data_list = []
	for train_dataset in ['DRGN', 'docm']:
		for feature in feature_list:
			data_list = []
			for experiment in experiment_set:
				for model in model_set:
					sub_df = master_df[master_df['experiment'] == experiment]
					sub_df = sub_df[sub_df['Model'] == model]
					sub_df = sub_df[(sub_df['Feature'] == feature) | (sub_df['Model'] == 'boostdm')]
					sub_df = sub_df[(sub_df['Train Dataset'] == train_dataset) | (sub_df['Model'] == 'boostdm')]
					# print(f"{sub_df=}")
					score = abs(stats.spearmanr(stats.rankdata(sub_df['score']), stats.rankdata(sub_df['prob'])).correlation)
					data_list.append((train_dataset, experiment, model, score))
					everything_data_list.append((experiment, model, train_dataset, feature, score))
			df = pd.DataFrame(data_list, columns=['Training Dataset', 'experiment', 'Model', 'spearman'])
			# print(df)

			sns.catplot(x="experiment", y='spearman', hue="Model", col="Training Dataset", kind="bar", data=df, errorbar="sd")
			plt.ylabel("spearman", fontdict={"fontsize": "large"})
			plt.xticks(
				rotation=45,
				horizontalalignment='right',
				fontweight='light',
				fontsize='x-large'
			)
			plt.savefig(f"{plot_location}/boostdm_{train_dataset}_mavedb_{feature}.png", bbox_inches="tight")
	df = pd.DataFrame(everything_data_list, columns=['experiment', 'Model', 'Train Dataset', 'Feature', 'spearman'])
	# pd.set_option('display.max_rows', None)
	# print(df.groupby(['Model', 'Train Dataset', 'Feature'])['auROC'].median())

	conn.commit()
	conn.close()


def chasmplus():
	results = pd.read_pickle('datasets/mmc2_DMSK.pkl')[
		['protein_id', 'reference_aa', 'protein_position', 'mutant_aa', 'label']].rename(
		columns={"reference_aa": "AA_ref", "mutant_aa": "AA_mut", "protein_position": "pos"})
	key_columns = ['protein_id', 'AA_ref', 'pos', 'AA_mut']

	conn = sqlite3.connect('diff_results.db')
	c = conn.cursor()

	command = """
	SELECT chasmplus.protein_id, chasmplus.AA_ref, chasmplus.pos, chasmplus.AA_mut, score as prob
	FROM chasmplus
	"""
	c.execute(command)

	chasm_df = pd.DataFrame(c.fetchall(), columns=['protein_id', 'AA_ref', 'pos', 'AA_mut', 'prob'])
	chasm_df['Model'] = 'chasmplus'

	key_df = chasm_df[key_columns]
	df_list = [chasm_df]
	# feature_list = ['DMSK', 'BD', 'BERT', 'PhysChem', 'PhysChem_No_Con', 'BP', 'BN']
	feature_list = ['DMSK', 'BD', 'BN']

	for train_dataset in ['DRGN', 'docm']:
		for model in ['GB']:
			for feature in feature_list:
				command = f"""
				SELECT protein as protein_id, AA_ref, pos, AA_mut, model, prob
				FROM diff_results
				WHERE train_metric = 'auROC' AND test_metric = 'auROC' AND model = '{model}' and train_alias LIKE '{train_dataset}_minus_mmc2_{feature}' AND test_alias LIKE 'mmc2_{feature}'
				"""
				c.execute(command)

				df = pd.DataFrame(c.fetchall(), columns=['protein_id', 'AA_ref', 'pos', 'AA_mut', 'Model', 'prob'])
				df['Feature'] = feature
				df['Train Dataset'] = train_dataset
				df_list.append(df)

	key_df = pd.merge(key_df, results, on=key_columns, how='inner')[key_columns]
	key_df = pd.merge(key_df, chasm_df, on=key_columns, how='inner')[key_columns]
	for df in df_list:
		key_df = pd.merge(key_df, df, on=key_columns, how='inner')[key_columns]

	chasm_df = pd.merge(chasm_df, key_df, on=key_columns, how='inner')
	chasm_df = pd.merge(chasm_df, results)
	for i, df in enumerate(df_list):
		df = pd.merge(df, key_df, on=key_columns, how='inner')
		df = df.merge(results, on=key_columns)
		df_list[i] = df

	master_df = pd.concat(df_list)

	print(f"{results=}")
	print(f"{master_df=}")
	print(f"{df=}")
	print(f"{chasm_df=}")
	print(f"{key_df=}")

	protein_set = AVL_Set(master_df['protein_id'])
	model_set = AVL_Set(master_df['Model'])

	everything_data_list = []
	for train_dataset in ['DRGN', 'docm']:
		for feature in feature_list:
			data_list = []
			for protein in protein_set:
				for model in model_set:
					sub_df = master_df[master_df['protein_id'] == protein]
					sub_df = sub_df[sub_df['Model'] == model]
					sub_df = sub_df[(sub_df['Feature'] == feature) | (sub_df['Model'] == 'chasmplus')]
					sub_df = sub_df[(sub_df['Train Dataset'] == train_dataset) | (sub_df['Model'] == 'chasmplus')]
					# print(f"{sub_df=}")
					data_list.append((train_dataset, protein, model, roc_auc_score(sub_df['label'], sub_df['prob'])))
					everything_data_list.append(
						(protein, model, train_dataset, feature, roc_auc_score(sub_df['label'], sub_df['prob'])))
			df = pd.DataFrame(data_list, columns=['Training Dataset', 'protein', 'Model', 'auROC'])
			# print(df)

			sns.catplot(x="protein", y='auROC', hue="Model", col="Training Dataset", kind="bar", data=df, errorbar="sd")
			plt.ylabel("auROC", fontdict={"fontsize": "large"})
			plt.xticks(
				rotation=45,
				horizontalalignment='right',
				fontweight='light',
				fontsize='x-large'
			)
			plt.savefig(f"{plot_location}/chasmplus_{train_dataset}_mmc2_{feature}.png", bbox_inches="tight")
	df = pd.DataFrame(everything_data_list, columns=['protein', 'Model', 'Train Dataset', 'Feature', 'auROC'])
	# pd.set_option('display.max_rows', None)
	# print(df.groupby(['Model', 'Train Dataset', 'Feature'])['auROC'].median())

	key_columns = ['protein_id', 'experiment', 'AA_ref', 'pos', 'AA_mut']

	command = """
	SELECT DISTINCT chasmplus.protein_id, experiment, chasmplus.AA_ref, chasmplus.pos, chasmplus.AA_mut, score as prob
	FROM chasmplus
	NATURAL JOIN diff_results
	WHERE protein_id = protein;
	"""
	c.execute(command)
	results = pd.read_csv('datasets/mavedb_mut_DMSK.tsv', sep='\t')[
		['protein_id', 'experiment', 'reference_aa', 'protein_position', 'mutant_aa', 'score']].rename(
		columns={"reference_aa": "AA_ref", "mutant_aa": "AA_mut", "protein_position": "pos"})
	chasm_df = pd.DataFrame(c.fetchall(), columns=['protein_id', 'experiment', 'AA_ref', 'pos', 'AA_mut', 'prob'])
	chasm_df['Model'] = 'chasmplus'

	key_df = chasm_df[key_columns]
	df_list = [chasm_df]
	feature_list = ['DMSK', 'BD', 'BERT', 'PhysChem', 'PhysChem_No_Con', 'BP', 'BN']
	# feature_list = ['DMSK', 'BD']

	for train_dataset in ['DRGN', 'docm']:
		for model in ['Logistic']:
			for feature in feature_list:
				command = f"""
				SELECT protein as protein_id, experiment, AA_ref, pos, AA_mut, model, prob
				FROM diff_results
				WHERE train_metric = 'auROC' AND test_metric = 'spearman' AND model = '{model}' and train_alias LIKE '{train_dataset}_minus_mavedb_{feature}' AND test_alias LIKE 'mavedb_mut_{feature}%'
				"""
				c.execute(command)
				df = pd.DataFrame(c.fetchall(),
								  columns=['protein_id', 'experiment', 'AA_ref', 'pos', 'AA_mut', 'Model', 'prob'])
				df['Feature'] = feature
				df['Train Dataset'] = train_dataset
				df_list.append(df)

	key_df = pd.merge(key_df, results, on=key_columns, how='inner')[key_columns]
	key_df = pd.merge(key_df, chasm_df, on=key_columns, how='inner')[key_columns]
	for df in df_list:
		key_df = pd.merge(key_df, df, on=key_columns, how='inner')[key_columns]

	chasm_df = pd.merge(chasm_df, key_df, on=key_columns, how='inner')
	chasm_df = pd.merge(chasm_df, results)
	for i, df in enumerate(df_list):
		df = pd.merge(df, key_df, on=key_columns, how='inner')
		df = df.merge(results, on=key_columns)
		df_list[i] = df

	master_df = pd.concat(df_list)
	master_df = master_df[master_df['experiment'] != 'PTEN']

	print(f"{results=}")
	print(f"{master_df=}")
	print(f"{df=}")
	print(f"{chasm_df=}")
	print(f"{key_df=}")

	experiment_set = AVL_Set(master_df['experiment'])
	model_set = AVL_Set(master_df['Model'])

	everything_data_list = []
	for train_dataset in ['DRGN', 'docm']:
		for feature in feature_list:
			data_list = []
			for experiment in experiment_set:
				for model in model_set:
					sub_df = master_df[master_df['experiment'] == experiment]
					sub_df = sub_df[sub_df['Model'] == model]
					sub_df = sub_df[(sub_df['Feature'] == feature) | (sub_df['Model'] == 'chasmplus')]
					sub_df = sub_df[(sub_df['Train Dataset'] == train_dataset) | (sub_df['Model'] == 'chasmplus')]
					# print(f"{sub_df=}")
					score = abs(stats.spearmanr(stats.rankdata(sub_df['score']), stats.rankdata(sub_df['prob'])).correlation)
					data_list.append((train_dataset, experiment, model, score))
					everything_data_list.append((experiment, model, train_dataset, feature, score))
			df = pd.DataFrame(data_list, columns=['Training Dataset', 'experiment', 'Model', 'spearman'])
			# print(df)

			sns.catplot(x="experiment", y='spearman', hue="Model", col="Training Dataset", kind="bar", data=df, errorbar="sd")
			plt.ylabel("auROC", fontdict={"fontsize": "large"})
			plt.xticks(rotation=45)
			plt.savefig(f"{plot_location}/chasmplus_{train_dataset}_mavedb_{feature}.png", bbox_inches="tight")
	df = pd.DataFrame(everything_data_list, columns=['experiment', 'Model', 'Train Dataset', 'Feature', 'spearman'])
	# pd.set_option('display.max_rows', None)
	# print(df.groupby(['Model', 'Train Dataset', 'Feature'])['auROC'].median())

	conn.commit()
	conn.close()


def main():
	train_and_test_metrics()
	DRGN()
	docm()
	mmc2()
	maveDB()
	BERT_layers()
	BERT_which_layer()
	boostdm()
	chasmplus()
	# logistic_lc()


if __name__ == '__main__':
	main()
