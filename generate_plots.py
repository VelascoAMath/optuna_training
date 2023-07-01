"""

A program to generate plots that compare differences between the datasets

"""



from pathlib import Path
from pprint import pprint
from vel_data_structures import AVL
from vel_data_structures import AVL_Dict
from vel_data_structures import AVL_Set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import re
import seaborn as sns



model_order = ['Logistic','GB', 'GNB', 'DT','Random', 'WeightedRandom', 'Frequent']


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

	data_list = [(*key, val) for key,val in result_dict.items() if ('BERT' not in key[0] or key[5] == 'BERT_1') and key[3] == key[4]]
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

	data_list = [(*key, val) for key,val in result_dict.items() if ('BERT' not in key[0] or key[5] == 'BERT_1') and key[3] == key[4]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Feature', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Source', 'Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	df['Source'] = df['Train_dataset']
	df['Source'] = df['Source'].apply( lambda x:  'DRGN' if 'DRGN' in x else 'docm' )
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


	# sns.catplot(x="Metric", y="Score_Norm", hue="Feature", kind="bar", data=df, ci="sd")
	# plt.title("Scores on the mmc2 dataset (normalized by model and source)")
	# # plt.show()
	# plt.close()
	# plt.savefig("plots/mmc2_models_norm_by_model_and_source.png", bbox_inches="tight")


	# sub_df_list = []	
	# model_set = list(AVL_Set(df['Model']))
	# for selected_model in model_set:
	# 	df_model = df[df['Model'] == selected_model]
	# 	# df_model['Score_Norm'] = 2 * (df_model['Score'] - df_model['Score'].min()) / ( df_model['Score'].max() - df_model['Score'].min() ) - 1
	# 	df_model['Score_Norm'] = 2 * (df_model['Score'] - df_model['Score'].mean()) / ( df_model['Score'].std() )
	# 	sub_df_list.append(df_model)
	# df = pd.concat(sub_df_list, ignore_index=True, sort=True)


	# sns.catplot(x="Metric", y="Score_Norm", hue="Feature", kind="bar", data=df, ci="sd")
	# plt.title("Scores on the mmc2 dataset (normalized by model)")
	# # plt.show()
	# plt.savefig("plots/mmc2_models_norm_by_model.png", bbox_inches="tight")
	# plt.close()


	data_list = [(*key, val) for key,val in result_dict.items() if ('BERT' not in key[0] or key[5] == 'BERT_1') and '_bg' not in key[4]]
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
	plt.close()




def maveDB():
	'''
	Generate the plots for the maveDB datasets
	'''
	result_file_name = 'maveDB_result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)


	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key,val in result_dict.items() if ('BERT' not in key[0] or key[5] == 'BERT_1')]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Score']
	df = df[columns]
	df['Features'] = df['Train_dataset'].replace({
		'DRGN_minus_mavedb_BERT_Intersect': 'BERT',
		'docm_minus_mavedb_BERT': 'BERT',
		'DRGN_minus_mavedb_DMSK_Intersect': 'DMSK',
		'docm_minus_mavedb_DMSK': 'DMSK',
		'DRGN_minus_mavedb_PhysChem_Intersect': 'PhysChem',
		'docm_minus_mavedb_PhysChem': 'PhysChem',
		'DRGN_minus_mavedb_PhysChem_No_Con_Intersect': 'PhysChem\nWithout Conservation',
		'docm_minus_mavedb_PhysChem_No_Con': 'PhysChem\nWithout Conservation',
	})
	# df = df[ df['Train_metric'] == 'f-measure']
	df = df[ df['Model'] != 'Frequent']
	df = df[df['Features'] == 'BERT']
	df = df[ df['Train_metric'] == 'auROC']
	df.sort_values(by=['Train_dataset', 'Test_dataset'], inplace=True)
	df.reset_index(drop=True, inplace=True)

	pprint(df)


	# Create a plot for every dataset
	sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Features", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	# for i, ax in enumerate(g.axes.flatten()):
		# ax.axhline(0)
		# ax.set_xlabel('Metric for training on DRGN')
		# ax.set_title(f'Scores from using\n{["BERT", "DMSK", "PhysChem without Conservation", "PhysChem"][i // 2] } features and training on\n{["DRGN", "docm"][i % 2] } and testing on maveDB')
	plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"plots/maveDB.png", bbox_inches='tight')
	plt.close()

	sns.catplot(x="Train_metric", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, ci="sd")
	plt.savefig(f"plots/maveDB_model.png", bbox_inches='tight')
	# plt.show()
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
	sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Train_dataset", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.axhline(0)
		ax.set_xlabel(f'Metric for training on DRGN using \n{["GB PhysChem without Conservation", "All PhysChem without Conservation" ][i]} features')
		ax.set_title(f'Scores from using \n{["GB PhysChem without Conservation", "All PhysChem without Conservation" ][i] } features\n and testing on maveDB')
	plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"plots/maveDB_GB.png", bbox_inches='tight')
	plt.close()



def BERT_layers():
	'''
	Generate the plots for the BERT layers experiments
	'''
	result_file_name = 'BERT_layers.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pattern = re.compile('BERT_\d+')
	data_list = [(*key, val) for key,val in result_dict.items() if len(key) == 5 and key[3] is not None and pattern.fullmatch(key[3])]
	pprint(data_list)
	print(len(data_list))
	columns = ['Train_dataset', 'Model', 'Metric', 'Layers', 'Fold', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df['Layers'] = df['Layers'].apply(lambda x: int(x[5:]) )
	df = df[ df['Model'] != 'Random']
	df = df[ df['Model'] != 'WeightedRandom']
	df = df[ df['Metric'] == 'auROC']
	df.sort_values(by=columns, inplace=True)
	print(df)
	metric_set = list(AVL_Set(df['Metric']))

	plt.title('DRGN and docm auROC vs the number of BERT layers selected')
	ax = sns.lineplot(x="Layers", y="Score", hue="Model", hue_order=model_order, data=df)
	sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
			

	plt.savefig(f"plots/BERT_layers.png", bbox_inches='tight')
	plt.close()


def mmc2_layers():
	'''
	Generate the plots for the BERT layers experiments
	'''
	result_file_name = 'mmc2_layers.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	data_list = [(*key, val) for key,val in result_dict.items()]
	pprint(data_list)
	print(len(data_list))
	columns = ['Dataset', 'Test_dataset', 'Model', 'Metric_train', 'Metric', 'Layers', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Dataset', 'Model', 'Metric', 'Layers', 'Score']
	df = df[columns]
	df.rename(columns={'Metric': 'Metric (auROC)'}, inplace=True)
	df = df[ df['Metric (auROC)'] == 'auROC']
	df['Layers'] = df['Layers'].apply(lambda x: int(x[5:]) )
	print(df)


	sns.catplot(x="Metric (auROC)", y="Score", hue="Model", hue_order=model_order, kind="bar", data=df, col="Layers", ci="sd")

	plt.savefig(f"plots/mmc2_layers.png", bbox_inches='tight')
	# plt.show()
	# plt.close()


def BERT_timeout():
	result_file_name = 'BERT_timeout.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	data_list = [(*key, val) for key,val in result_dict.items()]

	pprint(data_list)
	pd.set_option("display.max_rows", None, "display.max_columns", None)

	columns = ['Train_dataset', 'Model', 'Metric', 'Time (min)', 'Fold', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df = df[df['Metric'] == 'auROC']
	model_set = list(AVL_Set(df['Model']))
	for selected_model in model_set:
		df_model = df[df['Model'] == selected_model]
		df_model['Time (min)'] = df_model['Time (min)'].apply(lambda x: int( x.split('_')[2] ) )
		columns = ['Model', 'Metric', 'Time (min)', 'Fold', 'Score']
		df_model = df_model[columns]
		# df_model = df_model[df_model['Time (min)'] < 15]
		df_model['Time (min)'] = df_model['Time (min)'].replace({1440: 20})
		df_model.sort_values(by=columns, inplace=True, ignore_index=True)
		print(df_model)

		sns.lmplot(x="Time (min)", y="Score", hue="Metric", data=df_model, fit_reg=False, facet_kws={'sharex':True})
		sns.lineplot(x="Time (min)", y="Score", hue="Metric", data=df_model, legend=False)
		plt.title(f'Average fold score ({selected_model}) vs timeout')

		plt.savefig(f"plots/Timeout_{selected_model}.png", bbox_inches="tight")
		# plt.show()


def logistic_lc():
	# (args.data_alias, args.model_name, args.scoring_metric, param["penalty"], param["l1_ratio"], param["C"], param["max_iter"])

	result_file_name = 'logistic_lc.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	for model in ['Logistic', 'GB']:
		Path(f"plots/{model}").mkdir(parents=True, exist_ok=True)
		data_list = [(*key, *val) for key,val in result_dict.items() if key[1] == model]

		pd.set_option("display.max_rows", None, "display.max_columns", None, 'expand_frame_repr', False)

		if model == 'Logistic':
			columns = ['Dataset', 'Model', 'Metric', 'Penalty', 'L1 Ratio', 'C', 'Max iter', 'Score', 'Time']
			parameter_list = ['L1 Ratio', 'Log(C)', 'Log(Max iter)']
		elif model == 'GB':
			columns = ['Dataset', 'Model', 'Metric', "max_depth", "n_estimators", "min_samples_split", "min_impurity_decrease", "min_samples_leaf" , 'Score', 'Time']
			parameter_list = ["Log(max_depth)", "n_estimators", "min_samples_split", "min_impurity_decrease", "min_samples_leaf"]
		df = pd.DataFrame(data_list, columns=columns)
		# df.sort_values(by=columns, ignore_index=True, inplace=True)

		if model == 'Logistic':
			df['Log(Max iter)'] = np.log10(df['Max iter'])
			df['Log(C)'] = np.log10(df['C'])
		elif model == 'GB':
			df['Log(max_depth)'] = np.log2(df['max_depth'])

		for x in parameter_list:
			for y in ['Score', 'Time']:
				plt.title(f"{y} vs {x}")
				sns.relplot(x=x, y=y, kind="line", hue='Dataset', data=df, col='Metric', errorbar=("pi", 100))
				# sns.relplot(x=x, y=y, kind="scatter", hue='Dataset', data=df, col='Metric')
				# plt.show()
				plt.savefig(f"plots/{model}/{y}_vs_{x}")
				plt.close()



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
