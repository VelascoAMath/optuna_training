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
import re
import seaborn as sns


def DRGN():
	'''
	Generate the plots for the DRGN dataset
	'''
	result_file_name = 'BERT_results.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key,val in result_dict.items() if 'BERT' not in key[0] or key[3] == 'BERT_1']
	columns = ['Dataset', 'Model', 'Metric', 'Feature', 'Trial', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df = df[df['Dataset'].str.contains('DRGN')]
	columns = ['Dataset', 'Model', 'Metric', 'Trial', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)

	pprint(data_list)

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
			axes[i, j].set_title('')
			axes[i, j].set_ylabel(f'')
			axes[i, j].set_xlabel(f'')
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


	# See which feature set is the best
	data_list = [(*key, val) for key,val in result_dict.items() if 'BERT' not in key[0] or key[3] == 'BERT_1']
	columns = ['Dataset', 'Model', 'Metric', 'Feature', 'Trial', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Dataset', 'Model', 'Metric', 'Trial', 'Score']
	df = df[columns]
	df.rename(columns={'Dataset': 'Features'}, inplace=True)
	columns = ['Features', 'Model', 'Metric', 'Trial', 'Score']

	# Get the non-random classifiers
	df = df[df['Model'] != 'Random']
	df = df[df['Model'] != 'WeightedRandom']
	df = df[df['Model'] != 'Frequent']

	df.sort_values(by=columns, inplace=True, ignore_index=True)

	df['Features'] = df['Features'].replace({
		'DRGN_BERT_Intersect': 'BERT',
		'docm_BERT': 'BERT',
		'DRGN_DMSK_Intersect': 'DMSK',
		'docm_DMSK': 'DMSK',
		'DRGN_PhysChem_Intersect': 'PhysChem',
		'docm_PhysChem': 'PhysChem',
		'DRGN_PhysChem_Intersect_No_Con': 'PhysChem\nWithout Conservation',
		'docm_PhysChem_No_Con': 'PhysChem\nWithout Conservation',
	})
	pd.set_option("display.max_rows", None, "display.max_columns", None)
	pprint(df)

	sns.barplot(data=df, x="Metric", y="Score", hue="Features")
	plt.title('DRGN')
	plt.legend(loc='lower left')
	# plt.show()
	plt.savefig(f"plots/DRGN_feature.png", bbox_inches='tight', dpi=300)
	plt.close()





def docm():
	'''
	Generate the plots for the docm dataset
	'''
	result_file_name = 'docm_results.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	# https://stackoverflow.com/a/71446415/6373424
	data_list = [(*key, val) for key,val in result_dict.items() if 'BERT' not in key[0] or key[3] == 'BERT_1']
	columns = ['Dataset', 'Model', 'Metric', 'Feature', 'Trial', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	df = df[df['Dataset'].str.contains('docm')]
	columns = ['Dataset', 'Model', 'Metric', 'Trial', 'Score']
	df = df[columns]
	df.sort_values(by=columns, inplace=True)

	pprint(data_list)

	# Create a plot for every model
	g = sns.catplot(x="Metric", y="Score", hue="Model", kind="bar", data=df, col="Dataset", ci="sd")
	plt.ylim(0, 1)
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_title(f'Scores of docm with {["BERT", "DMSK", "PhysChem", "PhysChem without Conservation"][i] }')

	plt.savefig(f"plots/docm_dataset.png")
	# plt.show()
	plt.close()

	# Create a plot for every model and metric
	model_set = list(AVL_Set(df['Model']))
	metric_set  = list(AVL_Set(df['Metric']))

	fig, axes = plt.subplots(len(metric_set), len(model_set), sharex=True, figsize=(20,8))
	fig.suptitle('Average results of a specified metric on docm vs a specified model')

	for i, metric in enumerate(metric_set):
		for j, model in enumerate(model_set):
			h = df[df['Model'] == model]
			h = h[h['Metric'] == metric]
			h.reset_index(drop=True, inplace=True)
			sns.barplot(ax=axes[i, j], x="Metric", y="Score", hue="Dataset", data=h, ci="sd")
			print(h)
			axes[i, j].set_title('')
			axes[i, j].set_ylabel(f'')
			axes[i, j].set_xlabel(f'')
			axes[-1, j].set_xlabel(f'Trained on {model}')
			axes[i, j].set_ylim([0, 1])
			axes[i, j].get_legend().remove()
			
		axes[i, 0].set_ylabel(f'Score when tested on\n{metric}')
		axes[i, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))


	plt.savefig("plots/docm_model.png")
	# plt.show()
	plt.close()


	# See which feature set is the best
	data_list = [(*key, val) for key,val in result_dict.items() if 'BERT' not in key[0] or key[3] == 'BERT_1']
	columns = ['Dataset', 'Model', 'Metric', 'Feature', 'Trial', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Dataset', 'Model', 'Metric', 'Trial', 'Score']
	df = df[columns]
	df.rename(columns={'Dataset': 'Features'}, inplace=True)
	columns = ['Features', 'Model', 'Metric', 'Trial', 'Score']

	# Get the non-random classifiers
	df = df[df['Model'] != 'Random']
	df = df[df['Model'] != 'WeightedRandom']
	df = df[df['Model'] != 'Frequent']

	df.sort_values(by=columns, inplace=True, ignore_index=True)

	df['Features'] = df['Features'].replace({
		'DRGN_BERT_Intersect': 'BERT',
		'docm_BERT': 'BERT',
		'DRGN_DMSK_Intersect': 'DMSK',
		'docm_DMSK': 'DMSK',
		'DRGN_PhysChem_Intersect': 'PhysChem',
		'docm_PhysChem': 'PhysChem',
		'DRGN_PhysChem_Intersect_No_Con': 'PhysChem\nWithout Conservation',
		'docm_PhysChem_No_Con': 'PhysChem\nWithout Conservation',
	})
	pd.set_option("display.max_rows", None, "display.max_columns", None)
	pprint(df)

	sns.barplot(data=df, x="Metric", y="Score", hue="Features")
	plt.title('docm')
	plt.legend(loc='lower left')
	# plt.show()
	plt.savefig(f"plots/docm_feature.png", bbox_inches='tight', dpi=300)
	plt.close()

def mmc2():
	'''
	Generate the plots for the mmc2 dataset
	'''
	result_file_name = 'mmc2_results.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	# pprint(list(result_dict.keys()))

	data_list = [(*key, val) for key,val in result_dict.items() if ('BERT' not in key[0] or key[5] == 'BERT_1') and key[3] == key[4]]
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Features', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	columns = ['Train_dataset', 'Test_dataset', 'Model', 'Train_metric', 'Test_metric', 'Score']
	df = df[columns]
	df = df[df['Train_metric'] == 'auROC']
	df = df[df['Model'] != 'Frequent']

	# pprint(data_list)
	df['Test_dataset'] = df['Test_dataset'].replace(
		{
			'mmc2_BERT_Intersect'            : 'BERT',
			'mmc2_DMSK_Intersect'            : 'DMSK',
			'mmc2_PhysChem_3_pos_neg_No_Con' : 'PhysChem No Cons',
			'mmc2_PhysChem_Intersect'        : 'PhysChem'
		}
	)
	df.rename(columns={"Test_dataset": "Features"}, inplace=True)
	df.sort_values(by=['Features', 'Train_dataset'], inplace=True)
	df.reset_index(drop=True, inplace=True)
	pprint(df)
	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Features", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	for i, ax in enumerate(g.axes.flatten()):
		ax.set_xlabel('Metric')
		# ax.set_title(f'Scores from using {["BERT", "DMSK", "PhysChem without Conservation", "PhysChem"][i // 2] } features\n and training on {["DRGN", "docm"][i % 2] } and testing on mmc2')
	plt.ylim(0,1)

	# plt.show()
	plt.savefig(f"plots/mmc2.png", bbox_inches='tight')
	plt.close()

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
	df = df[ df['Train_metric'] == 'f-measure']
	df = df[ df['Model'] != 'Frequent']
	df.sort_values(by=['Train_dataset', 'Test_dataset'], inplace=True)
	df.reset_index(drop=True, inplace=True)

	pprint(df)


	# Create a plot for every dataset
	g = sns.catplot(x="Train_metric", y="Score", hue="Model", kind="bar", data=df, col="Features", ci="sd")

	# https://stackoverflow.com/a/67524391/6373424
	# for i, ax in enumerate(g.axes.flatten()):
		# ax.axhline(0)
		# ax.set_xlabel('Metric for training on DRGN')
		# ax.set_title(f'Scores from using\n{["BERT", "DMSK", "PhysChem without Conservation", "PhysChem"][i // 2] } features and training on\n{["DRGN", "docm"][i % 2] } and testing on maveDB')
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
	df.sort_values(by=columns, inplace=True)
	print(df)
	metric_set = list(AVL_Set(df['Metric']))

	fig, axes = plt.subplots(len(metric_set), 1, sharex=True, figsize=(20,8))
	fig.suptitle('DRGN and docm scores vs the number of BERT layers selected')
	for i, metric in enumerate(metric_set):
		sns.lineplot(ax=axes[i], x="Layers", y="Score",
			hue="Model", data=df[df['Metric'] == metric])
		axes[i].set_title(metric)
		# axes[i, j].set_xlabel(f'')
		# axes[-1, j].set_xlabel(f'Trained on {train_metric}')
		# axes[i, j].set_ylim([0, 1])
		# axes[i, j].get_legend().remove()
			
		# axes[i, 0].set_ylabel(f'Score when tested on\n{test_metric}')
		axes[i].legend(loc='upper right', bbox_to_anchor=(1, 0.5))

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
	df['Layers'].fillna('BERT_13', inplace=True)
	df['Layers'] = df['Layers'].apply(lambda x: int(x[5:]) )
	print(df)


	g = sns.catplot(x="Metric", y="Score", hue="Model", kind="bar", data=df, col="Layers", ci="sd")

	plt.savefig(f"plots/mmc2_layers.png", bbox_inches='tight')
	plt.show()
	plt.close()


def BERT_timeout():
	result_file_name = 'BERT_timeout.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	data_list = [(*key, val) for key,val in result_dict.items()]

	pprint(data_list)
	pd.set_option("display.max_rows", None, "display.max_columns", None)

	columns = ['Train_dataset', 'Model', 'Metric', 'Time (min)', 'Fold', 'Score']
	df = pd.DataFrame(data_list, columns=columns)
	selected_model = 'DT'
	df = df[df['Model'] == selected_model]
	df['Time (min)'] = df['Time (min)'].apply(lambda x: int( x.split('_')[2] ) )
	columns = ['Model', 'Metric', 'Time (min)', 'Fold', 'Score']
	df = df[columns]
	df = df[df['Time (min)'] < 15]
	df.sort_values(by=columns, inplace=True, ignore_index=True)
	print(df)

	sns.lmplot(x="Time (min)", y="Score", hue="Metric", data=df, fit_reg=False, facet_kws={'sharex':True})
	sns.lineplot(x="Time (min)", y="Score", hue="Metric", data=df, legend=False)
	plt.title(f'Average fold score ({selected_model}) vs timeout')

	plt.savefig(f"plots/Timeout_{selected_model}.png", bbox_inches="tight")
	plt.show()




if __name__ == '__main__':
	DRGN()
	docm()
	mmc2()
	maveDB()
	# maveDB_GB()
	BERT_layers()
	mmc2_layers()
	BERT_timeout()
