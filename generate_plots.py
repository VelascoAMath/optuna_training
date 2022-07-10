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



def main():
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	pprint(result_dict)

	result_tree = AVL_Dict()
	for key, val in result_dict.items():
		result_tree[key] = val

	result_dict = result_tree
	del result_tree

	dataset_set = AVL(list({x[0] for x in result_dict if x[3] != 'spearman'}))
	model_set   = AVL(list({x[1] for x in result_dict if x[3] != 'spearman'}))
	measure_set = AVL(list({x[3] for x in result_dict if x[3] != 'spearman'}))
	print(f"{dataset_set}")
	print(f"{model_set}")
	print(f"{measure_set}")

	for dataset in dataset_set:
		for model in model_set:
			for measure in measure_set:
				key  = (dataset, model, 'label', measure)
				key2 = (dataset, model, measure, measure)
				if key in result_dict:
					print(f"{dataset} {model} {measure} = {result_dict[key]}")
				elif key2 in result_dict:
					print(f"{dataset} {model} {measure} = {result_dict[key2]}")					
				else:
					print(f"{dataset} {model} {measure} = {-1}")


	# Create a plot for every model
	for model in model_set:
		df = pd.DataFrame()
		std = pd.DataFrame()
		for dataset in dataset_set:
			for measure in measure_set:
				key  = (dataset, model, 'label', measure)
				key2 = (dataset, model, measure, measure)
				if key in result_dict:
					df.at[measure, dataset] = result_dict[key][0]
					std.at[measure, dataset] = result_dict[key][1]
				elif key2 in result_dict:
					df.at[measure, dataset] = result_dict[key2][0]
					std.at[measure, dataset] = result_dict[key2][1]
				else:
					df.at[measure, dataset] = -1
					std.at[measure, dataset] = 1

		labels = measure_set

		x = np.arange(len(labels))  # the label locations
		width = 0.1  # the width of the bars

		fig, ax = plt.subplots()

		i = 0
		for dataset in dataset_set:
			rects1 = ax.bar(x + i * width, df[dataset], width, yerr=std[dataset], label=dataset)
			ax.bar_label(rects1, rotation=75, padding=3)
			i += 1

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Scores')
		ax.set_title(f'Scores generated from {model}')
		ax.set_xticks(x, labels)
		plt.ylim(0, 1.5)
		plt.legend(loc=3, bbox_to_anchor=(0.0, 1.15), borderaxespad=0.0)

		fig.tight_layout()
		plt.savefig(f"plots/{model}.png")
		# plt.show()

	# Create a plot for every dataset
	for dataset in dataset_set:
		df = pd.DataFrame()
		std = pd.DataFrame()
		for model in model_set:
			for measure in measure_set:
				key = (dataset, model, 'label', measure)
				key2 = key2 = (dataset, model, measure, measure)
				if key in result_dict:
					df.at[measure, model] = result_dict[key][0]
					std.at[measure, model] = result_dict[key][1]
				elif key2 in result_dict:
					df.at[measure, model] = result_dict[key2][0]
					std.at[measure, model] = result_dict[key2][1]
				else:
					df.at[measure, model] = -1

		labels = measure_set

		x = np.arange(len(labels))  # the label locations
		width = 0.1  # the width of the bars

		fig, ax = plt.subplots()

		i = 0
		for model in model_set:
			rects1 = ax.bar(x + i * width, df[model], width, yerr=std[model], label=model)
			ax.bar_label(rects1, rotation=75, padding=3)
			i += 1

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Scores')
		ax.set_title(f'Scores generated from {dataset}')
		ax.set_xticks(x, labels)
		plt.ylim(0, 1.5)
		plt.legend(loc=3, bbox_to_anchor=(0.0, 1.15), borderaxespad=0.0)

		fig.tight_layout()
		plt.savefig(f"plots/{dataset}.png")
		# plt.show()


def main2():
	'''
	TODO
	'''
	result_file_name = 'result.pkl'
	with open(result_file_name, 'rb') as f:
		result_dict = pkl.load(f)

	result_tree = AVL_Dict()
	for key, val in result_dict.items():
		result_tree[key] = val

	result_dict = result_tree
	del result_tree

	dataset_set = AVL(list({x[0] for x in result_dict if x[3] == 'spearman'}))
	model_set   = AVL(list({x[1] for x in result_dict if x[3] == 'spearman'}))
	measure_set = AVL(list({x[2] for x in result_dict if x[3] == 'spearman'}))
	print(f"{dataset_set}")
	print(f"{model_set}")
	print(f"{measure_set}")

	for dataset in dataset_set:
			for model in model_set:
				for measure in measure_set:
					key = (dataset, model, measure, 'spearman')
					if key in result_dict:
						print(f"{dataset} {model} {measure} = {result_dict[key]}")
					else:
						print(f"{dataset} {model} {measure} = {-1}")

	# Create a plot for every dataset
	for dataset in dataset_set:
		df = pd.DataFrame()
		std = pd.DataFrame()
		for model in model_set:
			for measure in measure_set:
				key = (dataset, model, measure, 'spearman')
				if key in result_dict:
					df.at[measure, model] = result_dict[key][0]
					std.at[measure, model] = result_dict[key][1]
				else:
					df.at[measure, model] = -1

		labels = measure_set

		x = np.arange(len(labels))  # the label locations
		width = 0.1  # the width of the bars

		fig, ax = plt.subplots()

		i = 0
		for model in model_set:
			try:
				rects1 = ax.bar(x + i * width, df[model], width, yerr=std[model], label=model)
				ax.bar_label(rects1, rotation=75, padding=3)
				i += 1
				pass
			except Exception as e:
				i += 1
				pass
			else:
				i += 1
				pass
			finally:
				pass

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Scores')
		ax.set_title(f'Scores generated from {dataset}')
		ax.set_xticks(x, labels)
		plt.ylim(-1, 1.5)
		plt.legend(loc=3, bbox_to_anchor=(0.0, 1.15), borderaxespad=0.0)

		fig.tight_layout()
		plt.savefig(f"plots/{dataset}.png", bbox_inches='tight')
		# plt.show()



if __name__ == '__main__':
	main()
	main2()
