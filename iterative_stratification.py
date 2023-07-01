'''
This module is supposed to replace the KFold modules in sklearn by making sure that folds in our datasets
are splitted in a way guarantees that the proteins are within the same folds but attempts to make sure than
each fold gets a desired number of items and that the proportion of positive/negative labels is roughly the same in each fold

@author: Alfredo Velasco
'''

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pprint import pprint
from vel_data_structures import List_Heap
import numpy as np
import pandas as pd
import random


@dataclass
class _Bin(object):
	'''
	Represents a bin which is a collection of indices of a list of items
	'''

	# List of indices
	bin: List_Heap = field(default_factory=List_Heap)
	# Desired size of the bin
	bin_size: int = -1
	# Desired number of positives
	pos_size: int = -1

	def __init__(self, label_list, bin_size=-1, pos_size=-1):
		self.label_list = label_list

		self.bin_size = bin_size
		self.pos_size = pos_size
		self.bin = List_Heap(min=False)


	def __post_init__(self):
		# Error checking		
		if  not isinstance(self.bin_size, int):
			raise Exception(f"bin_size should be an int but is instead type {type(self.bin_size)}!")

		if  not isinstance(self.pos_size, int):
			raise Exception(f"pos_size should be an int but is instead type {type(self.pos_size)}!")

		if self.bin_size < 1:
			raise Exception(f"bin_size is {self.bin_size} but it should be a positive integer!")

		if self.pos_size < 1:
			raise Exception(f"pos_size is {self.pos_size} but it should be a positive integer!")

	def insert_index(self, i):
		'''
		Inserts an index into our bin

		:param i: index to insert
		:raises Exception: if index is already in the bin
		'''
		bin = self.bin
	
		if i in bin:
			raise Exception("Can't insert {i} because it's already in our {bin=}!")

		bin.insert(i)


	def insert_indices(self, l):
		'''
		Inserts a collection of indices into our bin

		:param l: collection of indices to insert
		'''
		self.bin.extend(l)

	def remove_index(self, i):
		'''
		Removes an index from our bin

		:param i: index to remove
		:raises Exception: if index is not in the bin
		'''

		bin = self.bin
	
		if i not in bin:
			raise Exception("Can't remove {i} because it's not already in our {bin=}!")

		bin.remove(i)


	def remove_indices(self, l):
		'''
		Removes a collection of indeces from our bin

		:param l: list of indices to remove
		'''
		for i in l:
			self.remove_index(i)


	def score_bin(self):
		'''
		Calculates a score for our bin.
		This is a non-negative function where a score of 0 means our bin is perfectly packed		
	
		:returns int: the score of the bin 
		'''
		self.num_pos = 0
		for i in self.bin:
			if self.label_list[i] == 1:
				self.num_pos += 1


		num_pos_exp = (self.pos_size / self.bin_size) * self.bin_size

		return 1 * abs(len(self.bin) - self.bin_size) + 1 * round(abs(self.num_pos - num_pos_exp))

	def __repr__(self):
		bin = self.bin
		bin_size = self.bin_size
		pos_size = self.pos_size
		score = self.score_bin()
		if len(bin) == 0:
			# return f'_Bin({bin=} {bin_size=}, bin_len={len(bin)}, {pos_size=} {0.0:.4}%, {score=})'
			return f'_Bin({bin_size=}, bin_len={len(bin)}, {pos_size=} {0.0:.4}%, {score=})'
		# return f'_Bin({bin=} {bin_size=}, bin_len={len(bin)}, {pos_size=} {100 * self.num_pos/len(bin):.4}%, {score=})'
		return f'_Bin({bin_size=}, bin_len={len(bin)}, {pos_size=} {100 * self.num_pos/len(bin):.4}%, {score=})'


@dataclass
class StratifiedGroupKFold():
	'''
	This module is supposed to replace sklearn's StratifiedGroupKFold module.
	It returns folds where the proteins stay together, and the folds have an approximately equal number of items and propotion of positives

	'''
	n_splits: int = 5
	shuffle: bool = True
	random_state: int = 7


	# def split(self, X, y, groups):
	# 	'''
	# 	Return the splits used  by sklearn
	# 	'''
	# 	import sklearn.model_selection
	# 	gss = sklearn.model_selection.StratifiedGroupKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

	# 	n = X.shape[0]

	# 	# Calculate the number of items/positives in each fold
	# 	num_pos = int(np.sum([i for i in y]))
	# 	for i in range(self.n_splits):
	# 		bin_size = n // self.n_splits
	# 		if i < (n % self.n_splits):
	# 			bin_size += 1

	# 		pos_size = num_pos // self.n_splits
	# 		if i < (num_pos % self.n_splits):
	# 			pos_size += 1

		
	# 	# Store the folds in a list of bins
	# 	self.bin_list = [_Bin(y, bin_size=bin_size, pos_size=pos_size) for x in range(self.n_splits)]
	# 	i = 0
	# 	for train, test in gss.split(X, y, groups):
	# 		self.bin_list[i].insert_indices(test)
	# 		yield train, test
	# 		i += 1


	def split(self, X, y, groups):
		import hashlib
		'''
		Splits the inputted data into folds where identical items in a group stay within the same fold.
		This uses a heuristic to attempt to keep the fold sizes and proportion of positives in each fold identical
	
		:param X: The data we'll split up. Assumed to be a numpy 2D array
		:param y: the list of labels
		:param groups: the list of items that must be kept together. Assumed to be a pandas Series
		:yields train: the indices for the training dataset
		:yields test: the indicies for the testing dataset
		:raises Exception: if the number of items in X, y, and groups are different
		'''

		n = X.shape[0]
		
		if n != len(groups):
			raise Exception("{X=}\n and {groups=}\n are not of the same size!")

		if n != len(y):
			raise Exception("{X=}\n and {y=}\n are not of the same size!")


		# Record the collection of indices for each protein
		protein_to_index_list = defaultdict(list)

		for i, protein in groups.iteritems():
			protein_to_index_list[protein].append(i)

		protein_set = sorted(list(protein_to_index_list))

		# Shuffle so that proteins of identical frequencies can get swapped
		random.seed(self.random_state)
		random.shuffle(protein_set)

		# We want to insert more common proteins first
		protein_set.sort(key=lambda protein:len(protein_to_index_list[protein]))
		print(f"protein_set=", hashlib.sha256(bytes(f"{protein_set}", "utf-8")).hexdigest(), len(protein_set))

		# Distribute the number of items and positives amongst the bins
		num_pos = int(np.sum([i for i in y]))
		for i in range(self.n_splits):
			bin_size = n // self.n_splits
			if i < (n % self.n_splits):
				bin_size += 1

			pos_size = num_pos // self.n_splits
			if i < (num_pos % self.n_splits):
				pos_size += 1


			self.bin_list = [_Bin(y, bin_size=bin_size, pos_size=pos_size) for x in range(self.n_splits)]

		# Begin the heuristic algorithm
		# For every protein, insert its indicies into the bin that minimizes the average score amongst the bins
		while protein_set:
			protein = protein_set.pop()
			index_list = protein_to_index_list[protein]

			score_list = [self.bin_list[i].score_bin() for i in range(self.n_splits)]
			best_score = float("inf")
			best_bin = None
			for i, bin in enumerate(self.bin_list):
				old_score = score_list[i]

				bin.insert_indices(index_list)
				
				score = bin.score_bin()
				
				score_list[i] = score
				
				if np.mean(score_list) < best_score:
					best_bin = bin
					best_score = np.mean(score_list)
				bin.remove_indices(index_list)
				score_list[i] = old_score

			best_bin.insert_indices(index_list)


		# Return the indices from the bins
		for i in range(self.n_splits):
			train = List_Heap(min=False)
			test = None
			for j in range(self.n_splits):
				if i == j:
					test = self.bin_list[i].bin.item_list
				else:
					train.extend(self.bin_list[j].bin)
			if len( set(train) & set(test)) != 0:
				raise Exception(f"{train=}\nand\n{test=}\nhave overlapping indices ({set(train) & set(test)})!")

			if len( set(groups[train.item_list]) & set(groups[test])) > 0:
				raise Exception(f"protein(s) {set(groups[train]) & set(groups[test.item_list])} are found in the {train} and {test=} indices!")
			
			yield train.item_list, test




def main():
	# Number of bins
	k = 5
	# Number of protein types
	p = 25
	# Number of copies
	c = 4
	# Number of data points
	n = p * 2 * c * k
	# Column names
	column_names = ['protein_id', 'f1', 'label']
	# D = list(product(range(p), range(2))) * c * k
	D = [[None, None, None] for i in range(n)]
	split_index_list = []

	start_index = 0
	bin_size = (n // k)
	r = n % k
	for i in range(k):
		start_index += bin_size
		if i < (n % k):
			start_index += 1

		split_index_list.append(start_index)

	split_index_list.extend( random.sample([i for i in range(n) if i not in split_index_list], p - k ))
	split_index_list.sort(reverse=True)

	protein_index = 0
	for i in range(n):
		if i == split_index_list[-1]:
			protein_index += 1
			split_index_list.pop()
		D[i][0] = protein_index
		if i % 25 < 7:
			D[i][-1] = 1
		else:
			D[i][-1] = 0

	# df = pd.DataFrame(D, columns=column_names)
	for dataset in [
		'datasets/DRGN_BERT_Intersect.tsv',
		'datasets/DRGN_BERT.tsv',
		'datasets/DRGN_minus_mmc2_BERT_Intersect.tsv',
		'datasets/DRGN_minus_mmc2_BERT.tsv',
		'datasets/DRGN_minus_mmc2_PhysChem_Intersect_No_Con.tsv',
		'datasets/DRGN_minus_mmc2_PhysChem_Intersect.tsv',
		'datasets/DRGN_minus_mmc2_PhysChem_No_Con.tsv',
		'datasets/DRGN_minus_mmc2_PhysChem.tsv',
		'datasets/DRGN_PhysChem_Intersect_No_Con.tsv',
		'datasets/DRGN_PhysChem_Intersect.tsv',
		'datasets/DRGN_PhysChem_No_Con.tsv',
		'datasets/DRGN_PhysChem.tsv',
		'datasets/mmc2_BERT_3_pos_neg.tsv',
		'datasets/mmc2_BERT_Intersect.tsv',
		'datasets/mmc2_BERT.tsv',
		'datasets/mmc2_PhysChem_3_pos_neg_No_Con.tsv',
		'datasets/mmc2_PhysChem_3_pos_neg.tsv',
		'datasets/mmc2_PhysChem_Intersect_No_Con.tsv',
		'datasets/mmc2_PhysChem_Intersect.tsv',
		'datasets/mmc2_PhysChem_No_Con.tsv',
		'datasets/mmc2_PhysChem.tsv',
	]:
		print(f"{dataset=}")
		df = pd.read_csv(dataset, sep='\t')
		n = df.shape[0]


		features = df['protein_id'].to_numpy()
		y = df['label'].to_numpy()

		gss = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=7)
		split = gss.split(features, y, df["protein_id"])

		for train, test in split:
			pass
			# print(train, test)

		pprint(gss.bin_list)
		print([bin.score_bin() for bin in gss.bin_list])
		print(np.mean([bin.score_bin() for bin in gss.bin_list]))
		return



if __name__ == '__main__':
	main()
