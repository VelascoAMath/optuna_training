'''
Splits a file up based on the a column.
Each outputted file is a partition of the original

@author: Alfredo Velasco
'''

import os
import pandas as pd



def main(input_file, group):
	'''
	Takes a file to a tsv or pandas pkl file and partitions it based on the elements in group

	:param input_file str: a tsv or pandas pkl file to read
	:param group str: the column in the input_file which we'll use to partition the file
	:raises Exception: if input_file is not a .tsv or .pkl file 
	'''
	filen, file_ext = os.path.splitext(input_file)

	if file_ext == ".tsv":
		df = pd.read_csv(input_file, header=0, sep='\t', comment='#')
	elif file_ext == ".pkl":
		df = pd.read_pickle(input_file)
	else:
		raise Exception(f"Unsupported file extension {file_ext}!")

	for member in set(df[group]):
		print(df[df[group] == member])
		print(f"{filen}_{group}_{member}{file_ext}")

		# df[df[group] == member].to_csv(path_or_buf=f"{filen}_{group}_{member}.tsv", sep='\t', header=True, index=False)
		df[df[group] == member].to_pickle(f"{filen}_{group}_{member}.pkl")





if __name__ == '__main__':
	main('datasets/mavedb_mut_PhysChem_No_Con.tsv', 'experiment')
	main('datasets/mavedb_mut_PhysChem.tsv', 'experiment')
	main('datasets/mavedb_mut_BERT.tsv', 'experiment')
	main('datasets/mavedb_mut_DMSK.tsv', 'experiment')
	main('datasets/mavedb_mut_BD.tsv', 'experiment')
	main('datasets/mavedb_mut_BN.tsv', 'experiment')
	main('datasets/mavedb_mut_BP.tsv', 'experiment')
