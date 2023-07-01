'''

Generates a random dataset based off of an inputted dataset

@author: Alfredo Velasco
'''

import argparse
import os

import numpy as np
import pandas as pd


def create_random_dataset(input_file, header, features_start, output_file):
    if not isinstance(features_start, int) or features_start < 0:
        raise Exception(f"{features_start=} should be a non-negative integer!")
    filen, file_ext = os.path.splitext(input_file)
    
    # Reads in and processes training/testing data
    if file_ext == ".pkl":
        input_data = pd.read_pickle(input_file)
    elif file_ext == ".tsv":
        input_data = pd.read_csv(input_file, header = 0, sep='\t', comment='#').dropna()
    else:
        raise Exception(f"Unrecognized input file format {file_ext}!")

    filen, file_ext = os.path.splitext(output_file)



    # Remove cases where mutant amino acids match the reference amino acids
    input_data = input_data[input_data["reference_aa"] != input_data["mutant_aa"]].reset_index(drop = True)

    # Binarize objective label column
    input_data['label'] = input_data["label"].replace("positives", 1).replace("negatives", 0)



    # Drop Duplicate columns
    input_data = input_data.drop_duplicates().reset_index(drop = True)


    # Subset features and labels
    features = input_data.iloc[:,features_start:(len(input_data.columns))].to_numpy()

    np.random.seed(4302022)
    a = np.random.rand(features.shape[0], features.shape[1]) * (np.max(features) - np.min(features)) + np.min(features)
    input_data.iloc[:,features_start:(len(input_data.columns))] = a


    if file_ext == '.tsv':
        input_data.to_csv(output_file, sep='\t', header=True, index=False)
    elif file_ext == ".pkl":
        input_data.to_pickle(output_file)
    else:
        raise Exception(f"Unrecognized output file format {file_ext}!")




def main():
    parser = argparse.ArgumentParser(description="A program to generate random data based off of a real dataset")

    parser.add_argument("--input", "-i", type=str, required=True, help="Name of the file that serves as the basis of the output file")
    parser.add_argument("--output-file", "-o", type=str, help="Name of the output testing file")
    
    parser.add_argument("--header", type=str, required=True, help="The header name of the proteins to keep together")
    parser.add_argument("--start-index", "-s", default=0, type=int, help="Index of column containing first feature.")    

    args = parser.parse_args()
    
    create_random_dataset(args.input, args.header, args.start_index, args.output_file)

if __name__ == '__main__':
    main()
