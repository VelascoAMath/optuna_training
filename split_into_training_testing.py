'''

Splits a file up into training and testing, but makes sure that the protein_ids all stay within the same file

@author: Alfredo Velasco
'''

import os
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_file(input_file_name, header, training_name=None, testing_name=None):
    '''
    Split a file into a training and testing file while making sure the the headers remain in the same file.
    input: 
    input_file_name - Name of the file to split
    header - name of the column that contains the headers
    output:
    training_name - name of the training file. By default, it is the input_file_name with "_train" appended
    testing_name - name of the testing file. By default, it is the input_file_name with "_test" appended

    '''
    if not isinstance(input_file_name, str):
        raise Exception(f"The input file({input_file_name}) is not a str!")
    if not isinstance(header, str):
        raise Exception(f"The header({header}) is not a str!")
    if training_name is not None and not isinstance(training_name, str):
        raise Exception(f"The training file({training_name}) is not a str!")
    if testing_name is not None and not isinstance(testing_name, str):
        raise Exception(f"The testing file({testing_name}) is not a str!")

    if training_name is None:
        f_name, f_ext = os.path.splitext(input_file_name)
        training_name = f"{f_name}_train{f_ext}"

    if testing_name is None:
        f_name, f_ext = os.path.splitext(input_file_name)
        testing_name = f"{f_name}_test{f_ext}"


    input_df = pd.read_csv(input_file_name, header=0, sep='\t', comment='#')
        
    gss = GroupShuffleSplit(n_splits=1, train_size = 2 / 3, random_state = 4122022)
    split = gss.split(input_df, groups = input_df[header])

    i = 0
    for train, test in split:
        if i > 0:
            raise Exception("We have multiple splits!")
        input_df.loc[train].to_csv(path_or_buf=training_name, sep='\t',header=True, index=False)
        input_df.loc[test].to_csv (path_or_buf=testing_name , sep='\t',header=True, index=False)
        i += 1


def main():
    parser = argparse.ArgumentParser(description="A program to split your data into training/testing while making sure that the proteins stay together")

    parser.add_argument("--input", "-i", type=str, required=True, help="File to split up")
    parser.add_argument("--header", type=str, required=True, help="The header name of the proteins to keep together")
    parser.add_argument("--training-file", type=str, help="Name of the output training file")
    parser.add_argument("--testing-file", type=str, help="Name of the output testing file")
    args = parser.parse_args()
    
    split_file(args.input, args.header, args.training_file, args.testing_file)



if __name__ == "__main__":
    main()









