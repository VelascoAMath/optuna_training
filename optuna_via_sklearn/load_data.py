"""Optuna optimization of hyperparameters: data loading module

Reads in and processes training/testing data for the optuna_training library.
Returns objects of class dict that contain features or metadata of origin specified by keys.

  Typical usage examples:
    from optuna_via_sklearn.load_data import *
    features, labels, input_df, metadata, feature_columns = load_data(ref_paths, mut_paths, start, cols, exclude, metas, args.feature_type)
"""
import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import Normalizer
from dataclasses import dataclass, field
import os
import hashlib



def fast_non_duplicate_rows(X):
    '''
    Given a numpy array X, return a list of indices where there are no repeats
    '''
    hex_set = set()

    row_index_list = []

    i = 0
    for row in X:
        m = hashlib.sha3_256()
        m.update(bytes(row))
        row_digest = m.hexdigest()
        if row_digest not in hex_set:
            hex_set.add(row_digest)
            row_index_list.append(i)
        
        i += 1

    return row_index_list






def fast_check_for_repeating_rows(X, Y=None):
    if len(X.shape) != 2:
        raise Exception("X must be a 2D array! We received an array of dimension {X.shape}")

    if Y is not None and len(Y.shape) != 2:
        raise Exception("Y must be a 2D array! We received an array of dimension {Y.shape}")


    hex_to_row_X = {}
    hex_to_line_X = {}

    i = 0
    for row in X:
        m = hashlib.sha3_256()
        m.update(bytes(row))
        row_digest = m.hexdigest()
        if row_digest in hex_to_row_X:
            print(f"X:line {hex_to_line_X[row_digest]}{hex_to_row_X[row_digest]} == X:line {i}{row}")
        hex_to_row_X[row_digest] = row
        hex_to_line_X[row_digest] = i
        i += 1

    if Y is None:
        return

    hex_to_row_Y = {}
    hex_to_line_Y = {}

    i = 0
    for row in Y:
        m = hashlib.sha3_256()
        m.update(bytes(row))
        row_digest = m.hexdigest()
        if row_digest in hex_to_row_Y:
            print(f"Y:line {hex_to_line_Y[row_digest]}{hex_to_row_Y[row_digest]} == Y:line {i}{row}")
        hex_to_row_Y[row_digest] = row
        hex_to_line_Y[row_digest] = i
        i += 1

    for row in X:
        m = hashlib.sha3_256()
        m.update(bytes(row))
        row_digest = m.hexdigest()
        if row_digest in hex_to_row_Y:
            print(f"X:line {hex_to_line_X[row_digest]}{hex_to_row_X[row_digest]} == Y:line {hex_to_line_Y[row_digest]}{hex_to_row_Y[row_digest]}")

    for row in Y:
        m = hashlib.sha3_256()
        m.update(bytes(row))
        row_digest = m.hexdigest()
        if row_digest in hex_to_row_X:
            print(f"X:line {hex_to_line_X[row_digest]}{hex_to_row_X[row_digest]} == Y:line {hex_to_line_Y[row_digest]}{hex_to_row_Y[row_digest]}")
        



def process_data(path, features_start, exclude, convert_to_pickle=False, prediction_column="label"):

    if not isinstance(features_start, int) or features_start < 0:
        raise Exception(f"{features_start=} should be a non-negative integer!")

    if not isinstance(convert_to_pickle, bool):
        raise Exception(f"{convert_to_pickle=} should be a boolean!")

    filen, file_ext = os.path.splitext(path)
    
    # Reads in and processes training/testing data
    print("Loading " + path)
    if file_ext == ".pkl":
        input_data = pd.read_pickle(path)
    elif file_ext == ".tsv":
        input_data = pd.read_csv(path, header = 0, sep='\t', comment='#').dropna()
    else:
        raise Exception(f"Unrecognized file format {file_ext}!")

    # Remove ensembl protein IDs specified by exclude
    input_data = input_data[input_data["protein_id"].isin(exclude) == False]

    # Remove cases where mutant amino acids match the reference amino acids
    input_data = input_data[input_data["reference_aa"] != input_data["mutant_aa"]].reset_index(drop = True)

    # Binarize objective label column
    if prediction_column == "label":
        input_data['label'] = input_data["label"].replace("positives", 1).replace("negatives", 0)



    # Drop Duplicate columns
    input_data = input_data.drop_duplicates().reset_index(drop = True)



    if convert_to_pickle:
        input_data.to_pickle(f"{filen}.pkl")



    # Subset features and labels
    features = input_data.iloc[:,features_start:(len(input_data.columns))].to_numpy()
    labels = input_data[prediction_column].to_numpy()

    if prediction_column in input_data.iloc[:,features_start:(len(input_data.columns))].columns:
        raise Exception(f"{prediction_column} is part of the training data!")

    # Drop duplicate rows
    non_dup_index_list = fast_non_duplicate_rows(features)
    features = features[non_dup_index_list]
    labels = labels[non_dup_index_list]
    input_data = input_data.iloc[non_dup_index_list]
    input_data.reset_index(drop=True, inplace=True)

    fast_check_for_repeating_rows(features)

    # L2 Normalizer
    L2_features = Normalizer(norm = "l2").fit_transform(features)

    return input_data, L2_features, labels

def load_data(config):
    # Initialize data dicts
    data_path_to_dataset = {}
    metadata = {};
    
    # Load Data
    for key in config.data_paths.keys():
        if(config.data_paths[key] is None):
            continue

        print(key)
        # Reading in data
        prediction_column = None
        if key == "training":
            prediction_column = config.train_prediction_col
        elif key == "testing":
            prediction_column = config.test_prediction_col
        elif key == "data":
            prediction_column = config.prediction_col
        else:
            raise Exception(f"Unrecognized {key=}!")
        input_df, features, labels = process_data(config.data_paths[key], config.start[key], config.exclude[key],
            convert_to_pickle=config.convert_to_pickle, prediction_column=prediction_column)
        
        metadata[key] = input_df.iloc[:,0:config.start[key]]
        data_path_to_dataset[key] = Dataset(input_df, features, labels)
    
    return data_path_to_dataset, metadata


@dataclass
class Dataset:
    input_df: pd.DataFrame() = field(default_factory=pd.DataFrame)
    features: np.array = field(default_factory=np.array)
    labels: np.array = field(default_factory=np.array)
    

class PCA:
    
    def __init__(self, pca_key, config):
        self.key = pca_key
        self.path = config.pca_mats[self.key]
        if "claire" in pca_key:
            handle = open(self.path, "rb")
            self.pca = pickle.load(handle)
        else:
            self.pca = faiss.read_VectorTransform(self.path)

    def apply_pca(self, input):
        if "claire" in self.key:
            output = input @ self.pca["pcamatrix"].T + self.pca["bias"]
        else:
            output = self.pca.apply_py(np.ascontiguousarray(input.astype('float32')))
        return(output)



def main():
    a = np.random.randint(0, 10, (100000, 4))
    index_list = fast_non_duplicate_rows(a)
    a = a[index_list]
    fast_check_for_repeating_rows(a)
    print(a.shape)

if __name__ == '__main__':
    main()
