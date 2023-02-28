import os
import glob
import argparse
import numpy as np
import pandas as pd
from skmultilearn.dataset import load_dataset
from sklearn.preprocessing import MinMaxScaler
from Metrics import compute_metrics

# MPG methods:
from MPG.ALL import ALL
from MPG.MRHC import MRHC
from MPG.MChen import MChen
from MPG.MRSP3 import MRSP3

# Classifier:
from skmultilearn.adapt import MLkNN

# Classification values:
k_values = [1]


# Params dict:
merging_policies = {
    'ALL' : ['Base'],
    'MRHC' : ['Base', 'PolicyI', 'PolicyII'],
    'MRSP3' : ['Base', 'PolicyI', 'PolicyII'],
    'MChen' : ['Base', 'PolicyI', 'PolicyII']
}

# Params dict:
red_algos_param = {
    'ALL' : [1],
    'MRHC' : [1],
    'MRSP3' : [1],
    'MChen' : [10, 50, 90],
}



# Classifiers:
classifiers = ['MLkNN']


# Corpora for the experiments:
corpora = ['bibtex', 'birds', 'Corel5k', 'emotions', 'genbase', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'scene', 'yeast']
corpora = ['rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'scene', 'yeast']

# Cases to compute:
excluding_imbalance_cases = ['Base', 'ExcludingSamples']


# General paths:
Results_root_path = 'Results'
Reduction_root_path = os.path.join(Results_root_path, 'Reduction')
if not os.path.exists(Reduction_root_path): os.makedirs(Reduction_root_path)
Classification_root_path = os.path.join(Results_root_path, 'Classification')


""" Argument parser """
def parse_arguments():
    parser = argparse.ArgumentParser(description="Multilabel PG in imbalanced scenarios")
    parser.add_argument("--ddbb", type=str, default="ALL", help="Corpus to process")
    parser.add_argument("--excl_imbalance", type=str, default="ALL", help="Cases to process")
    parser.add_argument("--norm", type=str, default="True", help="Normalize data")
    args = parser.parse_args()
    
    # Corpora:
    args.ddbb = args.ddbb.split(",")
    if 'ALL' in args.ddbb: args.ddbb = corpora

    # Excluding samples based on imbalance indicators:
    args.excl_imbalance = args.excl_imbalance.split(",")
    if 'ALL' in args.excl_imbalance: args.excl_imbalance = excluding_imbalance_cases

    # Data normalization:
    args.norm = True if args.norm.lower() == 'true' else False

    return args


""" Main function for the experiments """
def experiments(res_dict, excluding_imbalance_case, corpus_name, norm_corpus):

    # Loading current corpus:
    X_train, y_train, X_test, y_test = load_corpus(corpus_name, norm_corpus)

    # Results file:
    results_path = os.path.join(Classification_root_path, excluding_imbalance_case)
    if not os.path.exists(results_path): os.makedirs(results_path)
    dst_results_file = os.path.join(results_path, "Results_Norm-{}_ddbb-{}.csv".format(norm_corpus, corpus_name))

    # Iterating through each reduction method (including exhaustive search):
    for red_method in [ALL, MRHC, MChen, MRSP3]:
        # Iterating through every reduction parameter:
        for red_parameter in red_algos_param[red_method.__name__]:
            # Iterating through every merging policy:
            for merging_policy in merging_policies[red_method.__name__]:
                # Creating the dictionary of parameters for the reduction method:
                params_dict = red_method.getParameterDictionary()
                params_dict['red'] = red_parameter
                params_dict['imb'] = excluding_imbalance_case
                params_dict['merging'] = merging_policy

                # Destination of the reduced files:
                dst_path = os.path.join(Reduction_root_path, excluding_imbalance_case, corpus_name, red_method.getFileName(params_dict))
                if not os.path.exists(dst_path): os.makedirs(dst_path)

                # Entry in the results dictionary:
                res_dict['PG_method'] = red_method.__name__
                res_dict['Reduction_parameter'] = red_parameter
                res_dict['Prototype_merging'] = merging_policy

                # Reduced set destination file:
                X_dst_file = os.path.join(dst_path, 'X_Norm-{}.csv.gz'.format(norm_corpus))
                y_dst_file = os.path.join(dst_path, 'y_Norm-{}.csv.gz'.format(norm_corpus))

                if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
                    X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
                    y_red = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))

                else:
                    X_red, y_red = red_method().reduceSet(X_train, y_train, params_dict)

                    pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None, compression='gzip')
                    pd.DataFrame(y_red).to_csv(y_dst_file, header=None, index=None, compression='gzip')

                # Iterating through classifiers:
                for single_classifier in classifiers:
                    res_dict['classifier'] = single_classifier
                    # Iterating through k values:
                    for single_k in k_values:
                        res_dict['k'] = single_k
                        
                        # Instantiating classifier:
                        cls = eval(str(single_classifier) + '(k=' + str(single_k) + ')')

                        # Fitting classifier:
                        cls.fit(X_red, y_red)
                        
                        # Inference stage
                        y_pred = cls.predict(X_test)
                        y_pred = y_pred.toarray()

                        # Computing metrics:
                        res_dict.update(compute_metrics(y_true = y_test, y_pred = y_pred))
                        res_dict['Size'] = 100*X_red.shape[0]/X_train.shape[0]

                        # Writing results:
                        if os.path.isfile(dst_results_file):
                            in_file = pd.read_csv(dst_results_file)
                            out_file = in_file.append(res_dict, ignore_index=True)
                        else:
                            out_file = pd.DataFrame([res_dict])
                        out_file.to_csv(dst_results_file,index=False)
                        pass
                    pass
                pass
            pass
        pass
    pass
    return


""" Loading a corpus """
def load_corpus(corpus_name, norm = False):
    # Loading corpus:
    X_train, y_train, feature_names, label_names = load_dataset(corpus_name, 'train')
    X_test, y_test, feature_names, label_names = load_dataset(corpus_name, 'test')

    # Corpus to standard array:
    X_train = X_train.toarray().copy()
    y_train = y_train.toarray().copy()
    X_test = X_test.toarray().copy()
    y_test = y_test.toarray().copy()

    if norm:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


""" Function for combining the result for each DDBB into a single file """
def combineResults():
    results_path = 'Results/Classification/**'
    
    file_list = [u for u in glob.glob(results_path, recursive = True) if u.endswith('csv')]
    in_file = pd.DataFrame()
    for single_file in file_list:
        in_file = in_file.append(pd.read_csv(single_file), ignore_index=True)

    in_file.to_csv('Results/Results_Summary.csv',index=False)
    return



if __name__ == '__main__':

    args = parse_arguments()

    res_dict = dict()
    # Iterating through the different corpora:
    for corpus_name in args.ddbb:
        res_dict['DDBB'] = corpus_name
        res_dict['NormData'] = args.norm
        print("Current corpus: {}".format(corpus_name))

        # Iterating through the different imbalance cases:
        for excl_imbalance_case in args.excl_imbalance:         
            res_dict['Exclude_imbalanced_samples'] = excl_imbalance_case
            experiments(res_dict, excl_imbalance_case, corpus_name, args.norm)
        pass
    pass

    # Combining the individual results:
    combineResults()