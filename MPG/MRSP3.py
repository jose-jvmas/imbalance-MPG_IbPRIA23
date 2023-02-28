import numpy as np
from sklearn.metrics import hamming_loss, f1_score
from skmultilearn.dataset import load_dataset, available_data_sets
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

import time


""" Multilabel Reduction through Space Partitioning (version 3) algorithm """
class MRSP3():

    @staticmethod
    def getFileName(param_dict):
        return 'MRSP3_Imb-{}_Merging-{}'.format(param_dict['imb'], param_dict['merging'])


    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['imb'] = 'Base' # None, ExcludingSamples
        param_dict['merging'] = 'Base' # Base, PolicyI, PolicyII
        return param_dict


    """ IRLbl and MeanIR """
    def _imbalanceMetrics(self):

        temp = [np.count_nonzero(self.y_init[:,it]) for it in range(self.y_init.shape[1])]
        temp = [1 if (u == 0) else u for u in temp]
        
        self.IRLbl = np.array([max(temp)/temp[u] for u in range(len(temp))])
        self.MeanIR = np.average(self.IRLbl)

        return


    """ Method to get the most distant prototypes in the cluster """
    def getMostDistantPrototypes(self, in_list):
        out1, out2 = -1, -1

        if len(in_list) > 0:
            idx1, idx2 = np.where(self.distances[in_list,:][:,in_list] == np.max(self.distances[in_list,:][:,in_list]))
            out1 = in_list[idx1[0]]
            out2 = in_list[idx2[0]]
        
        return out1, out2


    """ Divide cluster into subsets based on the most-distant prototypes in it """
    def divideBIntoSubsets(self, B_indexes, p1, p2):
        B1_indexes = np.array(B_indexes)[np.where(np.array([self.distances[u][p1] <= self.distances[u][p2] for u in B_indexes]) == True)[0]]
        B2_indexes = np.array(sorted(list(set(B_indexes) - set(B1_indexes))))

        if len(B2_indexes) == 0:
            B2_indexes = np.array([B1_indexes[-1]])
            B1_indexes = B1_indexes[:-1]

        return list(B1_indexes), list(B2_indexes)


    """ Checking cluster homogeneity """
    def checkClusterCommonLabel(self, in_elements):
        # Checking whether there is a common label in ALL elements in the set:
        common_label_vec = [len(np.nonzero(self.y_toReduce[in_elements, it]==1)[0]) == len(in_elements) for it in range(self.y_toReduce.shape[1])]

        return True if True in common_label_vec else False


    """ Procedure for generating a new prototype """
    def _generatePrototype(self, C):
        r = np.median(self.X_toReduce[C], axis = 0)

        r_labelset = list()

        for it_label in range(self.y_toReduce.shape[1]):
            n = len(np.where(self.y_toReduce[C, it_label] == 1)[0])

            if self.merging == 'Base':
                r_labelset.append(1) if n > len(C)//2 else r_labelset.append(0)
            elif self.merging == 'PolicyI':
                r_labelset.append(1) if n > int(np.floor((len(C)//2)/self.IRLbl[it_label])) else r_labelset.append(0)
            elif self.merging == 'PolicyII':
                if n > len(C)//2:
                    r_labelset.append(1)
                elif n > 0 and self.IRLbl[it_label] > self.MeanIR:
                    r_labelset.append(1)
                else:
                    r_labelset.append(0)

        return (r, r_labelset)


    """ Precompute pairwise distances """
    def computeDistances(self):
        self.distances = pairwise_distances(X = self.X_toReduce, n_jobs = -1)
        return


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.imb = param_dict['imb']
        self.merging = param_dict['merging']
        return


    """ Method for performing the space splitting stage """
    def _spaceSplittingStage(self):

        # Computing pairwise distances:
        self.computeDistances()

        # Indexes:
        self.indexes = list(range(self.X_toReduce.shape[0]))


        # Starting stack of elements:
        Q = list()
        Q.append(self.indexes)
        CS = list()

        # Processing element at the top of the stack:
        while len(Q) > 0:
            C = Q.pop() # Dequeing Q

            # Getting most distant elements in C:
            p1, p2 = self.getMostDistantPrototypes(C)

            if len(C) > 2:
                B1, B2 = self.divideBIntoSubsets(C, p1, p2)
            else:
                B1 = [p1]
                B2 = [p2]
            pass

            for single_partition in [B1, B2]:
                if len(single_partition) > 0:
                    if self.checkClusterCommonLabel(single_partition) or len(single_partition) == 1:
                        CS.append(single_partition)
                    else:
                        Q.append(single_partition)
                    pass
                pass
            pass
        pass
        return CS


    """ Method for performing the reduction """
    def reduceSet(self, X, y, param_dict):
        # Processing parameters:
        self.processParameters(param_dict)

        # Initial assignments:
        self.X_init = X
        self.y_init = y

        # Computing imbalance metrics:
        self._imbalanceMetrics()

        # Checking whether to address data imbalance:
        idx_excluded = list()
        idx_included = list()

        if self.imb == 'Base':
            idx_included = list(range(self.X_init.shape[0]))
        elif self.imb == 'ExcludingSamples':
            # Iterating through the samples:
            for it_sample in range(self.X_init.shape[0]):
                # Checking whether to process the sample based on imbalance ratio (IRLbl):         
                if True in list(self.IRLbl[np.where(self.y_init[it_sample] == 1)[0].tolist()] > self.MeanIR):
                    idx_excluded.append(it_sample)
                else:
                    idx_included.append(it_sample)

        # Perform space splitting stage:
        self.X_toReduce = self.X_init[idx_included, :]
        self.y_toReduce = self.y_init[idx_included, :]
        CS = self._spaceSplittingStage()

        # Perform prototype merging stage:
        X_out = list()
        y_out = list()
        for single_cluster in CS:
            if len(single_cluster) > 0:
                prot, labels = self._generatePrototype(single_cluster)
                X_out.append(prot)
                y_out.append(labels)
            pass
        pass
        
        # Adding formerly excluded labels (if there existed):
        X_out = np.append(np.array(X_out), self.X_init[idx_excluded, :], axis = 0)
        y_out = np.append(np.array(y_out), self.y_init[idx_excluded, :], axis = 0)

        return X_out, y_out



if __name__ == '__main__':
    start_time = time.time()
    X_train, y_train, feature_names, label_names = load_dataset('medical', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('medical', 'test')

    X_train = X_train.toarray()
    y_train = y_train.toarray()

    X_test = X_test.toarray()
    y_test = y_test.toarray()

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_sc = scaler.transform(X_test)
    X_train_sc = X_train
    X_test_sc = X_test

    params_dict = MRSP3.getParameterDictionary()
    params_dict['imb'] = 'Base'
    params_dict['merging'] = 'Base'
    params_dict['split'] = 'Base'
    X_red_imbFalse, y_red_imbFalse = MRSP3().reduceSet(X_train_sc, y_train, params_dict)

    params_dict['imb'] = 'Base'
    params_dict['merging'] = 'Base'
    params_dict['split'] = 'PolicyII'
    X_red_imbTrue, y_red_imbTrue = MRSP3().reduceSet(X_train_sc, y_train, params_dict)


    cls_ori = MLkNN(k=1).fit(X_train, y_train)
    cls_red_imbFalse = MLkNN(k=1).fit(X_red_imbFalse, y_red_imbFalse)
    cls_red_imbTrue = MLkNN(k=1).fit(X_red_imbTrue, y_red_imbTrue)

    y_pred_ori = cls_ori.predict(X_test_sc)
    y_pred_red_imbFalse = cls_red_imbFalse.predict(X_test_sc)
    y_pred_red_imbTrue = cls_red_imbTrue.predict(X_test_sc)


    print("Results:")
    print("\t - Hamming Loss (init): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss (imb-None): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red_imbFalse), 100*X_red_imbFalse.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss (imb-ExclSamples): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red_imbTrue), 100*X_red_imbTrue.shape[0]/X_train.shape[0]))
    print("\t" + "---" * 20)
    print("\t - F1 (init): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_ori, average = 'macro'), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - F1 (imb-None): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_red_imbFalse, average = 'macro'), 100*X_red_imbFalse.shape[0]/X_train.shape[0]))
    print("\t - F1 (imb-ExclSamples): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_red_imbTrue, average = 'macro'), 100*X_red_imbTrue.shape[0]/X_train.shape[0]))
    print("Done!")
    print("--- %s seconds ---" % (time.time() - start_time))