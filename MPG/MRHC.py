import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import hamming_loss, f1_score
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier


""" Multilabel Reduction through Homogeneous Clustering algorithm """
class MRHC():

    @staticmethod
    def getFileName(param_dict):
        return 'MRHC_Imb-{}_Merging-{}'.format(param_dict['imb'], param_dict['merging'])

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


    """ Checking cluster homogeneity """
    def checkClusterCommonLabel(self, in_elements):
        # Checking whether there is a common label in ALL elements in the set:
        common_label_vec = [len(np.nonzero(self.y_toReduce[in_elements,it]==1)[0]) == len(in_elements) for it in range(self.y_init.shape[1])]

        return True if True in common_label_vec else False


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.imb = param_dict['imb']
        self.merging = param_dict['merging']
        return


    """ Procedure for generating a new prototype """
    def _generatePrototype(self, indexes):

        r = np.median(self.X_toReduce[indexes], axis = 0)

        r_labelset = list()

        for it_label in range(self.y_toReduce.shape[1]):
            n = len(np.where(self.y_toReduce[indexes, it_label] == 1)[0])
            if self.merging == 'Base':
                r_labelset.append(1) if n > len(indexes)//2 else r_labelset.append(0)
            elif self.merging == 'PolicyI':
                r_labelset.append(1) if n > int(np.floor((len(indexes)//2)/self.IRLbl[it_label])) else r_labelset.append(0)
            elif self.merging == 'PolicyII':
                if n > len(indexes)//2:
                    r_labelset.append(1)
                elif n > 0 and self.IRLbl[it_label] > self.MeanIR:
                    r_labelset.append(1)
                else:
                    r_labelset.append(0)

        return (r, r_labelset)


    """ Method for performing the space splitting stage """
    def _spaceSplittingStage(self):

        Q = list()
        Q.append(list(range(self.X_toReduce.shape[0])))
        CS = list()

        while len(Q) > 0:
            C = Q.pop()
            if self.checkClusterCommonLabel(C) or len(C) == 1:
                CS.append(C)
            else:
                M = list()

                # Obtaining set of label-centroids:
                for it_label in range(self.y_toReduce[C].shape[1]):
                    label_indexes = np.where(self.y_toReduce[C, it_label] == 1)[0]
                    if len(label_indexes) > 0:
                        M.append(np.median(self.X_toReduce[np.array(C)[label_indexes],:], axis = 0))
                M = np.array(M) # label X n_features

                resulting_labels = list(range(len(C)))
                if len(C) > M.shape[0]  and M.shape[0] > 1:
                    # Kmeans with M as initial centroids:
                    kmeans = KMeans(n_clusters = M.shape[0], init = M)
                    kmeans.fit(np.array(self.X_toReduce[C] + 0.001, dtype = 'double'))
                    resulting_labels = kmeans.labels_
                pass

                # Create new groups and enqueue them:
                for cluster_index in np.unique(resulting_labels):
                    indexes = list(np.array(C)[np.where(resulting_labels == cluster_index)[0]])
                    Q.append(indexes)
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
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    params_dict = MRHC.getParameterDictionary()
    params_dict['imb'] = 'Base'
    X_red_imbFalse, y_red_imbFalse = MRHC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

    params_dict['imb'] = 'ExcludingSamples'
    X_red_imbTrue, y_red_imbTrue = MRHC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)


    cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
    cls_red_imbFalse = BRkNNaClassifier(k=1).fit(X_red_imbFalse, y_red_imbFalse)
    cls_red_imbTrue = BRkNNaClassifier(k=1).fit(X_red_imbTrue, y_red_imbTrue)

    y_pred_ori = cls_ori.predict(X_test)
    y_pred_red_imbFalse = cls_red_imbFalse.predict(X_test)
    y_pred_red_imbTrue = cls_red_imbTrue.predict(X_test)


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