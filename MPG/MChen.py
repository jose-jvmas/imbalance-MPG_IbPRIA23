import time
import numpy as np
from skmultilearn.dataset import load_dataset
from sklearn.metrics import pairwise_distances
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.metrics import hamming_loss, f1_score


""" Multilabel Chen algorithm """
class MChen():

    @staticmethod
    def getFileName(param_dict):
        return 'MChen_Imb-{}_Merging-{}_Red-{}'.format(param_dict['imb'], param_dict['merging'], param_dict['red'])

    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['red'] = 50
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


    """ Method for obtaining the most distant prototypes within a set of elements """
    def getMostDistantPrototypes(self, in_list):
        out1, out2 = -1, -1

        if len(in_list) > 0:
            idx1, idx2 = np.where(self.distances_dict[in_list,:][:,in_list] == np.max(self.distances_dict[in_list,:][:,in_list]))
            out1 = in_list[idx1[0]]
            out2 = in_list[idx2[0]]
        
        return out1, out2


    """ Split a set of data into two non-overlapping subsets """
    def divideBIntoSubsets(self, B, p1, p2):
        B1_indexes = [B[idx] for idx in np.where(np.array([self.distances_dict[min(u, p1)][max(u, p1)] <= self.distances_dict[min(u, p2)][max(u, p2)] for u in B]) == True)[0]]
        B2_indexes = list(sorted(list(set(B) - set(B1_indexes))))

        return B1_indexes, B2_indexes


    """ Method for checking whether current set contains several labels """
    def setContainSeveralClasses(self, in_set):
        return True if np.unique(self.y_toReduce[in_set], axis = 0).shape[0] > 1 else False


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


    """ Method for computing all pairwise distances among the instances """
    def computePairwiseDistances(self):

        self.distances_dict = pairwise_distances(X = self.X_toReduce, n_jobs = -1)

        return


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.red = param_dict['red']
        self.imb = param_dict['imb']
        self.merging = param_dict['merging']
        return


    """ Method for performing the space splitting stage """
    def _spaceSplittingStage(self):
        # Number of out elements:
        n_out = int(self.red * self.X_toReduce.shape[0]/100)

        # Precompute pairwise distances:
        self.computePairwiseDistances()

        # Out elements:
        C = [list(range(self.X_toReduce.shape[0]))]

        # Step 2:
        most_distant_prototypes_distances = list()

        bc = 0
        Qchosen = 0
        several_classes_list = list()
        prototypeIndexesQchosen = [self.getMostDistantPrototypes(C[0])]
        several_classes_list.append(self.setContainSeveralClasses(C[0]))
        most_distant_prototypes_distances.append(self.distances_dict[prototypeIndexesQchosen[0]])

        for _ in range(n_out - 1):

            # Step 3:
            B = C[Qchosen]

            # Step 4:
            p1, p2 = prototypeIndexesQchosen[Qchosen]

            # Step 5:
            B1_indexes, B2_indexes = self.divideBIntoSubsets(B, p1, p2)
            B1 = B1_indexes
            B2 = B2_indexes

            # Step 6:
            i = Qchosen
            bc += 1
            C[i] = B1
            C.append(B2)
            
            prototypeIndexesQchosen[i] = self.getMostDistantPrototypes(C[i])
            most_distant_prototypes_distances[i] = self.distances_dict[prototypeIndexesQchosen[i]]
            prototypeIndexesQchosen.append(self.getMostDistantPrototypes(C[bc]))
            most_distant_prototypes_distances.append(self.distances_dict[prototypeIndexesQchosen[bc]])
            several_classes_list[i] = self.setContainSeveralClasses(C[i])
            several_classes_list.append(self.setContainSeveralClasses(C[bc]))

            # Step 7:
            selected_case = True if True in np.array(several_classes_list) else False
            selected_indexes = np.where(np.array(several_classes_list) == selected_case)[0]
            Qchosen = selected_indexes[np.where(np.array(most_distant_prototypes_distances)[selected_indexes] == max(np.array(most_distant_prototypes_distances)[selected_indexes]))[0][0]]

            pass

        return C 


    """ Method for performing the reduction """
    def reduceSet(self, X, y, param_dict):
        # Processing parameters:
        self.processParameters(param_dict)

        # Initial assignments:
        self.X_init = X
        self.y_init = y

        # Imbalance metrics:
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
        C = self._spaceSplittingStage()

        # Perform prototype merging stage:
        X_out = list()
        y_out = list()
        for single_cluster in C:
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

    params_dict = MChen.getParameterDictionary()
    params_dict['imb'] = 'Base'
    X_red_imbFalse, y_red_imbFalse = MChen().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

    params_dict['imb'] = 'ExcludingSamples'
    X_red_imbTrue, y_red_imbTrue = MChen().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)


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