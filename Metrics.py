import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score, label_ranking_loss



""" Function for computing the metrics """
def compute_metrics(y_true, y_pred):
    res = dict()

    res['HL'] = hamming_loss(y_true = y_true, y_pred = y_pred)
    res['EMR'] = accuracy_score(y_true = y_true, y_pred = y_pred)
    res['jaccard-m'] = jaccard_score(y_true = y_true, y_pred = y_pred, average = 'micro')
    res['F1-m'] = f1_score(y_true = y_true, y_pred = y_pred, average = 'micro')
    res['jaccard-M'] = jaccard_score(y_true = y_true, y_pred = y_pred, average = 'macro')
    res['F1-M'] = f1_score(y_true = y_true, y_pred = y_pred, average = 'macro')
    res['jaccard-s'] = jaccard_score(y_true = y_true, y_pred = y_pred, average = 'samples')
    res['F1-s'] = f1_score(y_true = y_true, y_pred = y_pred, average = 'samples')
    res['RL'] = label_ranking_loss(y_true = y_true, y_score = y_pred)

    return res



if __name__ == '__main__':
    y_true = np.random.randint(2, size=(5,10))
    y_pred = np.random.randint(2, size=(5,10))

    res = compute_metrics(y_true = y_true, y_pred = y_pred)

    pass