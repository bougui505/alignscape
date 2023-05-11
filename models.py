import som_seq
import quicksom.som
import quicksom
import pickle
import functools
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter


def get_max_occurance(li):
    count_keys = list(Counter(li).keys())
    count_values = list(Counter(li).values())
    m = max(count_values)
    idxs_m = [i for i, j in enumerate(count_values) if j == m]
    return [count_keys[idx] for idx in idxs_m]

class KNeighborsBMU(object):
    """
    """

    def __init__(self,k):
        """
        """
        self.k = k

    def fit(self,dmatrix, train_idxs, train_types, unclassified_idxs):
        self.dmatrix = np.copy(dmatrix)
        self.train_idxs = train_idxs
        self.unclassified_idxs = unclassified_idxs

        self.train_types = train_types
        dic_types_train = {}
        for idx,typ in zip(self.train_idxs,self.train_types):
            if idx not in dic_types_train.keys():
                dic_types_train[idx] = [typ]
            else:
                dic_types_train[idx].append(typ)
        self.dic_types_train = dic_types_train

    def score(self,test_idxs,test_types):
        predict_types = []
        for i,test_idx in enumerate(test_idxs):
            predict_type = self.predict(test_idx,idxs_toinf=test_idxs)
            predict_types.append(predict_type)
        score = sum(1 for x,y in zip(test_types,predict_types) if x == y) / len(test_types)
        return score

    def predict(self,idx,idxs_toinf=[]):
        kclosest_types = []
        dist_row = self.dmatrix[idx,:]
        #Set to inf the distances of unclassified and the ones from idxs_toinf (if not None)
        for _idx in self.unclassified_idxs:
            dist_row[_idx] = np.inf
        if len(idxs_toinf) > 0:
            for _idx in idxs_toinf:
                dist_row[_idx] = np.inf
        #Check if there are training elements in the testing bmuidx
        if idx in self.dic_types_train.keys():
            kclosest_types += self.dic_types_train[idx]
            if len(kclosest_types) >= self.k:
                predict_type = get_max_occurance(kclosest_types)[0]
                return predict_type
            else:
                _k = self.k - len(kclosest_types)
        else:
            _k = self.k
        #Get indices of remaining k-th closest bmusidx in the training set
        kclosest_idxs = np.argsort(dist_row)[:_k]
        #Get the types of the remaining k-th closest bmusidx in the training set
        i=0
        while len(kclosest_types) < self.k:
            kclosest_types += self.dic_types_train[kclosest_idxs[i]]
            i+=1
        predict_type = get_max_occurance(kclosest_types)[0]
        return predict_type

class KNeighborsB62(object):
    """
    """

    def __init__(self,k):
        """
        """
        self.k = k

    def fit(self,dmatrix, train_titles):
        self.dmatrix = dmatrix
        self.train_titles = train_titles

    def predict(self,title):
        aux_dict = {}
        for train_title in self.train_titles:
            aux_dict[train_title]=self.dmatrix[title,train_title]
        kclosest = sorted(aux_dict.items(), key = itemgetter(1))[:self.k]
        kclosest_types = [t[0].split('_')[0] for t in kclosest]
        predict_type = get_max_occurance(kclosest_types)[0]
        return predict_type

    def score(self,test_titles):
        tosum = 0
        print('----------------------------------------------')
        for i,test_title in enumerate(test_titles):
            test_type = test_title.split('_')[0]
            predict_type = self.predict(test_title)
            if test_type == predict_type: tosum +=1
        try:
            score = tosum / len(test_titles)
        except:
            score = 0
        return score
