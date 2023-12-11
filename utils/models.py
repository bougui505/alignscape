import pickle
import functools
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter
from alignscape.align_scape import get_blosum62,seqmetric
from alignscape.utils import jax_imports
from alignscape.utils import minsptree
from alignscape.quicksom import som

def get_max_occurance(li):
    count_keys = list(Counter(li).keys())
    count_values = list(Counter(li).values())
    m = max(count_values)
    idxs_m = [i for i, j in enumerate(count_values) if j == m]
    return [count_keys[idx] for idx in idxs_m]

def load_som(filename):
    #Load the SOM
    with open(filename, 'rb') as fileaux:
        somobj = pickle.load(fileaux)
    b62 = get_blosum62()
    if somobj.jax:
        somobj.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
    else:
       somobj.metric = functools.partial(seqmetric, b62=b62)
    somobj.metric = functools.partial(seqmetric, b62=b62)
    return somobj

def load_dmatrix(somobj):
    #Load localadj matrix. It contains the Umatrix distances between input sequences
    if not hasattr(somobj, 'localadj'):
        localadj, localadj_paths = minsptree.get_localadjmat(somobj.umat,somobj.adj,somobj.bmus,verbose=True)
        somobj.localadj = localadj
        somobj.localadj_paths = localadj_paths
    dmatrix = somobj.localadj.tocsr()
    dmatrix = dmatrix.todense()
    dmatrix = np.asarray(dmatrix)
    #Setting to inf the distances between unnconected bmus
    dmatrix[dmatrix == 0] = np.inf
    return dmatrix

def split_data(types,bmus,del_unclassified,titles=[]):
    #Split the data between classified and unclassified
    idxs_unclassified = np.squeeze(np.asarray(np.where(types == del_unclassified)))
    idxs_classified = np.squeeze(np.asarray(np.where(types != del_unclassified)))
    types_unclassified = np.delete(types,idxs_classified)
    types_classified = np.delete(types,idxs_unclassified)
    bmus_unclassified = np.delete(bmus,idxs_classified)
    bmus_classified = np.delete(bmus,idxs_unclassified)
    if len(titles)>0:
        titles_unclassified = np.delete(titles,idxs_classified)
        titles_classidied = np.delete(titles,idxs_unclassified)
        return idxs_unclassified,idxs_classified,types_unclassified,types_classified,bmus_unclassified,   bmus_classified,titles_unclassified,titles_classidied
    else:
        return idxs_unclassified,idxs_classified,types_unclassified,types_classified,bmus_unclassified,   bmus_classified

def get_bmu_type_dic(bmus,types,del_unclassified):
    bmu_type = {}
    for i,bmu in enumerate(bmus):
        if bmu not in bmu_type:
            bmu_type[bmu] = types[i]
        else:
            if bmu_type[bmu] == types[i]: pass
            else:
                if bmu_type[bmu] == del_unclassified: bmu_type[bmu] = types[i]
                elif types[i] == del_unclassified: pass
                else:
                    print('In the %d BMU there are sequences from %s and %s'%(bmu,bmu_type[bmu],          types[i]))
                    if bmu_type[bmu] == 'A-delta': bmu_type[bmu] == types[i]
                    else: continue
    return bmu_type

def get_b62_dmatrix(aln,outname=None):
    aln = AlignIO.read(open(aln), 'fasta')
    calculatorb62 = DistanceCalculator('blosum62')
    dmatrixb62 = calculatorb62.get_distance(aln)
    if outname!=None:
        with open(outname,'wb') as outp:
            pickle.dump(dmatrixb62,outp)

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
            if _idx in self.dic_types_train.keys(): pass
            else: dist_row[_idx] = np.inf
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
