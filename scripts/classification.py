import sys
sys.path.insert(1,'../')
from models import KNeighbors
import som_seq
import quicksom.som
import quicksom
import pickle
import functools
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plot_umat

#Load the SOM
kinomefile = '/home/ifilella/Projects/quicksom_seq/results/kinome_article/kinome.p'
with open(kinomefile, 'rb') as kinomefileaux:
    kinome_som = pickle.load(kinomefileaux)
b62 = som_seq.get_blosum62()
kinome_som.metric = functools.partial(som_seq.seqmetric, b62=b62)

#Parse all data
titles = np.asarray([label.replace('>','') for label in kinome_som.labels])
types = np.asarray([label.replace('>','').split('_')[0] for label in kinome_som.labels])
n1,n2 = kinome_som.umat.shape
idxbmus = np.asarray([np.ravel_multi_index(bmu,(n1,n2)) for bmu in kinome_som.bmus])
#
dmatrix = kinome_som.localadj.tocsr()
dmatrix = dmatrix.todense()
dmatrix = np.asarray(dmatrix)
#
dmatrix[dmatrix == 0] = np.inf
print(f'Total sequences: {len(titles)}')


#Split the data between classified and unclassified
idxs_unclassified = np.squeeze(np.asarray(np.where(types == 'OTHER')))
idxs_classified = np.squeeze(np.asarray(np.where(types != 'OTHER')))

types_unclassified = np.delete(types,idxs_classified)
types_classified = np.delete(types,idxs_unclassified)

titles_unclassified = np.delete(titles,idxs_classified)
titles_classidied = np.delete(titles,idxs_unclassified)

idxbmus_unclassified = np.delete(idxbmus,idxs_classified)
idxbmus_classified = np.delete(idxbmus,idxs_unclassified)


#Split classified data into training and testing
idxbmus_train, idxbmus_test, types_train, types_test = train_test_split(idxbmus_classified,types_classified,random_state=0)

#scores = []
#ks= []
#for k in range(15):
#    k+=1
#    knn = KNeighbors(k)
#    knn.fit(dmatrix,idxbmus_train, types_train,idxbmus_unclassified)
#    score = knn.score(idxbmus_test,types_test)
#    scores.append(score)
#    ks.append(k)
#    print(k,score)
#
#plt.plot(ks,scores)
#plt.ylabel("Accuracy")
#plt.xlabel("n_neighbors")
#plt.savefig('classification_test.png')

plot_umat._plot_umat(kinome_som.umat,list(zip(*kinome_som.bmus.T)),types,hideSeqs=False)
plot_umat._plot_msptree(kinome_som.msptree_pairs, kinome_som.msptree_paths, kinome_som.umat.shape)
plt.savefig('umat.png')

knn = KNeighbors(3)
knn.fit(dmatrix, idxbmus_classified, types_classified, idxbmus_unclassified)

for idx,idxbmu,title in zip(idxs_unclassified,idxbmus_unclassified,titles_unclassified):
    predicted_type = knn.predict(idxbmu)
    print(idxbmu,np.unravel_index(idxbmu,(n1,n2)),title,predicted_type)
    types[idx] = predicted_type

plot_umat._plot_umat(kinome_som.umat,list(zip(*kinome_som.bmus.T)),types,hideSeqs=False)
plot_umat._plot_msptree(kinome_som.msptree_pairs, kinome_som.msptree_paths, kinome_som.umat.shape)
plt.savefig('predicted_umat.png')

