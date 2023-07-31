import functools
import pickle
import matplotlib.pyplot as plt
import newick
import numpy as np
import networkx as nx
from quicksom_seq import som_seq
from quicksom_seq import plot_umat
from quicksom_seq.som_seq import seqmetric
from quicksom_seq.quicksom import somax
from quicksom_seq.quicksom import som
from quicksom_seq.utils import jax_imports
from quicksom_seq.utils.Timer import Timer
from quicksom_seq.utils import minsptree
from quicksom_seq.utils import models
from quicksom_seq.analysis import dmatrix
from quicksom_seq.analysis import cmatrix
from quicksom_seq.analysis import mutation_pathway

somfile = 'testout/som.pickle'
somfile_jax = 'testout/somjax.pickle'
timer = Timer(autoreset=True)

timer.start('loading the som (normal and jax)')
with open(somfile, 'rb') as somfileaux:
    somobj = pickle.load(somfileaux)
with open(somfile_jax, 'rb') as somfileaux:
    somobj_jax = pickle.load(somfileaux)
b62 = som_seq.get_blosum62()
somobj.metric = functools.partial(seqmetric, b62=b62)
somobj_jax.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
bmus = list(zip(*somobj.bmus[0:100].T))
bmus_jax = list(zip(*somobj_jax.bmus[0:100].T))
timer.stop()

timer.start('computing localadj between queries')
localadj, paths = minsptree.get_localadjmat(somobj.umat,somobj.adj,bmus)
localadj_jax, paths_jax = minsptree.get_localadjmat(somobj_jax.umat,somobj_jax.adj,bmus_jax)
timer.stop()

timer.start('computing the minsptree tree')
mstree,mstree_pairs,mstree_paths = minsptree.get_minsptree(localadj,paths,verbose=False)
mstree_jax,mstree_pairs_jax,mstree_paths_jax = minsptree.get_minsptree(localadj_jax,paths_jax,verbose=False)
timer.stop()

timer.start('computing the mstree graph')
minsptree.write_mstree_gml(mstree,somobj.bmus,somobj.labels,(somobj.m,somobj.n),outname='testout/mstree_ntw')
minsptree.write_mstree_gml(mstree_jax,somobj_jax.bmus,somobj_jax.labels,(somobj_jax.m,somobj_jax.n),outname='testout/mstree_ntw_jax')
timer.stop()

timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = minsptree.get_unfold_umat(somobj.umat, somobj.adj, bmus, mstree)
uumat_jax, mapping_jax, reversed_mapping_jax = minsptree.get_unfold_umat(somobj_jax.umat, somobj_jax.adj, bmus_jax, mstree_jax)
unfbmus = np.asarray([mapping[bmu] for bmu in bmus])
unfbmus_jax = np.asarray([mapping_jax[bmu] for bmu in bmus_jax])
somobj.uumat = uumat
somobj_jax.uumat = uumat_jax
somobj.mapping = mapping
somobj_jax.mapping = mapping_jax
somobj.reversed_mapping = reversed_mapping
somobj_jax.reversed_mapping = reversed_mapping_jax
timer.stop()

timer.start('get the minsptree paths in the unfold umat')
unf_mstree_pairs, unf_mstree_paths = minsptree.get_unfold_mstree(mstree_pairs,mstree_paths, somobj.umat.shape, somobj.uumat.shape, somobj.mapping)
unf_mstree_pairs_jax, unf_mstree_paths_jax = minsptree.get_unfold_mstree(mstree_pairs_jax,mstree_paths_jax, somobj_jax.umat.shape, somobj_jax.uumat.shape, somobj_jax.mapping)
timer.stop()

timer.start('Test plots')
plot_umat.main(somfile,outname='testout/umat',delimiter=None,hideSeqs=True,mst=False,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_remap',delimiter=None,hideSeqs=False,mst=False,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_labels',delimiter='_',hideSeqs=False,mst=False,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_minsptree',delimiter='_',hideSeqs=False,mst=True,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_unfold',delimiter='_',hideSeqs=False,mst=False,clst=False,unfold=True)
plot_umat.main(somfile,outname='testout/umat_minsptree_unfold',delimiter='_',hideSeqs=False,mst=True,clst=False,unfold=True)
plot_umat.main(somfile,outname='testout/umat_clst',delimiter='_',hideSeqs=False,mst=False,clst=True,unfold=False)
plot_umat.main(somfile,outname='testout/umat_minsptree_clst',delimiter='_',hideSeqs=False,mst=True,clst=True,unfold=False)
plot_umat.main(somfile,outname='testout/umat_unfold_clst',delimiter='_',hideSeqs=False,mst=False,clst=True,unfold=True)
plot_umat.main(somfile,outname='testout/umat_minsptree_unfold_clst',delimiter='_',hideSeqs=False,mst=True,clst=True,unfold=True)

plot_umat.main(somfile_jax,outname='testout/umat_jax',delimiter=None,hideSeqs=True,mst=False,clst=False,unfold=False)
plot_umat.main(somfile_jax,outname='testout/umat_remap_jax',delimiter=None,hideSeqs=False,mst=False,clst=False,unfold=False)
plot_umat.main(somfile_jax,outname='testout/umat_labels_jax',delimiter='_',hideSeqs=False,mst=False,clst=False,unfold=False)
plot_umat.main(somfile_jax,outname='testout/umat_minsptree_jax',delimiter='_',hideSeqs=False,mst=True,clst=False,unfold=False)
plot_umat.main(somfile_jax,outname='testout/umat_unfold_jax',delimiter='_',hideSeqs=False,mst=False,clst=False,unfold=True)
plot_umat.main(somfile_jax,outname='testout/umat_minsptree_unfold_jax',delimiter='_',hideSeqs=False,mst=True,clst=False,unfold=True)
plot_umat.main(somfile_jax,outname='testout/umat_clst_jax',delimiter='_',hideSeqs=False,mst=False,clst=True,unfold=False)
plot_umat.main(somfile_jax,outname='testout/umat_minsptree_clst_jax',delimiter='_',hideSeqs=False,mst=True,clst=True,unfold=False)
plot_umat.main(somfile_jax,outname='testout/umat_unfold_clst_jax',delimiter='_',hideSeqs=False,mst=False,clst=True,unfold=True)
plot_umat.main(somfile_jax,outname='testout/umat_minsptree_unfold_clst_jax',delimiter='_',hideSeqs=False,mst=True,clst=True,unfold=True)
timer.stop()

timer.start('Test k-neighbours')
k=1
knn = models.KNeighborsBMU(k)
titles = ['_'.join(label.split('_')[1:]) for label in somobj.labels]
types = [label.split('_')[0].replace('>','') for label in somobj.labels]
bmus = np.asarray([np.ravel_multi_index(bmu,somobj.umat.shape) for bmu in somobj.bmus])
dm = models.load_dmatrix(somobj)
idxs_unclass,idxs_class,types_unclass,types_class,bmus_unclass,bmus_class = models.split_data(np.asarray(types),np.asarray(bmus),'unk')
titles_unclass = [titles[idx] for idx in idxs_unclass]
knn.fit(dm, bmus_class, types_class, bmus_unclass)
f = open('testout/classification.csv','w')
for idx,bmu,title in zip(idxs_unclass,bmus_unclass,titles_unclass):
    predicted_type = knn.predict(bmu)
    types[idx] = predicted_type
    f.write(f'{title},{predicted_type}\n')
f.close()
plot_umat._plot_umat(somobj.umat,somobj.bmus,types,hideSeqs=False)
plot_umat._plot_mstree(mstree_pairs, mstree_paths, somobj.umat.shape)
plt.savefig('testout/umat_predicted.pdf')

knn_jax = models.KNeighborsBMU(k)
titles_jax = ['_'.join(label.split('_')[1:]) for label in somobj_jax.labels]
types_jax = [label.split('_')[0].replace('>','') for label in somobj_jax.labels]
bmus_jax = np.asarray([np.ravel_multi_index(bmu,somobj_jax.umat.shape) for bmu in somobj_jax.bmus])
dm_jax = models.load_dmatrix(somobj_jax)
idxs_unclass_jax,idxs_class_jax,types_unclass_jax,types_class_jax,bmus_unclass_jax,bmus_class_jax = models.split_data(np.asarray(types_jax),np.asarray(bmus_jax),'unk')
titles_unclass_jax = [titles[idx] for idx in idxs_unclass_jax]
knn_jax.fit(dm_jax, bmus_class_jax, types_class_jax, bmus_unclass_jax)
f = open('testout/classification_jax.csv','w')
for idx,bmu,title in zip(idxs_unclass_jax,bmus_unclass_jax,titles_unclass_jax):
    predicted_type = knn_jax.predict(bmu)
    types_jax[idx] = predicted_type
    f.write(f'{title},{predicted_type}\n')
f.close()
plot_umat._plot_umat(somobj_jax.umat,somobj_jax.bmus,types_jax,hideSeqs=False)
plot_umat._plot_mstree(mstree_pairs_jax, mstree_paths_jax, somobj_jax.umat.shape)
plt.savefig('testout/umat_predicted_jax.pdf')
timer.stop()

timer.start('Test distance matrix')
dmobj = dmatrix.Dmatrix(somfile=somfile,querieslist=somobj.labels,output='testout/dmatrix',delimiter='_')
dmobj_jax = dmatrix.Dmatrix(somfile=somfile_jax,querieslist=somobj_jax.labels,output='testout/dmatrix_jax',delimiter='_')
timer.stop()

timer.start('Test correlation matrix')
cmatrix.Cmatrix(dmatrices=['testout/dmatrix.p','testout/dmatrix_jax.p'],labels=['dm','dm_jax'],outname='testout/cmatrix')
timer.stop()

timer.start('Test mutation pathway')
mutation_pathway.main(unit1=somobj.bmus[0],unit2=somobj.bmus[1],somfile=somfile,outname='testout/mutation_pathway',verbose=False)
timer.stop()
