import functools
import pickle
import matplotlib.pyplot as plt
import newick
import numpy as np
import networkx as nx
import quicksom.som
import quicksom.somax
import quicksom.utils
import som_seq
from som_seq import seqmetric
from utils.Timer import Timer
from utils import minsptree
import plot_umat
somfile = 'testout/som.pickle'
timer = Timer(autoreset=True)

timer.start('loading the som')
with open(somfile, 'rb') as somfileaux:
    som = pickle.load(somfileaux)
b62 = som_seq.get_blosum62()
som.metric = functools.partial(som_seq.seqmetric, b62=b62)
bmus = list(zip(*som.bmus[0:100].T))
timer.stop()

timer.start('computing localadj between queries')
localadj, paths = minsptree.get_localadjmat(som.umat,som.adj,bmus)
timer.stop()

timer.start('computing the minsptree tree')
mstree,mstree_pairs,mstree_paths = minsptree.get_minsptree(localadj,paths,verbose=False)
timer.stop()

timer.start('computing the mstree graph')
minsptree.write_mstree_gml(mstree,som.bmus,som.labels,(som.m,som.n),outname='testout/mstree_ntw')
timer.stop()

timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = minsptree.get_unfold_umat(som.umat, som.adj, bmus, mstree)
unfbmus = np.asarray([mapping[bmu] for bmu in bmus])
som.uumat = uumat
som.mapping = mapping
som.reversed_mapping = reversed_mapping
timer.stop()

timer.start('get the minsptree paths in the unfold umat')
mstree_pairs, mstree_paths = minsptree.get_unfold_msptree(mstree_pairs,mstree_paths, som.umat.shape, som.uumat.shape, som.mapping)
timer.stop()

timer.start('Test plots')
labels = []
plot_umat.main(somfile,outname='testout/umat',delimiter=None,hideSeqs=True,minsptree=False,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_remap',delimiter=None,hideSeqs=False,minsptree=False,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_labels',delimiter='_',hideSeqs=False,minsptree=False,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_minsptree',delimiter='_',hideSeqs=False,minsptree=True,clst=False,unfold=False)
plot_umat.main(somfile,outname='testout/umat_unfold',delimiter='_',hideSeqs=False,minsptree=False,clst=False,unfold=True)
plot_umat.main(somfile,outname='testout/umat_minsptree_unfold',delimiter='_',hideSeqs=False,minsptree=True,clst=False,unfold=True)
plot_umat.main(somfile,outname='testout/umat_clst',delimiter='_',hideSeqs=False,minsptree=False,clst=True,unfold=False)
plot_umat.main(somfile,outname='testout/umat_minsptree_clst',delimiter='_',hideSeqs=False,minsptree=True,clst=True,unfold=False)
plot_umat.main(somfile,outname='testout/umat_unfold_clst',delimiter='_',hideSeqs=False,minsptree=False,clst=True,unfold=True)
plot_umat.main(somfile,outname='testout/umat_minsptree_unfold_clst',delimiter='_',hideSeqs=False,minsptree=True,clst=True,unfold=True)
timer.stop()
