import functools
import pickle
import matplotlib.pyplot as plt
import newick
import numpy as np
import networkx as nx
from quicksom import somax
from quicksom import som
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
    somobj = pickle.load(somfileaux)
b62 = som_seq.get_blosum62()
if somobj.jax:
    somobj.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
else:
    somobj.metric = functools.partial(seqmetric, b62=b62)
bmus = list(zip(*somobj.bmus[0:100].T))
timer.stop()

timer.start('computing localadj between queries')
localadj, paths = minsptree.get_localadjmat(somobj.umat,somobj.adj,bmus)
timer.stop()

timer.start('computing the minsptree tree')
mstree,mstree_pairs,mstree_paths = minsptree.get_minsptree(localadj,paths,verbose=False)
timer.stop()

timer.start('computing the mstree graph')
minsptree.write_mstree_gml(mstree,somobj.bmus,somobj.labels,(somobj.m,somobj.n),outname='testout/mstree_ntw')
timer.stop()

timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = minsptree.get_unfold_umat(somobj.umat, somobj.adj, bmus, mstree)
unfbmus = np.asarray([mapping[bmu] for bmu in bmus])
somobj.uumat = uumat
somobj.mapping = mapping
somobj.reversed_mapping = reversed_mapping
timer.stop()

timer.start('get the minsptree paths in the unfold umat')
mstree_pairs, mstree_paths = minsptree.get_unfold_msptree(mstree_pairs,mstree_paths, somobj.umat.shape, somobj.uumat.shape, somobj.mapping)
timer.stop()

timer.start('Test plots')
labels = []
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
timer.stop()
