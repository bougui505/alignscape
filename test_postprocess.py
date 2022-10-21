import quicksom.som
import quicksom.somax
import quicksom.utils
import functools
import som_seq
from som_seq import seqmetric
import pickle
from Timer import Timer
import matplotlib.pyplot as plt
import newick
import numpy as np
import networkx as nx
import minsptree as mspt
import plot_umat
somfile = 'som.pickle'
timer = Timer(autoreset=True)

with open(somfile, 'rb') as somfileaux:
    som = pickle.load(somfileaux)
b62 = som_seq.get_blosum62()
som.metric = functools.partial(som_seq.seqmetric, b62=b62)
bmus = list(zip(*som.bmus[0:100].T))

timer.start('computing localadj between queries')
localadj, paths = mspt.get_localadjmat(som.umat,som.adj,bmus)
timer.stop()

timer.start('computing the minsptree tree')
mstree,mstree_pairs,mstree_paths = mspt.get_minsptree(localadj,paths,verbose=False)
timer.stop()

timer.start('Plotting1')
labels = []
plot_umat._plot_umat(som.umat,som.bmus,labels,False)
plot_umat._plot_msptree(mstree_pairs,mstree_paths,(som.m,som.n))
plt.savefig('umat_msptree.pdf')
timer.stop()

timer.start('computing the mstree graph')
mspt.write_mstree_gml(mstree,som.bmus,som.labels,(som.m,som.n),outname='mstree_ntw')
timer.stop()

timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = mspt.get_unfold_umat(som.umat, som.adj, bmus, mstree)
unfbmus = np.asarray([mapping[bmu] for bmu in bmus])
som.uumat = uumat
som.mapping = mapping
som.reversed_mapping = reversed_mapping
som._get_unfold_adj()
timer.stop()

timer.start('get the minsptree paths in the unfold umat')
mstree, mstree_pairs, mstree_paths = mspt.get_unfold_msptree(mstree_pairs,mstree_paths, som.umat.shape, som.uumat.shape, som.mapping, som.uadj)
timer.stop()

timer.start('computing the unfold mstree graph')
mspt.write_mstree_gml(mstree,unfbmus,som.labels,uumat.shape, outname='unf_mstree_ntw')
timer.stop()

timer.start('Plotting2')
labels = []
plot_umat._plot_umat(uumat,unfbmus,labels,False)
plot_umat._plot_msptree(mstree_pairs,mstree_paths,uumat.shape)
plt.savefig('unfold_umat_msptree.pdf')
timer.stop()
