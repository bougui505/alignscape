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
timer.start('computing the tree')
mstree,mstree_pairs,paths = mspt.get_minsptree(localadj,paths,verbose=False)
timer.stop()
timer.start('computing the mstree graph')
mspt.write_mstree_gml(mstree,som.bmus,som.labels,(som.m,som.n),outname='mstree_ntw')
timer.stop()
timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = mspt.get_unfold_umat(som.umat, som.adj,bmus, mstree)
timer.stop()
exit()
unfbmus = [mapping[bmu] for bmu in bmus]

plt.matshow(uumat)
plt.colorbar()
plt.savefig('testunfold.pdf')
