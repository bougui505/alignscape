import minsptree as msptree
import quicksom.som
import quicksom.somax
import quicksom.utils
import functools
import som_seq
from som_seq import seqmetric
import pickle
from Timer import Timer
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
import newick

somfile = 'som.pickle'
timer = Timer(autoreset=True)

with open(somfile, 'rb') as somfileaux:
    som = pickle.load(somfileaux)
b62 = som_seq.get_blosum62()
som.metric = functools.partial(som_seq.seqmetric, b62=b62)
bmus = list(zip(*som.bmus[0:100].T))
timer.start('computing localadj between queries')
localadj, paths = msptree.get_localadjmat(som.umat,som.adj,bmus)
timer.stop()
timer.start('computing the tree')
mstree = csgraph.minimum_spanning_tree(localadj)
mstree = mstree.tocoo()
mstree_rows = mstree.row
mstree_cols = mstree.col
mstree_dists = mstree.data
mstree_rows = quicksom.utils.bmus_to_label(mstree_rows,som.bmus,som.labels,(som.m,som.n))
mstree_cols = quicksom.utils.bmus_to_label(mstree_cols,som.bmus,som.labels,(som.m,som.n))
timer.stop()
timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = msptree.get_unfold_umat(som.umat, som.adj,bmus, mstree)
timer.stop()

unfbmus = [mapping[bmu] for bmu in bmus]

plt.matshow(uumat)
plt.colorbar()
plt.savefig('testunfold.pdf')
