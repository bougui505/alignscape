import minsptree as msptree
import quicksom.som
import quicksom.somax
import functools
import som_seq
import pickle
import jax_imports
from Timer import Timer
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
somfile = 'results/Kinome/90x90_200e_noPLK5/kinome.p' 

timer = Timer(autoreset=True)

with open(somfile, 'rb') as somfileaux:
    som = pickle.load(somfileaux)
b62 = som_seq.get_blosum62()
som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)

bmus = list(zip(*som.bmus[0:100].T))
timer.start('computing localadj between queries')
localadj, paths = msptree.get_localadjmat(som.umat,som.adj,bmus)
timer.stop()
timer.start('computing the tree')
mstree = csgraph.minimum_spanning_tree(localadj)
print(mstree)
timer.stop()
timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = msptree.get_unfold_umat(som.umat, som.adj,bmus, mstree)
timer.stop()

unfbmus = [mapping[bmu] for bmu in bmus]

plt.matshow(uumat)
plt.colorbar()
plt.savefig('testunfold.pdf')
