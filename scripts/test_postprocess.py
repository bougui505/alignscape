#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import functools
import pickle
import numpy as np
from alignscape import align_scape
from alignscape import plot_umat
from alignscape.align_scape import seqmetric
from alignscape.utils.Timer import Timer
from alignscape.utils import minsptree
from alignscape.analysis import dmatrix
from alignscape.analysis import cmatrix
from alignscape.analysis import mutation_pathway
from alignscape.analysis import classification

somfile = 'testout/som.pickle'
timer = Timer(autoreset=True)

timer.start('loading the som')
with open(somfile, 'rb') as somfileaux:
    somobj = pickle.load(somfileaux)
b62 = align_scape.get_blosum62()
somobj.metric = functools.partial(seqmetric, b62=b62)
bmus = list(zip(*somobj.bmus[0:100].T))
timer.stop()

timer.start('computing localadj between queries')
localadj, paths = minsptree.get_localadjmat(somobj.umat,
                                            somobj.adj, bmus)
timer.stop()

timer.start('computing the minsptree tree')
mstree, mstree_pairs, mstree_paths = \
    minsptree.get_minsptree(localadj, paths, verbose=False)
timer.stop()

timer.start('computing the mstree graph')
minsptree.write_mstree_gml(mstree, somobj.bmus,
                           somobj.labels, (somobj.m, somobj.n),
                           outname='testout/mstree_ntw')
timer.stop()

timer.start('computing the unfolding')
uumat, mapping, reversed_mapping = \
    minsptree.get_unfold_umat(somobj.umat, somobj.adj,
                              bmus, mstree)
unfbmus = np.asarray([mapping[bmu] for bmu in bmus])
somobj.uumat = uumat
somobj.mapping = mapping
somobj.reversed_mapping = reversed_mapping
timer.stop()

timer.start('get the minsptree paths in the unfold umat')
unf_mstree_pairs, unf_mstree_paths = \
    minsptree.get_unfold_mstree(mstree_pairs,
                                mstree_paths,
                                somobj.umat.shape,
                                somobj.uumat.shape,
                                somobj.mapping)
timer.stop()

timer.start('Test plots')
plot_umat.main(somfile, outname='testout/umat',
               delimiter=None, hideSeqs=True,
               mst=False, clst=False, unfold=False)
plot_umat.main(somfile,outname='testout/umat_remap',
               delimiter=None, hideSeqs=False,
               mst=False, clst=False, unfold=False)
plot_umat.main(somfile, outname='testout/umat_labels',
               delimiter='_', hideSeqs=False,
               mst=False, clst=False, unfold=False)
plot_umat.main(somfile, outname='testout/umat_minsptree',
               delimiter='_', hideSeqs=False, mst=True,
               clst=False, unfold=False)
plot_umat.main(somfile, outname='testout/umat_unfold',
               delimiter='_', hideSeqs=False, mst=False,
               clst=False, unfold=True)
plot_umat.main(somfile, outname='testout/umat_minsptree_unfold',
               delimiter='_', hideSeqs=False, mst=True,
               clst=False, unfold=True)
plot_umat.main(somfile, outname='testout/umat_clst',
               delimiter='_', hideSeqs=False, mst=False,
               clst=True, unfold=False)
plot_umat.main(somfile, outname='testout/umat_minsptree_clst',
               delimiter='_', hideSeqs=False, mst=True,
               clst=True, unfold=False)
plot_umat.main(somfile, outname='testout/umat_unfold_clst',
               delimiter='_', hideSeqs=False, mst=False,
               clst=True, unfold=True)
plot_umat.main(somfile,
               outname='testout/umat_minsptree_unfold_clst',
               delimiter='_', hideSeqs=False, mst=True,
               clst=True, unfold=True)
timer.stop()

timer.start('Test k-neighbours')
classification.main(somfile=somfile,
                    outname='testout/classification',
                    delimiter='_', uclass='unk', k=1)
timer.stop()

timer.start('Test distance matrix')
dmobj = dmatrix.Dmatrix(somfile=somfile,
                        querieslist=somobj.labels,
                        output='testout/dmatrix', delimiter='_')
timer.stop()

timer.start('Test correlation matrix')
cmatrix.Cmatrix(dmatrices=['testout/dmatrix.p',
                           'testout/dmatrix.p'],
                labels=['dm', 'dm_'], outname='testout/cmatrix')
timer.stop()

timer.start('Test mutation pathway')
mutation_pathway.main(unit1=somobj.bmus[0],
                      unit2=somobj.bmus[1],
                      somfile=somfile,
                      outname='testout/mutation_pathway',
                      verbose=False)
timer.stop()
