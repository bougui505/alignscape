import torch
import pickle
import functools
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from alignscape.utils.Timer import Timer
from alignscape.align_scape import get_blosum62, seqmetric
from alignscape.utils import jax_imports
from alignscape.utils import seqdataloader
from alignscape.utils import minsptree

aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ|-')
timer = Timer(autoreset=True)


def main(unit1, unit2, somfile, outname, threshold=0.2, allinp=False,
         unfold=False, verbose=True, plot_ext='png'):

    # Load and safecheck the data
    if len(unit1) != 2:
        raise ValueError('c1 must provide a two elements list with the row \
                         and column of the first cell')
    unit1 = tuple(unit1)
    if len(unit2) != 2:
        raise ValueError('c1 must provide a two elements list with the row \
                         and column of the second cell')
    unit2 = tuple(unit2)
    with open(somfile, 'rb') as somfileaux:
        somobj = pickle.load(somfileaux)
    b62 = get_blosum62()
    if somobj.jax:
        somobj.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
    else:
        somobj.metric = functools.partial(seqmetric, b62=b62)
    threshold = float(threshold)

    # Parse the data
    subtypes = list()
    bmus = [tuple(bmu) for bmu in somobj.bmus]
    labels = ['_'.join(label.split("_")[1:]) for label in somobj.labels]
    subtypes = [label.replace(">", "").split("_")[0] for label
                in somobj.labels]

    # Get the shortest path between unit1 and unit2
    n1, n2 = somobj.umat.shape
    indx_unit1 = np.ravel_multi_index(unit1, (n1, n2))
    indx_unit2 = np.ravel_multi_index(unit2, (n1, n2))

    if verbose:
        print('Computing shortest path between: ' + str(unit1) + ' '
              + str(unit2))
    path = minsptree.get_shortestPath(somobj.adj, indx_unit1, indx_unit2)
    dok = somobj.adj.todok()
    pathDist = minsptree.get_pathDist(dok, path)
    if verbose:
        print(path)
        print(pathDist)
    unrpath = np.asarray(np.unravel_index(path, (n1, n2)))
    unrpath = np.vstack((unrpath[0], unrpath[1])).T
    if verbose:
        print(unrpath)

    # Get the consensus sequences of the cells of the spath between
    # unit1 and unit2
    fout = open(outname+'.fasta', 'w')
    if isinstance(somobj.centroids, torch.Tensor):
        centroids = somobj.centroids.detach().cpu().numpy()
        smap = centroids.reshape((somobj.m, somobj.n, -1))
    else:
        smap = np.asarray(somobj.centroids).reshape((somobj.m, somobj.n, -1))
    for cell in unrpath:
        cell = tuple(cell)
        neuron = smap[cell]
        cseq = seqdataloader.vec2seq(neuron, threshold)
        if cell in bmus:
            indx = bmus.index(cell)
            title = ">[" + str(tuple(np.asarray(cell).T)[0]) + "|" + \
                str(tuple(np.asarray(cell).T)[1]) + "] " + subtypes[indx] \
                + "_" + labels[indx]
        else:
            title = ">[" + str(cell[0]) + "|" + str(cell[1]) + "]"
        if verbose:
            print(title)
        fout.write(title+"\n")
        if verbose:
            print(cseq)
        fout.write(cseq+"\n")
    fout.close()

    if unfold:
        if not hasattr(somobj, 'localadj'):
            timer.start('computing localadj between queries')
            localadj, localadj_paths = minsptree.get_localadjmat(somobj.umat,
                                                                 somobj.adj,
                                                                 bmus,
                                                                 verbose=True)
            timer.stop()
            somobj.localadj = localadj
            somobj.localadj_paths = localadj_paths
        else:
            localadj = somobj.localadj
            localadj_paths = somobj.localadj_paths
        if not hasattr(somobj, 'mstree'):
            timer.start('compute the mstree')
            mstree, mstree_pairs, mstree_paths = \
                minsptree.get_minsptree(localadj, localadj_paths)
            timer.stop()
            somobj.mstree = mstree
            somobj.mstree_pairs = mstree_pairs
            somobj.mstree_paths = mstree_paths
        else:
            mstree = somobj.mstree
            mstree_pairs = somobj.mstree_pairs
            mstree_paths = somobj.mstree_paths
        timer.start('compute the umap unfolding')
        uumat, mapping, reversed_mapping = \
            minsptree.get_unfold_umat(somobj.umat, somobj.adj, bmus, mstree)
        timer.stop()
        somobj.uumat = uumat
        somobj.mapping = mapping
        somobj.reversed_mapping = reversed_mapping
        auxumat = uumat
        unfbmus = [mapping[bmu] for bmu in bmus]
        auxbmus = unfbmus
        auxpath = np.asarray([np.asarray(mapping[tuple(path)])
                              for path in unrpath])
    else:
        auxbmus = bmus
        auxumat = somobj.umat
        auxpath = unrpath

    # Print the shortest path between unit1 and unit2
    plt.matshow(auxumat)
    plt.colorbar()
    for j, step in enumerate(auxpath):
        if j == 0:
            continue
        # Check to avoid borders printting horizontal or vertical lines
        if (auxpath[j-1][0] == 0 and auxpath[j][0] == n1-1) or\
                (auxpath[j-1][0] == n1-1 and auxpath[j][0] == 0) or\
                (auxpath[j-1][1] == 0 and auxpath[j][1] == n2-1) or\
                (auxpath[j-1][1] == n2-1 and auxpath[j][1] == 0):
            continue
        aux = np.stack((auxpath[j-1], auxpath[j])).T
        plt.plot(aux[1], aux[0], c='w', linewidth=1)

    # Highlight the initial data if the shortest path pass through one of them

    texts = []
    for step in auxpath:
        step = tuple(step)
        if step in auxbmus:
            indx = auxbmus.index(step)
            if subtypes[indx] == 'CMGC':
                plt.scatter(step[1], step[0], c='black', s=15)
            if subtypes[indx] == 'CAMK':
                plt.scatter(step[1], step[0], c='white', s=15)
            if subtypes[indx] == 'TKL':
                plt.scatter(step[1], step[0], c='red', s=15)
            if subtypes[indx] == 'AGC':
                plt.scatter(step[1], step[0], c='orange', s=15)
            if subtypes[indx] == 'RGC':
                plt.scatter(step[1], step[0], c='pink', s=15)
            if subtypes[indx] == 'OTHER':
                plt.scatter(step[1], step[0], c='yellow', s=15)
            if subtypes[indx] == 'CK1':
                plt.scatter(step[1], step[0], c='lime', s=15)
            if subtypes[indx] == 'STE':
                plt.scatter(step[1], step[0], c='cyan', s=15)
            if subtypes[indx] == 'NEK':
                plt.scatter(step[1], step[0], c='magenta', s=15)
            if subtypes[indx] == 'TYR':
                plt.scatter(step[1], step[0], c='blue', s=15)
            aux = subtypes[indx] + '_' + labels[indx]
            texts.append(plt.text(step[1], step[0], aux,
                                  fontsize=7, c='gainsboro'))
    if len(texts) > 0:
        adjust_text(texts, only_move={'points': 'y', 'texts': 'y'},
                    arrowprops=dict(arrowstyle="->, head_width=0.2",
                                    color='gainsboro', lw=0.5))

    if allinp:
        # Hihlight the rest of initial sequences
        for k, bmu in enumerate(auxbmus):
            if subtypes[k] == 'CMGC':
                plt.scatter(bmu[1], bmu[0], c='black', s=2)
            if subtypes[k] == 'CAMK':
                plt.scatter(bmu[1], bmu[0], c='white', s=2)
            if subtypes[k] == 'TKL':
                plt.scatter(bmu[1], bmu[0], c='red', s=2)
            if subtypes[k] == 'AGC':
                plt.scatter(bmu[1], bmu[0], c='orange', s=2)
            if subtypes[k] == 'RGC':
                plt.scatter(bmu[1], bmu[0], c='pink', s=2)
            if subtypes[k] == 'OTHER':
                plt.scatter(bmu[1], bmu[0], c='yellow', s=2)
            if subtypes[k] == 'CK1':
                plt.scatter(bmu[1], bmu[0], c='lime', s=2)
            if subtypes[k] == 'STE':
                plt.scatter(bmu[1], bmu[0], c='cyan', s=2)
            if subtypes[k] == 'NEK':
                plt.scatter(bmu[1], bmu[0], c='magenta', s=2)
            if subtypes[k] == 'TYR':
                plt.scatter(bmu[1], bmu[0], c='blue', s=2)


    plt.savefig(outname + '.' + plot_ext, dpi=500)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--c1', nargs='+',
                        help='First umat cell coordinates (row-col format)',
                        required=True, type=int)
    parser.add_argument('--c2', nargs='+',
                        help='First umat cell coordinates (row-col format)',
                        required=True, type=int)
    parser.add_argument('-s', '--somfile', help='Som file',
                        required=True)
    parser.add_argument('-o', '--outname', help='Fasta outname',
                        required=True)
    parser.add_argument('--freq_thres', help='Frequency threshold to assign the\
                        most frequent residue for each site', default=0.2)
    parser.add_argument('--allinp', help='plot all input data', default=False,
                        action='store_true')
    parser.add_argument('--unfold', help='Unfold the Umat', default=False,
                        action='store_true')
    parser.add_argument('--plot_ext', help='Filetype extension for'
                        ' the UMAT plots (default: png)', default='png')
    args = parser.parse_args()

    main(unit1=args.c1, unit2=args.c2, somfile=args.somfile,
         threshold=args.freq_thres, outname=args.outname, allinp=args.allinp,
         unfold=args.unfold, plot_ext=args.plot_ext)
