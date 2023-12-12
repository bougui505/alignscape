import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import functools
from alignscape.align_scape import seqmetric
from alignscape.align_scape import get_blosum62
from alignscape.utils import minsptree
from alignscape.utils import jax_imports
from alignscape.utils.Timer import Timer

timer = Timer(autoreset=True)


def main(somfile, outname='umat', delimiter=None, hideSeqs=False,
         mst=False, clst=False, unfold=False, plot_ext='png', max_ppmd=None):

    # Load the data and parse it
    with open(somfile, 'rb') as somfileaux:
        somobj = pickle.load(somfileaux)

    b62 = get_blosum62()
    if somobj.jax:
        somobj.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)
    else:
        somobj.metric = functools.partial(seqmetric, b62=b62)
    bmus = list(zip(*somobj.bmus.T))
    titles = somobj.labels
    titles = [title.replace(">", "") for title in titles]
    if delimiter is not None:
        labels = [title.split(delimiter)[0] for title in titles]
    else:
        labels = []

    if mst or unfold:
        # Compute the local Adjacency Matrix between the qbmus
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
        # Compute the minimal spanning tree
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
        if mst:
            minsptree.write_mstree_gml(mstree,
                                       bmus,
                                       titles,
                                       somobj.umat.shape,
                                       outname='%s' % outname)

    if unfold:
        # Use the minimial spanning three between queries bmus to \
        # unfold the umat
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
        if mst:
            timer.start('get the minsptree paths in the unfold umat')
            mstree_pairs, mstree_paths = \
                minsptree.get_unfold_mstree(mstree_pairs,
                                            mstree_paths,
                                            somobj.umat.shape,
                                            somobj.uumat.shape,
                                            mapping)
            timer.stop()
        timer.stop()
    else:
        auxbmus = bmus
        auxumat = somobj.umat

    if clst:
        timer.start('clusterizing the umat')
        umat_clst = minsptree.get_clusterized_umat(somobj.umat,
                                                   somobj.adj,
                                                   somobj.umat.shape)
        if unfold:
            for unf_bmu in reversed_mapping.keys():
                fold_bmu = reversed_mapping[unf_bmu]
                fold_value = umat_clst[fold_bmu[0], fold_bmu[1]]
                auxumat[unf_bmu[0]][unf_bmu[1]] = fold_value
        else:
            auxumat = umat_clst
        f = open('%s.txt' % outname, 'w')
        f.write('#bmu_r #bmu_c #title #cluster\n')
        for i, bmu in enumerate(auxbmus):
            bmu_r = auxbmus[i][0]
            bmu_c = auxbmus[i][1]
            f.write('%d %d %s %d\n' % (bmu_r, bmu_c, titles[i],
                                       auxumat[bmu_r][bmu_c]))
        f.close()
        timer.stop()

    # Saving the SOM
    somobj.save_pickle(somfile)

    # Plotting
    _plot_umat(umat=auxumat, bmus=auxbmus, labels=labels, hideSeqs=hideSeqs,
               max_ppmd=max_ppmd)

    if mst and not unfold:
        _plot_mstree(mstree_pairs, mstree_paths, somobj.umat.shape)
    elif mst and unfold:
        _plot_mstree(mstree_pairs, mstree_paths, somobj.uumat.shape)

    plt.savefig(outname+'.'+plot_ext)
    plt.show()


def _plot_mstree(mstree_pairs, mstree_paths, somsize, verbose=False):
    n1, n2 = somsize
    for i, mstree_pair in enumerate(mstree_pairs):
        if verbose:
            print('Printing the shortest parth between %s and %s'
                  % (mstree_pair[0], mstree_pair[1]))
        mstree_path = mstree_paths[tuple(mstree_pair)]
        _mstree_path = np.asarray(np.unravel_index(mstree_path, (n1, n2)))
        _mstree_path = np.vstack((_mstree_path[0], _mstree_path[1])).T
        for j, step in enumerate(_mstree_path):
            if j == 0:
                continue
            # Check to avoid borders printting horizontal or vertical lines
            if (_mstree_path[j-1][0] == 0 and _mstree_path[j][0] == n1-1) or \
                    (_mstree_path[j-1][0] == n1-1 and _mstree_path[j][0] == 0)\
                    or (_mstree_path[j-1][1] == 0 and _mstree_path[j][1] == n2-1)\
                    or (_mstree_path[j-1][1] == n2-1 and _mstree_path[j][1] == 0):
                continue
            aux = np.stack((_mstree_path[j-1], _mstree_path[j])).T
            plt.plot(aux[1], aux[0], c='w', linewidth=0.8)


def _plot_umat(umat, bmus, labels, hideSeqs, dotsize=25,
               legend=True, dic_colors=None, cmap=None, max_ppmd=None):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    if cmap:
        if max_ppmd is not None:
            cax = ax.matshow(umat, cmap=cmap, vmax=max_ppmd)
        else:
            cax = ax.matshow(umat, cmap=cmap)
    else:
        if max_ppmd is not None:
            cax = ax.matshow(umat, vmax=max_ppmd)
        else:
            cax = ax.matshow(umat)
    figure.colorbar(cax)

    if not hideSeqs:
        if len(labels) == 0:
            for bmu in bmus:
                minsptree.highlight_cell(int(bmu[1]), int(bmu[0]),
                                         color="grey", linewidth=0.5)
        else:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            unique_labels = list(set(labels))
            # If unknown just highlight the cell
            if 'unk' in unique_labels:
                for i, label in enumerate(labels):
                    if label == 'unk':
                        minsptree.highlight_cell(int(bmus[i][1]),
                                                 int(bmus[i][0]),
                                                 color="grey",
                                                 linewidth=0.5)
            for unique_label in unique_labels:
                if unique_label == 'unk':
                    continue
                if dic_colors:
                    try:
                        color = dic_colors[unique_label]
                    except:
                        raise ValueError('%s label is missing in dic_colors' %
                                         unique_label)
                aux_X = []
                aux_Y = []
                for i, label in enumerate(labels):
                    if label == unique_label:
                        aux_X.append(bmus[i][1])
                        aux_Y.append(bmus[i][0])
                    else:
                        continue
                if dic_colors:
                    ax.scatter(aux_X, aux_Y, label=unique_label,
                               s=dotsize, color=color)
                else:
                    ax.scatter(aux_X, aux_Y, label=unique_label, s=dotsize)
            if legend:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                              fancybox=True, shadow=True, ncol=5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    requiredArguments = parser.add_argument_group('required arguments')
    requiredArguments.add_argument('-s', '--somfile', help='Som file',
                                   required=True)
    parser.add_argument('-o', '--outname', help='Output name for the umat',
                        default='umat')
    parser.add_argument('-d', '--delimiter', help='If gruping infomation was \
                        contained in the sequences title, the delimiter split \
                        it and select the prefix', default=None)
    parser.add_argument('--hide_seqs', help='To hide input sequences',
                        action='store_true')
    parser.add_argument('--mst', help='Plot the minimal spanning tree between \
                        BMUs', default=False, action='store_true')
    parser.add_argument('--clst', help='Clusterize the Umat', default=False,
                        action='store_true')
    parser.add_argument('--unfold', help='Unfold the Umat', default=False,
                        action='store_true')
    parser.add_argument('--plot_ext', help='Filetype extension for the UMAT \
                        plots (default: png)', default='png')
    parser.add_argument('--max_ppmd', help='Maximum PPMd value for which to color \
                        the Umat', default=None)
    args = parser.parse_args()

    main(somfile=args.somfile,
         outname=args.outname,
         delimiter=args.delimiter,
         hideSeqs=args.hide_seqs,
         mst=args.mst,
         clst=args.clst,
         unfold=args.unfold,
         plot_ext=args.plot_ext,
         max_ppmd=float(args.max_ppmd))
