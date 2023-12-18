import functools
import numpy as np
import scipy.sparse
import dill as pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import itertools
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import Phylo
from alignscape.align_scape import seqmetric, get_blosum62
from alignscape.utils import jax_imports


def get_neighbors(i, j, X, Y, ravel=True):
    neighbors = []
    if ravel:
        iRef = np.ravel_multi_index((i % X, j % Y), (X, Y))
        jRef1 = np.ravel_multi_index(((i - 1) % X, (j - 1) % Y), (X, Y))
        jRef2 = np.ravel_multi_index(((i - 1) % X, (j) % Y), (X, Y))
        jRef3 = np.ravel_multi_index(((i - 1) % X, (j + 1) % Y), (X, Y))
        jRef4 = np.ravel_multi_index(((i) % X, (j - 1) % Y), (X, Y))
        jRef5 = np.ravel_multi_index(((i) % X, (j + 1) % Y), (X, Y))
        jRef6 = np.ravel_multi_index(((i + 1) % X, (j - 1) % Y), (X, Y))
        jRef7 = np.ravel_multi_index(((i + 1) % X, (j) % Y), (X, Y))
        jRef8 = np.ravel_multi_index(((i + 1) % X, (j + 1) % Y), (X, Y))
        neighbors = [iRef, jRef1, jRef2, jRef3, jRef4,
                     jRef5, jRef6, jRef7, jRef8]
        return neighbors
    else:
        iRef = (i % X, j % Y)
        jRef1 = ((i - 1) % X, (j - 1) % Y)
        jRef2 = ((i - 1) % X, (j) % Y)
        jRef3 = ((i - 1) % X, (j + 1) % Y)
        jRef4 = ((i) % X, (j - 1) % Y)
        jRef5 = ((i) % X, (j + 1) % Y)
        jRef6 = ((i + 1) % X, (j - 1) % Y)
        jRef7 = ((i + 1) % X, (j) % Y)
        jRef8 = ((i + 1) % X, (j + 1) % Y)
        neighbors = [iRef, jRef1, jRef2, jRef3, jRef4, jRef5,
                     jRef6, jRef7, jRef8]
        return neighbors


def get_phyloDmatrix(nwkphylo):
    """
    """
    nwkphylo = nwkphylo
    tree = Phylo.read(nwkphylo, 'newick')
    keys = []
    for x in tree.get_terminals():
        keys.append(x.name)
    d = {key: {key: 0 for key in keys} for key in keys}
    for x, y in itertools.combinations(tree.get_terminals(), 2):
        v = tree.distance(x, y)
        xname = x.name
        yname = y.name
        d[xname][yname] = v
        d[yname][xname] = v
        d[xname][xname] = 0
        d[yname][yname] = 0
    df = pd.DataFrame(d)
    return df


class Dmatrix(object):
    """
    Distance matrix
    """
    def __init__(self, somfile=None, queries=None, querieslist=None,
                 output=None, load=None,  delimiter=None, plot_ext='png'):
        """
        """
        # Parse initial data
        self.out = output
        self.delimiter = delimiter
        self.plot_ext = plot_ext

        # Load an already calculated Dmatrix object
        if load is not None and all(x is None for x in [somfile,
                                                        queries, querieslist]):
            self.load = load
            self.df = pickle.load(open(load, 'rb'))
            self.columns = self.df.columns
            self.queries = self.df.columns.tolist()
        # Calculate a Dmatrix using a file/list of queries and a somfile
        elif somfile is not None and (queries is not None or querieslist
                                      is not None) and load is None:
            with open(somfile, 'rb') as somaux:
                self.somobj = pickle.load(somaux)
            b62 = get_blosum62()
            if self.somobj.jax:
                self.somobj.metric = functools.partial(jax_imports.seqmetric_jax,
                                                       b62=b62)
            else:
                self.somobj.metric = functools.partial(seqmetric, b62=b62)
            allbmus = self.somobj.bmus
            allbmus = list(zip(*allbmus.T))
            titles = self.somobj.labels
            titles = [title.replace(">", "") for title in titles]
            self.bmus = list()
            self.indxs = list()
            if queries is not None and querieslist is not None:
                raise ValueError('just queries or querieslist not both')
            elif queries is not None:
                f = open(queries, 'r')
                self.queries = [query.replace("\n", "").replace('>', '')
                                for query in f]
            elif querieslist is not None:
                self.queries = [query.replace('>', '') for query
                                in querieslist]
            for query in self.queries:
                indx = list(np.where(np.asarray(titles) == query)[0])
                if len(indx) == 0:
                    warnings.warn('The query: %s is not find in the somobj'
                                  % query)
                    continue
                elif len(indx) > 1:
                    warnings.warn('The query: %s is find multiple times in the \
                                  somobj' % query)
                indx = indx[0]
                self.bmus.append((int(allbmus[indx][0]),
                                  int(allbmus[indx][1])))
                self.indxs.append(indx)
            self.umat = self.somobj.umat
            self.dim = self.somobj.umat.shape[0]
            self.get_SOMDmatrix()
            if self.out is not None:
                self.plot(out=self.out)
                self.save(out=self.out)
        else:
            raise ValueError("To initialize a Dmatrix either use 'load' \
                             or 'som+bmus+queries'")

    def get_SOMDmatrix(self):
        """
        """
        self.columns = self.queries
        # Calculate the distance matrix between all conected cells of the umat
        row_list = []
        col_list = []
        data_list = []
        for i in range(self.dim):
            row_list.append(np.ravel_multi_index((i, i), (self.dim, self.dim)))
            col_list.append(np.ravel_multi_index((i, i), (self.dim, self.dim)))
            data_list.append(0.0)
            for j in range(self.dim):
                c_uval = self.umat[i, j]
                ngs = get_neighbors(i, j, self.dim, self.dim, ravel=False)[1:]
                for ng in ngs:
                    row_list.append(np.ravel_multi_index(ng,
                                                         (self.dim, self.dim)))
                    col_list.append(np.ravel_multi_index((i, j),
                                                         (self.dim, self.dim)))
                    n_uval = self.umat[ng]
                    data_list.append(np.linalg.norm(c_uval-n_uval))
        adjmat = scipy.sparse.coo_matrix((data_list, (row_list, col_list)),
                                         shape=(self.dim*self.dim,
                                                self.dim*self.dim))
        c_distmat = scipy.sparse.csgraph.shortest_path(adjmat, directed=False)

        # Get the dataframe where the queries are the columns/rows and the
        # values the shortest path between its corresponding umat cells
        data = np.zeros((len(self.columns), len(self.columns)))
        for i, b1 in enumerate(self.bmus):
            for j, b2 in enumerate(self.bmus):
                r_b1 = np.ravel_multi_index(b1, (self.dim, self.dim))
                r_b2 = np.ravel_multi_index(b2, (self.dim, self.dim))
                data[i, j] = c_distmat[r_b1, r_b2]
        self.df = pd.DataFrame(data, columns=self.columns, index=self.columns)

    def save(self, out=None):
        if out is None and self.out is not None:
            self.df.to_pickle(self.out+'.p')
        elif out is not None:
            self.df.to_pickle(out+'.p')
        else:
            self.df.to_pickle('dmatrix.p')

    def plot(self, out=None):
        # Color rows and columns by subtype (delimiter)
        if len(self.columns) == 0:
            warnings.warn('There are no queries', UserWarning)
            return 0
        elif len(self.columns) == 1:
            cg = sns.heatmap(self.df, cmap="RdBu_r", linewidths=0.30,
                             yticklabels=True, xticklabels=True)
        elif len(self.columns) > 1:
            if self.delimiter is not None:
                labels = [query.split(self.delimiter)[0] for
                          query in self.queries]
                uniq_labels = list(set(labels))
                uniq_labels.sort()
                labels_pal = sns.color_palette("Set1",
                                               n_colors=len(uniq_labels),
                                               desat=.99)
                labels_lut = dict(list(zip(list(map(str, uniq_labels)),
                                           labels_pal)))
                colors = []
                for i, name in enumerate(zip(self.columns, self.columns)):
                    subtype = name[0].split(self.delimiter)[0]
                    colors.append(labels_lut[subtype])
                dfcolors = pd.DataFrame({'subtype': colors},
                                        index=self.columns)
                cg = sns.clustermap(self.df, cmap="RdBu_r", linewidths=0.30,
                                    metric='cityblock', col_colors=dfcolors,
                                    row_colors=dfcolors, yticklabels=True,
                                    xticklabels=True)
                # Add subtype legend
                for label in uniq_labels:
                    cg.ax_col_dendrogram.bar(0, 0, color=labels_lut[label],
                                             label=label, linewidth=0)
                    cg.ax_col_dendrogram.legend(loc="best",
                                                bbox_to_anchor=(0, 1.2),
                                                ncol=1)
            else:
                cg = sns.clustermap(self.df, cmap="RdBu_r", linewidths=0.30,
                                    metric='cityblock', yticklabels=True,
                                    xticklabels=True)

        # Save the plot
        if out is None and self.out is not None:
            plt.savefig("%s.%s" % (self.out, self.plot_ext), dpi=300)
        elif out is not None:
            plt.savefig("%s.%s" % (out, self.plot_ext), dpi=300)
        else:
            plt.savefig("dmatrix.%s" % self.plot_ext, dpi=300)

    def update(self, dmatrix2):
        """
        """
        dict1 = self.df.to_dict()
        dict2 = dmatrix2.df.to_dict()
        for ki in dict2.keys():
            for kj in dict2[ki].keys():
                if ki in dict1.keys() and kj in dict1.keys():
                    dict1[ki][kj] = dict2[ki][kj]
        self.df = pd.DataFrame(dict1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--somfile', help='Som file',
                        required=True)
    parser.add_argument('-q', '--queries', help='Sequences to be remmaped',
                        required=True)
    parser.add_argument('-o', '--out', help='Output name for the dmatrix plot \
                        and pickle file', default='dmatrix')
    parser.add_argument('--deli', help='Delimiter to trim the queries tittles',
                        default=None, type=str)
    parser.add_argument('--plot_ext', help='Filetype extension for'
                        ' the UMAT plots (default: png)', default='png')
    args = parser.parse_args()

    dmatrix = Dmatrix(somfile=args.somfile,
                      queries=args.queries,
                      output=args.out,
                      delimiter=args.deli,
                      plot_ext=args.plot_ext)
    dmatrix.plot()
    dmatrix.save()
