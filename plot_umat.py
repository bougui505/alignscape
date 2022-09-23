import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import pickle
import functools
import som_seq
import jax_imports
import minsptree as msptree
import quicksom.som
import quicksom.somax


def main(somfile,bmusfile,outname='umat',delimiter=None,hideSeqs=False):
    
    #Load the data
    allbmus = np.genfromtxt(bmusfile, dtype=str, skip_header=1)
    with open(somfile, 'rb') as somfileaux:
        som = pickle.load(somfileaux)
    b62 = som_seq.get_blosum62()
    som.metric = functools.partial(jax_imports.seqmetric_jax, b62=b62)

    #Parse the data
    titles = list()
    bmus = list()
    labels = list()
    for k,bmu in enumerate(allbmus):
        bmus.append((int(bmu[0]),int(bmu[1])))
        title = bmu[-1].replace(">","")
        titles.append(title)
        if delimiter != None:
            labels.append(title.split(delimiter)[0])

    #print(bmus)
    #print(titles)
    #print(labels)

    auxbmus = bmus
    auxumat = som.umat
    auxadj = som.adj

    n1, n2 = auxumat.shape

    _plot_umat(auxumat,auxbmus,labels,outname,hideSeqs)


def _plot_umat(umat, bmus, labels, outname, hideSeqs):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.matshow(umat)
    figure.colorbar(cax)

    if not hideSeqs:
        if len(labels) == 0:
            for bmu in bmus:
                msptree.highlight_cell(int(bmu[1]),int(bmu[0]), color="grey", linewidth=0.5)
        else:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])
            for unique_label in list(set(labels)):
                aux_X = []
                aux_Y = []
                for i,label in enumerate(labels):
                    if label == unique_label:
                        aux_X.append(bmus[i][1])
                        aux_Y.append(bmus[i][0])
                    else: continue
                ax.scatter(aux_X,aux_Y,label=unique_label,s=15)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                              fancybox=True, shadow=True, ncol=5)

    plt.savefig(outname+'.pdf')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    requiredArguments = parser.add_argument_group('required arguments')
    requiredArguments.add_argument('-s', '--som', help = 'Som file', required = True)
    requiredArguments.add_argument('-b', '--bmus', help = 'BMUS of all sequences inputted for the Som', required = True)
    parser.add_argument('-o', '--outname', help = 'Output name for the umat',default='umat')
    parser.add_argument('-d', '--delimiter', help = 'If gruping infomation was contained in the sequences title, the delimiter split it and select the prefix',default=None)
    parser.add_argument('--hide_seqs',help = 'To hide input sequences',action='store_true')

    args = parser.parse_args()

    main(somfile=args.som,bmusfile=args.bmus,outname=args.outname,delimiter=args.delimiter,hideSeqs=args.hide_seqs)
