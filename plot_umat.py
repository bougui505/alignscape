import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import pickle
import functools
import scipy.sparse
import som_seq
import jax_imports
import minsptree as msptree
import quicksom.som
import quicksom.somax


def main(somfile,bmusfile,outname='umat',delimiter=None,hideSeqs=False,minsptree=False,save=None,load=None):
    
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

    #Load or compute the local Adjacency Matrix between the qbmus in case minsptree is asked
    if minsptree and load != None and save == None:
        try:
            localadj = msptree.load_localadjmat(load + '_localadj.npz')
        except:
            raise KeyError('%s_localadj.npz is missing or have a wrong name'%load)
        try:
            with open(load+'_paths.pkl', 'rb') as f:
                paths = pickle.load(f)
        except:
            raise KeyError('%s_paths.pkl is missing or have a wrong name'%load)
    elif minsptree and load == None:
        localadj, paths = msptree.get_localadjmat(som.umat,som.adj,bmus,verbose=True)
        if save is not None:
            scipy.sparse.save_npz(save + '_localadj.npz', localadj)
            with open(save + '_paths.pkl', 'wb') as f:
                pickle.dump(paths, f)
    elif not minsptree and (load != None or save != None):
        raise ValueError('The loading or calculation and posterior saving of the local Adjacency Matrix'
                         'is done to generate the minsptree. If not minsptree then there is no need to load'
                         ' or save anything')

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
    parser.add_argument('--minsptree',help='Plot the minimal spanning tree between BMUs', default = False, action = 'store_true')
    parser.add_argument('--save',help = 'Prefix to save the local Adjacency Matrix of the BMUs and its paths needed to compute the minsptree',default = None, type = str)
    parser.add_argument('--load',help = 'Prefix to load a precalculated local Adjacency Matrix of the BMUs and its paths needed to compute the minsptree',default = None, type = str)
    args = parser.parse_args()

    main(somfile=args.som,bmusfile=args.bmus,outname=args.outname,delimiter=args.delimiter,hideSeqs=args.hide_seqs,minsptree=args.minsptree,save=args.save,load=args.load)
