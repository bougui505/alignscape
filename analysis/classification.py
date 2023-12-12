import numpy as np
from alignscape.utils import models
from alignscape import plot_umat
import matplotlib.pyplot as plt


def main(somfile, outname, delimiter, uclass, k, plot_ext='png'):
    knn = models.KNeighborsBMU(k)
    somobj = models.load_som(somfile)
    titles = [delimiter.join(label.split(delimiter)[1:])
              for label in somobj.labels]
    types = [label.split(delimiter)[0].replace('>', '')
             for label in somobj.labels]
    bmus = np.asarray([np.ravel_multi_index(bmu, somobj.umat.shape)
                       for bmu in somobj.bmus])
    dm = models.load_dmatrix(somobj)
    idxs_unclass, idxs_class, types_unclass, types_class, \
        bmus_unclass, bmus_class = models.split_data(np.asarray(types),
                                                     np.asarray(bmus),
                                                     uclass)
    titles_unclass = [titles[idx] for idx in idxs_unclass]
    knn.fit(dm, bmus_class, types_class, bmus_unclass)
    f = open(outname+'.csv', 'w')
    for idx, bmu, title in zip(idxs_unclass, bmus_unclass, titles_unclass):
        predicted_type = knn.predict(bmu)
        types[idx] = predicted_type
        f.write(f'{title},{predicted_type}\n')
    f.close()
    plot_umat._plot_umat(somobj.umat, somobj.bmus, types, hideSeqs=False)
    plt.savefig(outname+'.'+plot_ext)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--somfile', help='Som file', required=True)
    parser.add_argument('-o', '--outname', help='Fasta outname', required=True)
    parser.add_argument('-d', '--delimiter',
                        help='Delimiter to split the sequence'
                        ' titles in class and identifier', required=True)
    parser.add_argument('--uclass', '--uclass', help='Class identifier for'
                        'unclassified sequences', required=True)
    parser.add_argument('-k', '--k', help='K of k-neighbours alhorithm',
                        default=2)
    parser.add_argument('--plot_ext', help='Filetype extension for'
                        ' the UMAT plots (default: png)', default='png')

    args = parser.parse_args()
    main(somfile=args.somfile, outname=args.outname, delimiter=args.delimiter,
         uclass=args.uclass, k=args.k, plot_ext=args.plot_ext)
