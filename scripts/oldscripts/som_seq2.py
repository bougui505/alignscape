""""
This is a hard copy of a file that yields a good speedup when used in the SOM context.
"""

import functools
import os
import dill as pickle  # For tricky pickles
# import pickle
import quicksom.som
from Bio.SubsMat import MatrixInfo
import numpy as np
import torch
import seqdataloader

try:
    import functorch

    FUNCTORCH_AVAIL = False
except ImportError:
    print("Running without functorch, please install it")
    FUNCTORCH_AVAIL = False


def read_fasta(fastafilename, names=None):
    """
    """
    sequences = []
    seq = None
    seqname = None
    seqnames = []
    with open(fastafilename) as fastafile:
        for line in fastafile:
            if line[0] == ">":  # sequence name
                if seq is not None:
                    if names is None or seqname in names:
                        seqnames.append(seqname)
                        sequences.append(seq)
                seqname = line[1:].strip()
                seq = ''
            else:
                seq += line.strip()
    if names is None or seqname in names:
        sequences.append(seq)
        seqnames.append(seqname)
    return seqnames, sequences


def get_blosum62():
    aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ|-')
    b62 = np.zeros((23, 23))
    for k in MatrixInfo.blosum62:
        i0 = aalist.index(k[0])
        i1 = aalist.index(k[1])
        b62[i0, i1] = MatrixInfo.blosum62[k]
        b62[i1, i0] = MatrixInfo.blosum62[k]
    return b62


def torchify(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        x = torch.from_numpy(x)
    except TypeError:
        pass
    x = x.to(device)
    x = x.float()
    return x


def score_matrix_vec(vec1, vec2, dtype="prot", gap_s=-5, gap_e=-1, b62=None, NUC44=None):
    """
    PyTorch Implementation
    """
    if dtype == 'prot':
        matrix = b62
    elif dtype == 'nucl':
        matrix = NUC44
    else:
        raise ValueError("dtype must be 'prot' or 'nucl'")
    vec1 = vec1.float()
    vec2 = vec2.float()
    matrix = matrix.float()
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    if vec2.ndim == 2:
        vec2 = vec2[None, ...]
    matv2 = torch.matmul(matrix[None, ...], torch.swapaxes(vec2[..., :-2], 1, 2))
    scores = torch.einsum('aij,bji->ab', vec1[..., :-2], matv2)

    gaps1, gaps2 = vec1[..., -2], vec2[..., -2]
    exts1, exts2 = vec1[..., -1], vec2[..., -1]
    if FUNCTORCH_AVAIL:
        print('using fancy vmap')
        vmax = functorch.vmap(torch.maximum, in_dims=(0, None))
        max_gaps = vmax(gaps1, gaps2)
        max_gaps_aggregated = max_gaps.sum(axis=2)
        max_exts = vmax(exts1, exts2)
        max_exts_aggregated = max_exts.sum(axis=2)
        scores = scores + max_gaps_aggregated * gap_s + max_exts_aggregated * gap_e
    else:
        print('using no vmap')
        for i in range(len(vec1)):
            # scores.shape = (a, b) with a: size of batch and b size of SOM
            scores[i] += torch.maximum(gaps1[i, ...], gaps2).sum(axis=1) * gap_s
            scores[i] += torch.maximum(exts1[i, ...], exts2).sum(axis=1) * gap_e
        # scores = list(scores.to('cpu').numpy())
    if len(scores) == 1:
        return scores[0]
    else:
        return scores


def seqmetric(seqs1, seqs2, b62):
    nchar = 25
    batch_size = seqs1.shape[0]
    seqlenght = seqs1.shape[-1] // nchar
    n2 = seqs2.shape[0]
    seqs1 = seqs1.reshape((batch_size, seqlenght, nchar))
    seqs2 = seqs2.reshape((n2, seqlenght, nchar))
    scores = score_matrix_vec(seqs1, seqs2, b62=b62)
    return -scores


def main(ali=None,
         inputvectors=None,
         seqnames=None,
         batch_size=None,
         somside=None,
         nepochs=None,
         alpha=None,
         sigma=None,
         load=None,
         somobj=None,
         periodic=None,
         scheduler=None,
         outname=None,
         doplot=None,
         plot_ext=None):
    if inputvectors is None and ali is None:
        raise ValueError('inputvectors or ali argument must be set. Both are None.')
    if inputvectors is not None and ali is not None:
        raise ValueError('inputvectors and ali arguments are given. Only one must be set.')
    if inputvectors is not None and seqnames is None:
        raise ValueError('When inputvectors argument is given, seqnames argument must be given too.')
    if inputvectors is not None and seqnames is not None:
        if len(inputvectors) != len(seqnames):
            raise ValueError(
                f'inputvectors (len: {len(inputvectors)}) and seqnames (len: {len(seqnames)}) have different length.')
    if load is not None and somobj is not None:
        raise ValueError('load and somobj cannot be both set')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on', device)

    dtype = 'prot'

    b62 = get_blosum62()
    b62 = torchify(b62)

    if ali is not None:
        # seqnames, sequences = read_fasta(ali)
        # seqnames = np.asarray(seqnames)
        # inputvectors = vectorize(sequences, dtype=dtype)
        dataset = seqdataloader.SeqDataset(ali)
        num_workers = os.cpu_count()
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 worker_init_fn=functools.partial(seqdataloader.workinit,
                                                                                  fastafilename=ali))
    n_inp = dataset.__len__()
    print('n_input:', n_inp)
    dim = dataset.__dim__()
    # inputvectors = torchify(inputvectors)
    baseoutname = os.path.splitext(outname)[0]

    if load is not None:
        print(f'Loading {load}')
        with open(load, 'rb') as somfile:
            som = pickle.load(somfile)
            # somsize = som.m * som.n
            som.to_device(device)
    elif somobj is not None:
        print(f'Using given som object: {somobj}')
        som = somobj
        som.to_device(device)
    else:
        # somsize = somside**2
        som = quicksom.som.SOM(somside,
                               somside,
                               n_epoch=nepochs,
                               dim=dim,
                               alpha=alpha,
                               sigma=sigma,
                               device=device,
                               periodic=periodic,
                               metric=lambda s1, s2: seqmetric(s1, s2, b62),
                               sched=scheduler)
    print('batch_size:', batch_size)
    print('sigma:', som.sigma)
    if som.alpha is not None:
        print('alpha:', som.alpha)
    som.fit(dataset=dataloader,
            batch_size=batch_size,
            do_compute_all_dists=False,
            unfold=False,
            normalize_umat=False,
            sigma=sigma,
            alpha=alpha,
            logfile=f'{baseoutname}.log')
    print('Computing BMUS')
    som.bmus, som.error, som.density, som.labels = som.predict(dataset=dataset,
                                                               batch_size=batch_size,
                                                               return_density=True)
    index = np.arange(len(som.bmus))
    out_arr = np.zeros(n_inp, dtype=[('bmu1', int), ('bmu2', int), ('error', float), ('index', int), ('label', 'U512')])
    out_arr['bmu1'] = som.bmus[:, 0]
    out_arr['bmu2'] = som.bmus[:, 1]
    out_arr['error'] = som.error
    out_arr['index'] = index
    out_arr['label'] = som.labels
    out_fmt = ['%d', '%d', '%.4g', '%d', '%s']
    out_header = '#bmu1 #bmu2 #error #index #label'
    np.savetxt(f"{baseoutname}_bmus.txt", out_arr, fmt=out_fmt, header=out_header, comments='')
    if doplot:
        import matplotlib.pyplot as plt
        plt.matshow(som.umat)
        plt.colorbar()
        plt.savefig(f'{baseoutname}_umat.{plot_ext}')
    som = som.to_device('cpu')
    pickle.dump(som, open(outname, 'wb'))


if __name__ == '__main__':
    import argparse

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--aln', help='Alignment file', required=True)
    parser.add_argument('-b', '--batch', help='Batch size (default: 100)', default=100, type=int)
    parser.add_argument('--somside', help='Size of the side of the square SOM', default=50, type=int)
    parser.add_argument('--alpha', help='learning rate', default=None, type=float)
    parser.add_argument('--sigma', help='Learning radius for the SOM', default=None, type=float)
    parser.add_argument('--nepochs', help='Number of SOM epochs', default=2, type=int)
    parser.add_argument("-o", "--out_name", default='som.p', help="name of pickle to dump (default som.p)")
    parser.add_argument('--noplot', help='Do not plot the resulting U-matrix', action='store_false', dest='doplot')
    parser.add_argument('--plot_ext', help='Filetype extension for the U-matrix plot (default: pdf)', default='pdf')
    parser.add_argument('--periodic', help='Periodic toroidal SOM', action='store_true')
    parser.add_argument('--scheduler',
                        help='Which scheduler to use, can be linear, exp or half (exp by default)',
                        default='exp')
    parser.add_argument('--load', help='Load the given som pickle file and use it as starting point for a new training')
    args = parser.parse_args()

    main(ali=args.aln,
         batch_size=args.batch,
         somside=args.somside,
         nepochs=args.nepochs,
         alpha=args.alpha,
         sigma=args.sigma,
         load=args.load,
         periodic=args.periodic,
         scheduler=args.scheduler,
         outname=args.out_name,
         doplot=args.doplot,
         plot_ext=args.plot_ext)
