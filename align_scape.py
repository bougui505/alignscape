#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

import functools
import os
import dill as pickle
from Bio.Align import substitution_matrices
import numpy as np
import torch
from alignscape.utils import seqdataloader
from alignscape.utils import memory

aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ|-')

try:
    import functorch

    FUNCTORCH_AVAIL = True
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
    b62 = np.zeros((23, 23))
    for pair, value in substitution_matrices.load('BLOSUM62').items():
        if '*' in pair:
            continue
        i0 = aalist.index(pair[0])
        i1 = aalist.index(pair[1])
        b62[i0, i1] = value
        b62[i1, i0] = value
    return b62


def torchify(x, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        x = torch.from_numpy(x)
    except TypeError:
        pass
    x = x.to(device)
    x = x.float()
    return x


def rscore_matrix_vec(vec1, vec2, dtype='prot', gap_s=5, gap_e=1,
                      b62=None, NUC44=None):
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
    nchars = len(aalist)
    rscore = np.shape(vec1)[1] * ((matrix.sum()) + nchars *
                                  (gap_s + gap_e)) / (np.shape(matrix)[0]
                                                      * np.shape(matrix)[1]
                                                      + nchars * 2)
    rscores = torch.tile(rscore, (np.shape(vec1)[0], np.shape(vec2)[0]))
    return rscores


def score_matrix_vec(vec1, vec2, dtype="prot", gap_s=-5, gap_e=-1,
                     b62=None, NUC44=None):
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
    matv2 = torch.matmul(matrix[None, ...],
                         torch.swapaxes(vec2[..., :-2], 1, 2))
    scores = torch.einsum('aij,bji->ab', vec1[..., :-2], matv2)
    gaps1, gaps2 = vec1[..., -2], vec2[..., -2]
    exts1, exts2 = vec1[..., -1], vec2[..., -1]
    if FUNCTORCH_AVAIL:
        vmax = functorch.vmap(torch.maximum, in_dims=(0, None))
        max_gaps = vmax(gaps1, gaps2)
        max_gaps_aggregated = max_gaps.sum(axis=2)
        max_exts = vmax(exts1, exts2)
        max_exts_aggregated = max_exts.sum(axis=2)
        scores = scores + max_gaps_aggregated * gap_s + \
            max_exts_aggregated * gap_e
    else:
        for i in range(len(vec1)):
            scores[i] += torch.maximum(gaps1[i, ...], gaps2).sum(axis=1) \
                * gap_s
            scores[i] += torch.maximum(exts1[i, ...], exts2).sum(axis=1) \
                * gap_e
    if len(scores) == 1:
        return scores[0]
    else:
        return scores


def iscore_matrix_vec(vec1, dtype="prot", gap_s=-5, gap_e=-1,
                      b62=None, NUC44=None, verbose=False):
    if dtype == 'prot':
        matrix = b62
    elif dtype == 'nucl':
        matrix = NUC44
    else:
        raise ValueError("dtype must be 'prot' or 'nucl'")
    vec1 = vec1.float()
    matrix = matrix.float()
    if vec1.ndim == 2:
        vec1 = vec1[None, ...]
    matv2 = torch.matmul(matrix[None, ...],
                         torch.swapaxes(vec1[..., :-2], 1, 2))
    if verbose:
        print(f"vec1[..., :-2]: {vec1[..., :-2].shape}")
        print(f"matv2.shape: {matv2.shape}")
    scores = torch.einsum('aij,aji->a', vec1[..., :-2], matv2)
    gaps = vec1[..., -2]
    exts = vec1[..., -1]
    gaps_aggregated = gaps.sum(axis=1)
    exts_aggregated = exts.sum(axis=1)
    scores += gaps_aggregated * gap_s + exts_aggregated * gap_e
    return scores


def seqmetric(seqs1, seqs2, b62):
    # seqs1 is the batch of input vectors and seqs2 the SOM
    nchar = 25
    batch_size = seqs1.shape[0]
    seqlenght = seqs1.shape[-1] // nchar
    n2 = seqs2.shape[0]
    seqs1 = seqs1.reshape((batch_size, seqlenght, nchar))
    seqs2 = seqs2.reshape((n2, seqlenght, nchar))
    if not torch.is_tensor(b62):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        b62 = torchify(b62, device=device)
    scores = score_matrix_vec(seqs1, seqs2, b62=b62)
    rscores = rscore_matrix_vec(seqs1, seqs2, b62=b62)

    iscores1 = iscore_matrix_vec(seqs1, b62=b62)
    iscores2 = iscore_matrix_vec(seqs2, b62=b62)

    iscores = (iscores1.reshape(-1, 1) + iscores2) / 2

    # Compute the B62 based distance
    denominators = iscores - rscores
    nominators = scores - rscores
    nominators[nominators < 0] = 0.001
    dists = (nominators) / (denominators)
    dists = -torch.log(dists) * 100

    return dists


def bmu_to_label(bmu, label, bmus):
    pass


def main(ali=None,
         batch_size=None,
         somside=None,
         nepochs=None,
         alpha=None,
         sigma=None,
         load=None,
         periodic=True,
         scheduler=None,
         outname=None,
         doplot=True,
         use_jax=False,
         plot_ext='png',
         seed=0):
    if ali is None:
        raise ValueError('ali argument must be set')

    if use_jax:
        import jax
        from alignscape.utils import jax_imports
        from alignscape.quicksom import somax
    else:
        from alignscape.quicksom import som

    # Get the data ready
    dataset = seqdataloader.SeqDataset(ali)
    # dataset.len = 20
    num_workers = os.cpu_count()
    torch.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             worker_init_fn=functools.partial(seqdataloader.workinit,fastafilename=ali))
    n_inp = dataset.__len__()
    print('n_input:', n_inp)
    dim = dataset.__dim__()
    baseoutname = os.path.splitext(outname)[0]

    # Now get the SOM ready, with the right computation function
    b62 = get_blosum62()
    if use_jax:
        device = jax.devices()[0]
        if device == 'cuda':
            device = memory.select_device()
        b62 = jax.device_put(b62, device=device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            device = memory.select_device()
        print(device)
        b62 = torchify(b62, device=device)

    if load is not None:
        print(f'Loading {load}')
        with open(load, 'rb') as somfile:
            somobj = pickle.load(somfile)
    else:
        if use_jax:
            somobj = somax.SOM(somside,
                               somside,
                               device=device,
                               n_epoch=nepochs,
                               dim=dim,
                               alpha=alpha,
                               sigma=sigma,
                               periodic=periodic,
                               metric=functools.partial(jax_imports.seqmetric_jax, b62=b62),
                               sched=scheduler,
                               seed=seed)
            somobj.jax = True
        else:
            somobj = som.SOM(somside,
                             somside,
                             n_epoch=nepochs,
                             dim=dim,
                             alpha=alpha,
                             sigma=sigma,
                             periodic=periodic,
                             metric=functools.partial(seqmetric, b62=b62),
                             sched=scheduler,
                             seed=seed)
            somobj.jax = False
    somobj.to_device(device)

    print('batch_size:', batch_size)
    print('sigma:', somobj.sigma)
    if somobj.alpha is not None:
        print('alpha:', somobj.alpha)
    somobj.fit(dataset=dataloader,
               batch_size=batch_size,
               do_compute_all_dists=False,
               unfold=False,
               normalize_umat=False,
               sigma=sigma,
               alpha=alpha,
               logfile=f'{baseoutname}.log')
    print('Computing BMUS')
    somobj.bmus, somobj.error, somobj.labels = \
        somobj.predict(dataset=dataset, batch_size=batch_size,
                       return_density=False, num_workers=1,
                       return_errors=False)

    index = np.arange(len(somobj.bmus))
    out_arr = np.zeros(n_inp, dtype=[('bmu_r', int), ('bmu_c', int),
                                     ('error', float), ('index', int),
                                     ('label', 'U512')])
    out_arr['bmu_r'] = somobj.bmus[:, 0]
    out_arr['bmu_c'] = somobj.bmus[:, 1]
    out_arr['error'] = somobj.error
    out_arr['index'] = index
    out_arr['label'] = somobj.labels
    out_fmt = ['%d', '%d', '%.4g', '%d', '%s']
    out_header = '#bmu_r #bmu_c #error #index #label'
    np.savetxt(f"{baseoutname}_bmus.txt", out_arr, fmt=out_fmt,
               header=out_header, comments='')
    # f = open(f"{baseoutname}_errors.txt", "w")
    # f.write("#quantification_error #topo_error\n%.8f %.8f" \
    #        % (quantification_error, topo_error))
    if doplot:
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        print('Saving umat')
        np.save('%s_umat' % baseoutname, somobj.umat)
        plt.matshow(somobj.umat)
        plt.colorbar()
        plt.savefig(f'{baseoutname}_umat.{plot_ext}')
    print('Saving SOM map')
    somobj.save_pickle(f'{baseoutname}.pickle')
    return somobj


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--aln', help='Alignment file (fasta format)')
    parser.add_argument('-b', '--batch', help='Batch size (default: 5)',
                        default=5, type=int)
    parser.add_argument('--somside', help='Size of the side of the square \
                        SOM (default: 50)', default=50, type=int)
    parser.add_argument('--alpha', help='learning rate', default=0.5,
                        type=float)
    parser.add_argument('--sigma', help='Learning radius for the SOM',
                        default=None, type=float)
    parser.add_argument('--nepochs', help='Number of SOM epochs (default: 100)',
                        default=100, type=int)
    parser.add_argument("-o", "--out_name", default='som',
                        help="basename (default som)")
    parser.add_argument('--noplot', help='Do not plot the resulting U-matrix',
                        action='store_false', dest='doplot')
    parser.add_argument('--plot_ext', help='Filetype extension for the \
                        U-matrix plot (default: pdf)', default='pdf')
    parser.add_argument('--noperiodic', help='Non periodic toroidal SOM',
                        action='store_false', default=True)
    parser.add_argument('--scheduler',
                        help='Which scheduler to use, can be linear,\
                        exp or half (exp by default)', default='exp')
    parser.add_argument('-j', '--jax', help='To use the jax version',
                        action='store_true')
    parser.add_argument('--load', help='Load the given som pickle file \
                        and use it as starting point for a new training')
    parser.add_argument('--seed', help='Seed for random inicialization and random \
                        batch selection', default=0)
    args = parser.parse_args()

    if args.sigma is None:
        sigma = np.sqrt(args.somside*args.somside)/4.0

    main(ali=args.aln,
         batch_size=args.batch,
         somside=args.somside,
         nepochs=args.nepochs,
         alpha=args.alpha,
         sigma=sigma,
         load=args.load,
         periodic=args.noperiodic,
         scheduler=args.scheduler,
         outname=args.out_name,
         doplot=args.doplot,
         plot_ext=args.plot_ext,
         use_jax=args.jax,
         seed=int(args.seed))
