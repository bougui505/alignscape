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

import glob
import dill as pickle  # For tricky pickles
import os
import numpy as np
import torch
from pathlib import Path
import re
import som_seq

file_pattern = re.compile(r'.*?(\d+).*?')


def get_order(infile):
    """
    Get file order based on name numbering
    See: https://stackoverflow.com/a/62941534/1679629
    """
    match = file_pattern.match(Path(infile).name)
    if not match:
        return np.inf
    return int(match.groups()[0])


def get_filelist(indir):
    filelist = glob.glob(f'{indir}/*')
    filelist.sort(key=get_order)
    outprintlist = '\n'.join([f"{i}\t\t{f}" for i, f in enumerate(filelist)])
    print("#" * 80)
    print(f"\nSorted list of file:\n\n{outprintlist}\n")
    print("#" * 80)
    return filelist


def get_probs(umat):
    probs = -umat
    probs = probs - probs.min()  # The min is now 0.
    probs /= abs(probs.max())  # and the max 1.
    probs /= probs.sum()
    return probs


def get_centroids_dataset(somobj, npts):
    """
    Build a list of codebooks vectors extracted from the given som centroids
    npts: number of points to return
    """
    som_centroids = somobj.centroids
    somsize, dim = som_centroids.shape
    probs = get_probs(somobj.umat).flatten()
    inds = np.random.choice(somsize, size=npts, p=probs)
    centroidnames = [f'som_{i}' for i in inds]
    return som_centroids[inds], centroidnames


def get_trainset(somobj, dataset, seqnames, memory=1.):
    npts = int(len(dataset) * memory)
    if npts == 0:
        return dataset, seqnames
    else:
        som_centroids, centroidnames = get_centroids_dataset(somobj=somobj, npts=npts)
        som_centroids = som_centroids.numpy()
        trainset = np.concatenate((som_centroids, dataset))
        trainseqnames = np.concatenate((centroidnames, seqnames))
        return trainset, trainseqnames


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--alndir', help='directory with alignments', required=True)
    parser.add_argument('-b', '--batch', help='Batch size (default: 100)', default=100, type=int)
    parser.add_argument('--somside', help='Size of the side of the square SOM', default=50, type=int)
    parser.add_argument('--alpha', help='learning rate', default=None, type=float)
    parser.add_argument('--sigma', help='Learning radius for the SOM', default=None, type=float)
    parser.add_argument('--nepochs', help='Number of SOM epochs', default=2, type=int)
    parser.add_argument("-o", "--out_name", default='som.p', help="name of pickle to dump (default som.p)")
    parser.add_argument('--periodic', help='Periodic toroidal SOM', action='store_true')
    parser.add_argument('--scheduler',
                        help='Which scheduler to use, can be linear, exp or half (exp by default)',
                        default='exp')
    parser.add_argument(
        '--memory',
        help=
        'Proportion (relative to training set) to keep from the previous SOM map for the new training set (default=1.)',
        default=1.,
        type=float)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on', device)

    baseoutname = os.path.splitext(args.out_name)[0]
    filelist = get_filelist(args.alndir)
    if os.path.isdir('soms'):
        raise OSError('soms directory already exists. Remove it or rename it for a new run')
    else:
        os.mkdir('soms')
    for i, ali in enumerate(filelist):
        outname = f'soms/{baseoutname}_{i}.p'
        if i == 0:
            som_seq.main(ali=ali,
                         batch_size=args.batch,
                         somside=args.somside,
                         nepochs=args.nepochs,
                         alpha=args.alpha,
                         sigma=args.sigma,
                         load=None,
                         periodic=args.periodic,
                         scheduler=args.scheduler,
                         nrun=1,
                         outname=outname,
                         doplot=True,
                         plot_ext='png')
        else:
            # load previous SOM as starting point
            prev_som = f'soms/{baseoutname}_{i-1}.p'
            som = pickle.load(open(prev_som, 'rb'))
            seqnames, sequences = som_seq.read_fasta(ali)
            seqnames = np.asarray(seqnames)
            inputvectors = som_seq.vectorize(sequences, dtype='prot')
            inputvectors, seqnames = get_trainset(somobj=som,
                                                  dataset=inputvectors,
                                                  seqnames=seqnames,
                                                  memory=args.memory)
            som_seq.main(inputvectors=inputvectors,
                         seqnames=seqnames,
                         batch_size=args.batch,
                         somside=args.somside,
                         nepochs=args.nepochs,
                         alpha=args.alpha,
                         sigma=args.sigma,
                         somobj=som,
                         periodic=args.periodic,
                         scheduler=args.scheduler,
                         nrun=1,
                         outname=outname,
                         doplot=True,
                         plot_ext='png')
