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

import re
import torch
import numpy as np


# add a character for gap opening
def _substitute_opening_gap_char(seq):
    rex = re.compile('[A-Z]-')
    newseq = list(seq)
    if newseq[0] == "-":
        newseq[0] = "|"
    iterator = rex.finditer(seq)
    for match in iterator:
        try:
            newseq[match.span()[1] - 1] = "|"
        except:
            continue
    return "".join(newseq)


# transform a sequence to a vector
def seq2vec(sequence, dtype='prot'):
    """
    - sequence: string
    """
    aalist = list('ABCDEFGHIKLMNPQRSTVWXYZ|-')
    nucllist = list('ATGCSWRYKMBVHDN|-')
    if dtype == 'prot':
        mapper = dict([(r, i) for i, r in enumerate(aalist)])
        naa_types = len(aalist)
    elif dtype == 'nucl':
        mapper = dict([(r, i) for i, r in enumerate(nucllist)])
        naa_types = len(nucllist)
    else:
        raise ValueError("dtype must be 'prot' or 'nucl'")
    sequence = _substitute_opening_gap_char(sequence)
    naa = len(sequence)
    vec = np.zeros((naa, naa_types))
    for i, res in enumerate(list(sequence)):
        ind = mapper[res]
        vec[i, ind] = 1.
    return vec


def vectorize(sequences, dtype='prot'):
    vectors = np.asarray([seq2vec(s, dtype).flatten() for s in sequences])
    return vectors


class Dataset(torch.utils.data.Dataset):
    """
    >>> dataset = Dataset('data/TssB.aln')
    >>> print(dataset.__len__())
    2916
    >>> seqname, inputvector = dataset.get_seq(23)
    >>> print(seqname)
    >A0A6I3XKS6
    >>> seqname, inputvector = dataset.get_seq(10)
    >>> print(seqname)
    >A0A143PHX8
    >>> print(inputvector)
    [0. 0. 0. ... 0. 0. 1.]
    >>> print(dataset.__len__())
    2916
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
    >>> dataloader_iter = iter(dataloader)
    >>> seqnames, inputvectors = next(dataloader_iter)
    >>> print(seqnames)
    ('>...', '>...', '>...', '>...', '>...', '>...', '>...', '>...', '>...', '>...')
    >>> print(inputvectors.shape)
    torch.Size([10, 5075])
    """
    def __init__(self, fastafilename):
        """
        fastafile
        """
        self.fastafilename = fastafilename
        self.fastafile = open(fastafilename, 'rb')
        self.mapping = self.seqmapping()
        self.dim = None

    def seqmapping(self):
        """
        Returns a dictionnary that give the line number for a given sequence index
        """
        self.fastafile.seek(0)
        mapping = dict()
        seqid = 0
        offset = 0
        for i, line in enumerate(self.fastafile):
            if line.decode()[0] == '>':
                mapping[seqid] = offset
                seqid += 1
            offset += len(line)
        self.fastafile.seek(0)
        return mapping

    def __del__(self):
        self.fastafile.close()

    def __len__(self):
        self.fastafile.seek(0)
        nseq = sum(1 for line in self.fastafile if line.decode()[0] == '>')
        self.fastafile.seek(0)
        return nseq

    def __dim__(self):
        seqname, inputvector = self.get_seq(0)
        return len(inputvector)

    def get_seq(self, seqid):
        fastafile = open(self.fastafilename, 'rb')
        offset = self.mapping[seqid]
        fastafile.seek(offset)
        seqname = fastafile.readline().decode()
        sequence = ''
        for line in fastafile:
            if line.decode()[0] == '>':
                break
            sequence += line.decode().strip()
        inputvector = vectorize([
            sequence,
        ]).squeeze()
        if self.dim is None:
            self.dim = len(inputvector)
        if len(inputvector) != self.dim:
            raise ValueError(
                f'Inputvector of incorrect length ({len(inputvector)} should be {self.dim}) for sequence {seqname}')
        return seqname.strip(), inputvector

    def __getitem__(self, index):
        seq = self.get_seq(index)
        return seq


def test_parallel(num_workers, batch_size=10, nloop=100):
    dataset = Dataset('data/TssB.aln')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_iter = iter(dataloader)
    t0 = time.time()
    for i in range(nloop):
        seqnames, inputvectors = next(dataloader_iter)
    deltat = time.time() - t0
    print(f'Timing for {num_workers} worker(s), batch_size {batch_size} and {nloop} loops: {deltat:.3f} s')
    return deltat


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    import time
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--test_parallel', help='Test the code for parallel execution', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.test_parallel:
        t1 = test_parallel(1, nloop=500, batch_size=5)
        t4 = test_parallel(4, nloop=500, batch_size=5)
        speedup = t1 / t4
        print(f'Speedup: {speedup:.4f}')