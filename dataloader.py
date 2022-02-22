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

import torch
from som_seq import vectorize, torchify


class Dataset(torch.utils.data.Dataset):
    """
    >>> dataset = Dataset('data/TssB.aln')
    >>> print(dataset.__len__())
    2916
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
    >>> seqnames, inputvectors = next(iter(dataloader))
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
        self.fastafile = open(fastafilename, 'r')
        self.mapping = self.seqmapping()

    def seqmapping(self):
        """
        Returns a dictionnary that give the line number for a given sequence index
        """
        mapping = dict()
        seqid = 0
        for lineid, line in enumerate(self.fastafile):
            if line[0] == '>':
                mapping[seqid] = lineid
                seqid += 1
        self.fastafile.seek(0)
        return mapping

    def __del__(self):
        self.fastafile.close()

    def __len__(self):
        nseq = sum(1 for line in self.fastafile if line[0] == '>')
        self.fastafile.seek(0)
        return nseq

    def __dim__(self):
        seqname, inputvector = self.get_seqs(0)
        return len(inputvector)

    def get_seqs(self, seqid):
        infile = open(self.fastafilename, 'r')
        lineid = self.mapping[seqid]
        for i, seqname in enumerate(infile):
            if i == lineid:
                break
        sequence = ''
        for line in infile:
            if line[0] == '>':
                break
            sequence += line.strip()
        infile.close()
        inputvector = vectorize([
            sequence,
        ]).squeeze()
        inputvector = torchify(inputvector)
        return seqname.strip(), inputvector

    def __getitem__(self, index):
        seq = self.get_seqs(index)
        return seq


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
