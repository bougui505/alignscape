#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Dec  7 09:42:07 2023

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection
set -x

MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

ALIGNSCAPEDIR="/usr/local/lib/python3.11/dist-packages/alignscape"

[ ! -d testout ] && mkdir testout
singularity run alignscape.sif head -60 $ALIGNSCAPEDIR/data/T6SS/TssB/TssB.fasta > $MYTMP/TssB60.fasta
if [[ $(lspci | grep -c -i '.* vga .* nvidia .*') -gt 0 ]]; then
    echo "GPU detected"
    NVOPTION="--nv"
else
    echo "CPU only"
    NVOPTION=""
fi

#singularity run -B $(pwd) $NVOPTION alignscape.sif $ALIGNSCAPEDIR/align_scape.py -a $MYTMP/TssB60.fasta -b 10 --nepochs 100 -o testout/som
singularity run -B $(pwd) $NVOPTION alignscape.sif python3 $ALIGNSCAPEDIR/scripts/test_postprocess.py
