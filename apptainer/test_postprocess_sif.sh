#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Dec  7 10:45:44 2023

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection
set -x

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

ALIGNSCAPEDIR="/usr/local/lib/python3.11/dist-packages/alignscape"
singularity run --nv alignscape.sif $ALIGNSCAPEDIR/test_postprocess.py
