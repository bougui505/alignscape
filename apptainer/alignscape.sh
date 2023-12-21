#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Dec  7 10:55:40 2023

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

ARGS="$@"
[ -z "$ARGS" ] && ARGS="-h"

if [[ $(lspci | grep -c -i 'nvidia') -gt 0 ]]; then
    echo "GPU detected"
    NVOPTION="--nv"
else
    echo "CPU only"
    NVOPTION=""
fi

singularity run -B $(pwd) $NVOPTION $DIRSCRIPT/alignscape.sif align_scape $ARGS \
