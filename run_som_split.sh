#!/usr/bin/env zsh
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

set -e  # exit on error
# set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -a, --alndir directory with alignments
    --somcmd SOM command to run
    -r, --reset reset the scheduler for each run
    --sigma, sigma for the SOM
EOF
}

ALNDIR='None'  # Default value
SOMCMD='None'
RESET=0
SIGMA='None'
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -a|--alndir) ALNDIR="$2"; shift ;;
        -s|--somcmd) SOMCMD="$2"; shift ;;
        -r|--reset) RESET=1;;
        --sigma) SIGMA="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

function calc () {
    python3 -c "print($@)"
}

if [ $ALNDIR = 'None' ]; then
    echo "\nYou must specify the directory with alignment files with -a, --alndir option\n"
    usage
    exit 1
fi

if [ $SOMCMD = 'None' ]; then
    echo "\nYou must specify a SOM command to run with -s, --somcmd option\ne.g.\n ./som_seq.py -b 50 --nepochs 10 --alpha 0.5 --periodic\n"
    usage
    exit 1
fi

i=0
NFILES=$(ls $ALNDIR | wc -l)
[ ! -d logs ] && mkdir logs
[ ! -d soms ] && mkdir soms
echo "Running SOM for $NFILES files"
for ALN in $(ls -v -d $ALNDIR/*); do
    iprev=$i
    (( i+=1 ))
    echo $i $ALN
    if [ $i -eq 1 ]; then
        CMD="python3 -u "$SOMCMD" --aln $ALN -o soms/som_$i.p --plot_ext png"
    else
        CMD="python3 -u "$SOMCMD" --aln $ALN -o soms/som_$i.p --load soms/som_$iprev.p --plot_ext png"
    fi
    if [ $RESET -eq 0 ]; then
        CMD="$CMD --nrun $NFILES"
    fi
    if [ $SIGMA != 'None' ]; then
        _SIGMA_=$(calc "$SIGMA-($i-1)/$NFILES")
        CMD="$CMD --sigma $_SIGMA_"
    fi
    CMD="$CMD > logs/som_$i.log"
    echo "\nRunning $CMD\nSee output in logs/som_$i.log\n"
    eval "$CMD"
done
