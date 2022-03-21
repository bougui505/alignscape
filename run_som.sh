#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Command line example for som_seq.py

./som_seq.py --nepochs 1 -a data/TssB.aln -b 8 --alpha 0.5 | tee som.log
