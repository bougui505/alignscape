#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

mkdir TssB_split
split -l 600 -a 2 -d --additional-suffix=.aln TssB.aln TssB_split/TssB
