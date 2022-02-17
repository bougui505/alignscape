#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

paste =(ls -dv logs/som_*.log | xargs cat | grep "| alpha:" | awk '{print $5}') \
    =(ls -dv logs/som_*.log | xargs cat | grep "| sigma:" | awk '{print $8}') \
    | plot --dax 12 --ylabel 'alpha' --ylabel2 'sigma' --xlabel 'iterations'
