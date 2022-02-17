#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

ls -dv logs/som_*.log | xargs cat | grep "| sigma:" | awk '{print $8}' | plot
