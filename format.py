#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
import pdb
from collections import defaultdict

# note: kind of specific to CNN+Dailymail
def format(path, outfile, split=.8):
    files = [path+x for x in os.listdir(path)]
    counter = 0
    split_point = split*float(len(files))
    src_file = open(outfile+"-src-train.txt", "w+")
    targ_file = open(outfile+"-targ-train.txt", "w+")
    val_flag = 0
    for f in files:
        with open(f, "r") as fh:
            sent = ""
            targ_flag = 0 # 1 if should be writing to target
            for row in fh:        
                if row == "\n": # skip blank lines
                    continue
                if row == "@highlight\n" and not targ_flag: # on first instance of @highlight, write to src
                    print >> src_file, sent + "\n"
                    sent = ""
                    targ_flag = 1
                elif row == "@highlight\n": # else also skip @highlight
                    continue
                else:
                    sent = sent + " </s> " + row.rstrip()
        print >> targ_file, sent + "\n"
        counter += 1
        if not (counter % 10000) and counter > 0:
            print counter
        if counter > split_point and not val_flag:
            print("Creating validation set")
            src_file.close()
            targ_file.close()
            src_file = open(outfile+"-src-valid.txt", "w+")
            targ_file = open(outfile+"-targ-valid.txt", "w+")
            val_flag = 1

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcdir', help="Path to source training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", type=str)
    parser.add_argument('--outfile', help="Prefix of the output file names. ", type=str)
    parser.add_argument('--split', help="Fraction of the data in train/valid. ", type=float, default=.8)
    args = parser.parse_args(arguments)
    src_dir = args.srcdir
    outfile = args.outfile
    split = args.split
    format(src_dir, outfile, split)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
