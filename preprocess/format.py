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
    for f in files:
        with open(f, "r") as fh:
            sent = ""
            targ_flag = 0 # 1 if should be writing to target
            for row in fh:        
                if row == "\n": # skip blank lines
                    continue
                if row == "@highlight\n" and not targ_flag: # on first instance of @highlight, write to src
                    print >> src_file, sent
                    sent = ""
                    targ_flag = 1
                elif row == "@highlight\n": # else also skip @highlight
                    continue
                else:
                    sent = sent + " </s> " + row.rstrip()
        print >> targ_file, sent
        counter += 1
        if not (counter % 10000) and counter > 0:
            print counter
        if counter > split_point:
            print("Creating validation set")
            src_file.close()
            targ_file.close()
            src_file = open(outfile+"-src-valid.txt", "w+")
            targ_file = open(outfile+"-targ-valid.txt", "w+")

# For the Lapata small dataset
def format2(args):
    files = [args.srcdir+x for x in os.listdir(args.srcdir)]
    files.sort()
    if args.training:
        idxfile = open(args.outfile+"-indices.txt", "w+")
        outfile = open(args.outfile+"-src.txt", "w")
    else:
        outfile = open(args.outfile+"-targ.txt", "w")
    for f in files:
        with open(f, "r") as fh:
            doc = ""
            indices = []
            for row in fh:
                if row == "\n":
                    continue
                if args.training:
                    indices.append(row[0])
                    doc = doc + " </s> " + row[1:].rstrip()
                else:
                    doc = doc + " </s> " + row.rstrip()
        if args.training:
            print >> idxfile, ', '.join(indices)
        print >> outfile, doc
                

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--training', help="1 if training", type=int, default=0)
    parser.add_argument('--srcdir', help="Path to source training data. ", type=str)
    parser.add_argument('--outfile', help="Prefix of the output file names. ", type=str)
    parser.add_argument('--split', help="Fraction of the data in train/valid. ", type=float, default=.8)
    parser.add_argument('--format', help="Format to use", type=int, default=1)
    args = parser.parse_args(arguments)
    if args.format == 1:
        src_dir = args.srcdir
        outfile = args.outfile
        split = args.split
        format(src_dir, outfile, split)
    elif args.format == 2:
        format2(args)
    else:
        print "Oopsie."

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
