#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Simple baseline: grab the first k sentences from every document
'''

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
import pdb
import re
from collections import defaultdict

def prefix(args):
    try:
        k = args.k
        srctxt = open(args.srctxt, 'r')
        docs = srctxt.read().split('\n')[:-1]
        predfile = h5py.File(args.predfile, 'r')
        srcfile = h5py.File(args.srcfile, 'r')
        order = np.array(srcfile['source_order'])
        sorted_docs = [] # need to get pruned version of docs
        for idx in order:
            sorted_docs.append(docs[idx])
        path = args.outfile+'/SYSTEM/'+args.system + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        for i, doc in enumerate(sorted_docs):
            sents = doc.split("</s>") # get corresponding sentences
            summary = sents[:k]
            with open(path+"news"+str(i)+"." + args.system + ".system", "w+") as fh:
                for s in summary:
                    fh.write(s+'\n')
    except Exception as e:
        pdb.set_trace()

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--k', help="Number of highest scoring sentences to take", type=int, default=3)
    parser.add_argument('--srcfile', help="Path to the source hdf5 file to get sentence order from. ", type=str)
    parser.add_argument('--srctxt', help="Path to the source text. ", type=str)
    parser.add_argument('--predfile', help="Path to the predictions. ", type=str)
    parser.add_argument('--outfile', help="Path to the folder that will contain the files. ", type=str)
    parser.add_argument('--system', help="Name of system; \'gold\' for gold", type=str, default='ne')
    parser.add_argument('--rouge', help="Generate ROUGE.properties or not; 2 for only ROUGE; 0 for none", type=int, default=1)
    parser.add_argument('--rougetype', help="Type of ROUGE score", type=str, default='normal')
    parser.add_argument('--ngram', help="Ngram size for ROUGE calculation", type=str)
    parser.add_argument('--rougedir', help="Name of directory to ROUGE system + reference files", type=str)
    args = parser.parse_args(arguments)
    if args.rouge != 2:
        gen_preds(args) # need to generate predictions for target
    if args.rouge:
        gen_rouge_properties(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
