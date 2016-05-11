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
import re
from collections import defaultdict

def gen_preds(args, gold=0):
    if not gold:
        try:
            k = args.k
            srctxt = open(args.srctxt, 'r') # text from source
            docs = srctxt.read().split('\n')[:-1]

            predfile = h5py.File(args.predfile, 'r') # predictions
            preds = np.array(predfile['preds'])

            srcfile = h5py.File(args.srcfile, 'r') # hdf5 fed into lua
            order = np.array(srcfile['source_order'])
            lengths = np.array(srcfile['target_l_all'])

            sorted_docs = [] # need to get pruned version of docs
            for idx in order:
                sorted_docs.append(docs[idx])

            path = 'tmp_SYSTEM/'+args.system + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            for i, (doc,pred) in enumerate(zip(sorted_docs,preds)):
                idxs = (-pred[1:lengths[i]-1]).argsort()[:k] # get the k-highest scoring indices; NEED TO NOT COUNT BOS/EOS
                idxs.sort() # sort them so they follow order of the article
                sents = doc.split("</s>") # get corresponding sentences
                summary = [sents[idx] for idx in idxs]
                with open(path+"news"+str(i)+"." + args.system + ".system", "w+") as fh:
                    for s in summary:
                        fh.write(s+'\n')
        except Exception as e:
            pdb.set_trace()
    else: # lazy coding
        try:
            srcfile = h5py.File(args.srcfile, 'r')
            order = np.array(srcfile['source_order'])
            goldtxt = open(args.goldfile, 'r')
            docs = goldtxt.read().split('\n')[:-1]
            sorted_docs = [] # need to get pruned version of docs
            for idx in order:
                sorted_docs.append(docs[idx])
            path = 'tmp_GOLD/'
            if not os.path.exists(path):
                os.makedirs(path)
            for i, summary in enumerate(sorted_docs):
                task = "news"+str(i)
                if not os.path.exists(path+task):
                    os.makedirs(path+task)
                with open(path+task+"/"+task+".1.gold", "w+") as fh:
                    for s in summary.split(" </s> "):
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
    parser.add_argument('--goldfile', help="Path to the gold standard summaries. ", type=str)
    parser.add_argument('--outfile', help="Path to the folder that will contain the files. ", type=str, default='')
    parser.add_argument('--system', help="Name of system; \'gold\' for gold", type=str, default='ne')
    parser.add_argument('--rougedir', help="Name of directory to ROUGE system + reference files", type=str)
    args = parser.parse_args(arguments)
    gen_preds(args) # generate predictions
    gen_preds(args, 1) # generate gold

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
