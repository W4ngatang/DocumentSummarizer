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

def gen_preds(args):
    if not args.gold:
        k = args.k
        srctxt = open(args.srctxt, 'r')
        docs = srctxt.read().split('\n')[:-1]
        predfile = h5py.File(args.predfile, 'r')
        preds = np.array(predfile['preds'])
        srcfile = h5py.File(args.srcfile, 'r')
        order = np.array(srcfile['source_order'])
        lengths = np.array(srcfile['target_l_all'])
        sorted_docs = [] # need to get pruned version of docs
        for idx in order:
            sorted_docs.append(docs[idx])
        for i, (doc,pred) in enumerate(zip(sorted_docs,preds)):
            idxs = (-pred[1:lengths[i]-1]).argsort()[:k] # get the k-highest scoring indices; NEED TO NOT COUNT BOS/EOS
            idxs.sort() # sort them so they follow order of the article
            sents = doc.split("</s>") # get corresponding sentences
            summary = [sents[idx] for idx in idxs]
            with open(args.outfile+"/news"+str(i)+"_NE.txt", "w+") as fh:
                for s in summary:
                    fh.write(s+'\n')
    else:
        srcfile = h5py.File(args.srcfile, 'r')
        order = np.array(srcfile['source_order'])
        srctxt = open(args.srctxt, 'r')
        docs = srctxt.read().split('\n')[:-1]
        sorted_docs = [] # need to get pruned version of docs
        for idx in order:
            sorted_docs.append(docs[idx])
        for i, summary in enumerate(sorted_docs):
            with open("rouge/" + args.rougedir + '/system/' + args.outfile+"/news"+str(i)+"_gold.txt", "w+") as fh:
                for s in summary.split(" </s> "):
                    fh.write(s+'\n')

def gen_rouge_properties(args):
    with open('rouge.properties', 'w') as fh:
        fh.write('project.dir=rouge/'+args.rougedir+'\n')            
        fh.write('rouge.type='+args.rougetype+'\n')            
        fh.write('ngram='+args.ngram+'\n')
        fh.write('stopwords.use=false\n')
        fh.write('stopwords.file=rouge/resources/stopwords-rouge-default.txt\n')
        fh.write('topic.type=nn|jj\n')
        fh.write('synonyms.use=false\n')
        fh.write('synonyms.dir=default\n')
        fh.write('pos_tagger_name=english-bidirectional-distsim.tagger\n')
        fh.write('output=file\n')
        fh.write('outputFile=results.csv\n')

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--k', help="Number of highest scoring sentences to take", type=int, default=3)
    parser.add_argument('--srcfile', help="Path to the source hdf5 file to get sentence order from. ", type=str)
    parser.add_argument('--srctxt', help="Path to the source text. ", type=str)
    parser.add_argument('--predfile', help="Path to the predictions. ", type=str)
    parser.add_argument('--outfile', help="Path to the folder that will contain the files. ", type=str)
    parser.add_argument('--gold', help="1 if gold, predictions otherwise", type=int, default=0)
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
