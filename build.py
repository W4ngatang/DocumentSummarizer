#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prune dataset for sentence extraction according to
   Cheng and Lapata (2016)
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
import pdb
import nltk
import re
from collections import defaultdict

def entity_overlap(src, targ):
    def entitify(t):
        entities = []
        if hasattr(t, 'node') and t.node:
            if t.node == 'NE':
                entities.append(' '.join([child[0] for child in t]))
            else:
                for child in t:
                    entities.extend(entitify(child))
        return set(entities)

    def get_entities(doc):
        sentences = nltk.sent_tokenize(doc) # some nltk preprocessing: tokenize, tag, chunk, NER
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        chunked_sentences = nltk.batch_ne_chunk(tagged_sentences, binary=True)

        entities = []
        for t in chunked_sentences:
            entities.append(entitify(t))

        return entities

    # Get target entities and join into one set
    targ_entities = get_entities(targ)
    targ_entity = set()
    for t in targ_entities:
        targ_entity |= t

    # Iterate over source entities and compute overlap
    src_entities = get_entities(src)
    num_entities = []
    for src_entity in src_entities:
        num_entities.append(float(len(src_entity & targ_entity)))
    
    return num_entities
        

def clean_string(string): # some Yoon Kim magic
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()
    
# for a doc and a summary, return binary array where 1 if sentence should be kept
def featurize(source, targ, weights, thresh):

    # get unigram and bigram features
    def gram_feats(src, targ):
        src_w = src.split()
        targ_w = targ.split()
        src_uni = set(src_w)
        targ_uni = set(targ_w)
        uni = len(src_uni & targ_uni)

        src_bi = []
        targ_bi = []
        for i in xrange(len(src_w)-1):
            src_bi.append(src_w[i] + " " + src_w[i+1])
        for i in xrange(len(targ_w)-1):
            targ_bi.append(targ_w[i] + " " + targ_w[i+1])
        bi = len(set(src_bi) & set(targ_bi))
        return float(uni), float(bi)

    num_entities = entity_overlap(source.replace("</s>", ""), targ.replace("</s>", "")) # get #entity overlap per sentence
    sentences = [clean_string(s) for s in source.strip().split("</s>")]
    indicators = []
    if len(num_entities) != len(sentences[1:]):
        return [0] # when NLTK messes up
    for i, sentence in enumerate(sentences[1:]): # artifact of split having leading ''
        uni, bi = gram_feats(sentence, targ) # below, take dot product of weights and features
        try:
            score = sum([feat*weight for feat,weight in zip([uni, bi, float(i+1), num_entities[i]], weights)])
        except:
            pdb.set_trace()
            print sentence
        indicators.append(source > thresh)

    return indicators

# note: kind of specific to CNN+Dailymail
def prune(args):
    sources = open(args.srcfile, 'r')
    targets = open(args.targfile, 'r')
    src_out = open(args.outfile+"-src.txt", 'w')
    targ_out = open(args.outfile+"-targ.txt", 'w')
    sources_all = sources.read().split("\n")
    targets_all = targets.read().split("\n")
    weights = [args.uni, args.bi, args.position, args.ne]
    thresh = args.thresh

    rejected = 0 # logging purposes
    sub3 = 0
    for i in xrange(len(sources_all)):
        indicators = featurize(sources_all[i], targets_all[i], weights, thresh)
        if sum(indicators):
            print >> src_out, sources_all[i]
            print >> targ_out, indicators
            if sum(indicators) < 3:
                sub3 += 1
        else:
            rejected += 1
        if not (i % 100):
            print "Finished: %d, rejected: %d, sub 3: %d" % (i, rejected, sub3)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('srcfile', help="Path to source training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", type=str)
    parser.add_argument('targfile', help="Path to target training data. ", type = str)
    parser.add_argument('--outfile', help="Prefix of the output file names. ", type=str, default="extract")
    parser.add_argument('--split', help="Fraction of the data in train/valid. ", type=float, default=.8)
    parser.add_argument('--uni', help="Unigram weight. ", type=float, default=1.)
    parser.add_argument('--bi', help="Bigram weight. ", type=float, default=2.)
    parser.add_argument('--ne', help="Named entity overlap weight ", type=float, default=2.)
    parser.add_argument('--position', help="Position weight. ", type=float, default=-.5)
    parser.add_argument('--thresh', help="Threshold for a sentence being considered", type=float, default=5.)
    args = parser.parse_args(arguments)
    prune(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
