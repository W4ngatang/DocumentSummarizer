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
from sklearn.linear_model import SGDClassifier
import pickle
#from build import featurize
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

    def get_entities(sentences):
        #sentences = nltk.sent_tokenize(doc) # some nltk preprocessing: tokenize, tag, chunk, NER
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
def featurize(source, targ):

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

    #num_entities = entity_overlap(source.split("</s>"), targ.split("</s>")) # get #entity overlap per sentence
    sentences = [clean_string(s) for s in source.split("</s>")]
    features = []
    #if len(num_entities) != len(sentences):
    #    return [0] # when NLTK messes up
    for i, sentence in enumerate(sentences[1:]): # artifact of split having leading ''
        uni, bi = gram_feats(sentence, targ) # below, take dot product of weights and features
        try:
            features.append([uni, bi, float(i+1)])#, num_entities[i]])
        except:
            pdb.set_trace()

    return features

# Tune the classifier
def tune(args):
    s = open(args.train, 'r')
    t = open(args.train_t, 'r')
    ind = open(args.train_i, 'r')
    srcs = s.read().split('\n')[:-1]
    targs = t.read().split('\n')[:-1]
    indices = ind.read().split('\n') # make sure formatted correctly`
    indices = [[int(x) for x in row.split(', ')] for row in indices[:-1]] # magic number due to split()
    X = []
    y = []

    for i in xrange(len(srcs)):
        X += featurize(srcs[i], targs[i])
        for j in xrange(len(indices[i])):
            try:
                if indices[i][j] == 2:
                    y.append(1) # maybe should do a 0??
                elif indices[i][j] == 3:
                    y.append(0)
                else:
                    y.append(1)
            except:
                pdb.set_trace()
        if not (i % 50):
            print "Featurized %d" % i

    model = SGDClassifier(loss=args.loss, penalty="l2")
    model.fit(X,y)
    return model, X, y

# note: kind of specific to CNN+Dailymail
def get_data(args):
    sources = open(args.srcfile, 'r')
    targets = open(args.targfile, 'r')
    sources_all = sources.read().split("\n")[:-1]
    targets_all = targets.read().split("\n")[:-1]
    features = []
    for i in xrange(len(sources_all)):
        features.append(featurize(sources_all[i], targets_all[i])) # append here because we will predict for each sentence at a time
        if not (i % 10000) and i:
            print "Featurized %d" % i
    return features

def prune(args, model, data):

    def clean(doc):
        sents = [clean_string(s) for s in doc.split("</s>")[1:]]
        doc = ' </s> '.join(sents)
        return doc

    split = args.split
    val_flag = 0
    srcfile = open(args.outfile+'-src.txt', 'w+')
    targfile = open(args.outfile+'-targ.txt', 'w+')
    docfile = open(args.srcfile, 'r')
    documents = docfile.read().split("\n")[:-1]
    accepted = []
    accepted_t = []
    ndocs = 0.
    for i in xrange(len(data)):
        try:
            if not data[i]:
                continue
            predictions = model.predict(data[i])
            if sum(predictions) > args.thresh: # if there is a nonzero entry
                accepted.append(clean(documents[i]))
                accepted_t.append(predictions)
                ndocs += 1
            if not (i % 10000):
                print "Pruned %d" % i
        except Exception as e:
            print "There was an error"
            pdb.set_trace()

    print "Writing training data..."
    for i,(doc,seq) in enumerate(zip(accepted, accepted_t)):
        try:
            srcfile.write(doc+'\n')
            targfile.write(np.array_str(seq, 1000)[1:-1]+'\n') # fkin numpy and their max line widths
            if i > split*ndocs and not val_flag:
                print "Writing validation data..."
                srcfile.close()
                targfile.close()
                srcfile = open(args.outfile+'-valid-src.txt', 'w+')
                targfile = open(args.outfile+'-valid-targ.txt', 'w+')
                val_flag = 1
        except Exception as e:
            print "Fukin shit"
            pdb.set_trace()

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcfile', help="Path to source training data. ", type = str, default="data/cnn-src.txt")
    parser.add_argument('--targfile', help="Path to target training data. ", type = str, default="data/cnn-targ.txt")
    parser.add_argument('--pickle', help="Load from pickle or no", type =int, default=1)
    parser.add_argument('--pickle_file', help="Path to pretrained pickle file to read or write. ", type = str, default='cnn.pkl')
    parser.add_argument('--thresh', help="Threshold for number of overlap sentences to be accepted. ", type=int, default=0)
    parser.add_argument('--outfile', help="Prefix of the output file names. ", type=str, default="extract")
    parser.add_argument('--split', help="Fraction of the data in train/valid. ", type=float, default=.8)
    parser.add_argument('--loss', help="Type of loss to use for classifier. ", type=str, default="hinge")
    parser.add_argument('--train', help="Path to classifier training docs. ", type=str, default="valid-src.txt")
    parser.add_argument('--train_t', help="Path to classifier training summaries. ", type=str, default="valid-targ.txt")
    parser.add_argument('--train_i', help="Path to classifier training classes. ", type=str, default="valid-indices.txt")
    args = parser.parse_args(arguments)
    if args.pickle:
        pickle_dump = pickle.load(open(args.pickle_file, 'rb'))
        model = pickle_dump[0]
        data = pickle_dump[1]
        print "Loaded data from pickle"
    else:
        print "Training model..."
        model, train, train_t = tune(args)
        print "Gathering data..."
        data = get_data(args)
        pickle.dump([model, data, train, train_t], open(args.pickle_file, 'wb'))
    print "Pruning dataset..."
    prune(args, model, data)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
