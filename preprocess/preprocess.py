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
import pickle
from collections import defaultdict

class Indexer:
    def __init__(self, symbols = ["*blank*","<unk>","<d>","</d>"]):
        self.vocab = defaultdict(int) # special dictionary type for counting
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOD = symbols[2]
        self.EOD = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOD: 3, self.EOD: 4}

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1
            
    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]

    def convert_full(self, w):
        return self.word2ind[w] 

    def convert_sequence(self, ls):
        return ([self.convert(l) for l in ls], [self.convert_full(l) for l in ls])

    def clean(self, s):
        s = s.replace(self.PAD, "")
#        s = s.replace(self.UNK, "") # why is this commented out?
        s = s.replace(self.BOD, "")
        s = s.replace(self.EOD, "")
        return s
        
    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k.encode('utf-8'), v
        out.close()

    def prune_vocab(self, k):
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def full_vocab(self):
        self.word2ind = {}
        self.ind2word = {}
        for k, v in self.d.iteritems():
            self.word2ind[k] = v
            self.ind2word[v] = k
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse=True) # could skip to the remaining V - k?
        for word,_ in vocab_list:
            if word not in self.word2ind:
                ind = len(self.word2ind)+1
                self.word2ind[word] = ind
                self.ind2word[ind] = word

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.decode("utf-8").strip().split()
            self.d[v] = int(k)

def build_embeds(fname, outfile, words):
    def load_bin_vec(fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        return word_vecs

    word_vecs = load_bin_vec(fname, words)
    embeds = np.random.uniform(-0.25, 0.25, (len(words), len(word_vecs.values()[0])))
    embeds[0] = 0
    for word, vec in word_vecs.iteritems():
        embeds[words[word]-1] = vec

    f = h5py.File(outfile+".hdf5", "w")
    f["word_vecs"] = np.array(embeds) # sources is now binary where 1 = not pad, 0 = pad
    f.close()

def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))

def get_data(args):
    src_indexer = Indexer(["<blank>","<unk>","<d>","</d>"])
    word_indexer = Indexer(["<blank>","<unk>","<s>", "</s>"])
    word_indexer.add_w([src_indexer.PAD, src_indexer.UNK, src_indexer.BOD, src_indexer.EOD])
    def make_vocab(srcfile, targetfile, seqlength, max_sent_l=0):
        num_docs = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip()) # do these do anything?
            src = src_orig.strip().split("</s>") # splits the doc into sentences
            if len(src) > seqlength or len(src) < 1:
                continue
            num_docs += 1

            # only words are in source; target is now binary
            for sent in src:
                sent = word_indexer.clean(sent)
                if len(sent) == 0:
                    continue
                words = sent.split() 
                max_sent_l = max(len(words)+2, max_sent_l)
                for word in words:
                    word_indexer.vocab[word] += 1                                        
                
        return max_sent_l, num_docs
                
    def convert(srcfile, targetfile, targettxtfile, batchsize, seqlength, outfile, num_docs,
                max_sent_l, max_doc_l=0, unkfilter=0):
        
        newseqlength = seqlength + 2 #add 2 for EOS and BOS; length in sents of the longest document
        targets = np.zeros((num_docs, newseqlength), dtype=int) # the target sequence
        target_output = np.zeros((num_docs, newseqlength), dtype=int) # next word to predict
        sources = np.zeros((num_docs, newseqlength), dtype=int) # input split into sentences
        source_lengths = np.zeros((num_docs,), dtype=int) # lengths of each document
        target_lengths = np.zeros((num_docs,), dtype=int) # lengths of each target sequence
        sources_word = np.zeros((num_docs, newseqlength, max_sent_l), dtype=int) # input by word
        sources_word_full = np.zeros((num_docs, newseqlength, max_sent_l), dtype=int) # input by word, no unknowns
        dropped = 0
        doc_id = 0
        srcfile_prune = open("pruned-src-"+outfile, "w+") # file to write pruned dataset to
        targfile_prune = open("pruned-targ-"+outfile, "w+") # file to write pruned dataset to
        targettxt_f = open(targettxtfile, 'r')
        targettxts = targettxt_f.read().split('\n')[:-1]
        for i, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ = [0] + targ_orig.strip().split() + [0] # targ and src should be same length
            src = [src_indexer.BOD] + src_orig.strip().split("</s>") + [src_indexer.EOD]
            max_doc_l = max(len(targ), len(src), max_doc_l)
            if len(src) > newseqlength or len(src) < 3:
                dropped += 1
                continue                   
            targ = pad(targ, newseqlength+1, 0)
            targ = np.array(targ, dtype=int)
            #targ += 1 # 1-indexing for lua

            srcfile_prune.write(src_orig+"\n") # write sentence if it survived pruning
            targfile_prune.write(targettxts[i]+"\n")
            src = pad(src, newseqlength, src_indexer.PAD)
            src_word = []
            src_word_full = []
            for sent in src:
                sent = word_indexer.clean(sent)
                if len(sent) == 0:
                    pad_word, pad_word_full = word_indexer.convert_sequence([word_indexer.PAD] * max_sent_l)
                    src_word.append(pad_word) # fill with padding 
                    src_word_full.append(pad_word_full) # fill with padding 
                    continue
                words = [word_indexer.BOD] + sent.split() + [word_indexer.EOD]
                if len(words) > max_sent_l:
                    words = words[:max_sent_l]
                    words[-1] = word_indexer.EOD
                word_idx, word_idx_full = word_indexer.convert_sequence(pad(words, max_sent_l, word_indexer.PAD))
                src_word.append(word_idx)
                src_word_full.append(word_idx_full)
            src = [1 if x != src_indexer.PAD else 0 for x in src] # 0 if pad, 1 o.w.
            src = np.array(src, dtype=int)
            
            if unkfilter > 0:
                src_unks = float((src_word == 2).sum().sum()) 
                if unkfilter < 1: #unkfilter is a percentage if < 1
                    doc_length = float((src_word != 1).sum().sum())
                    src_unks = src_unks/doc_length
                if src_unks > unkfilter:
                    dropped += 1
                    continue
                
            targets[doc_id] = np.array(targ[:-1],dtype=int) # get all but the last pad
            target_output[doc_id] = np.array(targ[1:],dtype=int)                    
            sources[doc_id] = np.array(src, dtype=int)
            source_lengths[doc_id] = (sources[doc_id] == 1).sum()            
            target_lengths[doc_id] = source_lengths[doc_id]
            sources_word[doc_id] = np.array(src_word, dtype=int)
            sources_word_full[doc_id] = np.array(src_word_full, dtype=int)
            doc_id += 1
            if not (doc_id % 100000):
                print("{}/{} sentences processed".format(doc_id, num_docs))

        srcfile_prune.close()

        #break up batches based on source lengths
        # get source_lengths into a particular shape then sort by length
        source_lengths = source_lengths[:doc_id]
        source_sort = np.argsort(source_lengths) 
        sources = sources[source_sort]
        targets = targets[source_sort]
        target_output = target_output[source_sort]
        target_l = target_lengths[source_sort] # define new arrays to be the lengths
        source_l = source_lengths[source_sort]

        curr_l = 0
        l_location = [] #idx where sent length changes
        
        for j,i in enumerate(source_sort): # iterate over the indices of the sorted sentences
            if source_lengths[i] > curr_l:
                curr_l = source_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources)) # l_location is array where sentence length changes happen

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        nonzeros = [] # number of non padding entries in the entire batch
        batch_l = [] # batch lengths (number of sentences in the batch)
        batch_w = [] # batch widths (length of the sentences in the batch)
        target_l_max = []
        for i in range(len(l_location)-1): # iterate over all the different document lengths
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1): # iterate over batch_idx
            batch_l.append(batch_idx[i+1] - batch_idx[i])            
            batch_w.append(source_l[batch_idx[i]-1])
            nonzeros.append((sources[batch_idx[i]-1:batch_idx[i+1]-1] == 1).sum().sum())
            target_l_max.append(max(target_l[batch_idx[i]-1:batch_idx[i+1]-1]))
        # NOTE: actual batching is done in data.lua

        # Write output
        f = h5py.File(outfile, "w")

        # NOTE: not changing the names of things so don't need to change data.lua
        f["source"] = sources # sources is now binary where 1 = not pad, 0 = pad
        f["target"] = target_output #used to be target
        f["target_output"] = targets #target_output
        f["target_l"] = np.array(target_l_max, dtype=int)
        f["target_l_all"] = target_l        
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_w"] = np.array(batch_w, dtype=int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["target_nonzeros"] = np.array(nonzeros, dtype=int)
        f["source_size"] = np.array([2]) #np.array([len(src_indexer.d)])
        f["target_size"] = np.array([2]) #np.array([len(target_indexer.d)])
        del sources, targets, target_output
        sources_word = sources_word[source_sort]
        sources_word_full = sources_word_full[source_sort]
        f["source_order"] = source_sort
        f["source_char"] = sources_word
        f["source_char_full"] = sources_word_full
        del sources_word
        f["char_size"] = np.array([len(word_indexer.d)])
        print("Saved {} documents (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()                
        return max_sent_l

    print("First pass through data to get vocab...")
    max_sentence_l, num_docs_train = make_vocab(args.srcfile, args.targetfile,
                                             args.seqlength, 0)
    print("Number of sentences in training: {}".format(num_docs_train))
    max_sentence_l, num_docs_valid = make_vocab(args.srcvalfile, args.targetvalfile,
                                             args.seqlength, max_sentence_l)
    print("Number of doc in valid: {}".format(num_docs_valid))    
    print("Max sentence length (before cutting): {}".format(max_sentence_l))
    max_sent_l = min(max_sentence_l, args.maxsentlength)
    print("Max sentence length (after cutting): {}".format(max_sent_l))

    #prune and write vocab
    word_indexer.prune_vocab(args.srcvocabsize)
    word_indexer.full_vocab() # get full vocabulary and reverse dictionaries for later
    word_indexer.write(args.outputfile + ".word.dict")
    with open(args.outputfile+".dicts.hdf5", "wb") as dict_file:
        pickle.dump((word_indexer.word2ind, word_indexer.ind2word), dict_file)
    print("Word vocab size: Original = {}, Pruned = {}".format(len(word_indexer.vocab), 
                                                          len(word_indexer.d)))

    if args.srcvocabfile != '':
        print('Building embeddings from ' + args.srcvocabfile)
        build_embeds(args.srcvocabfile, args.vocabfile, word_indexer.d)

    max_doc_l = 0
    max_doc_l = convert(args.srcvalfile, args.targetidxvalfile, args.targetvalfile, args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_docs_valid,
                         max_sent_l, max_doc_l, args.unkfilter)
    max_doc_l = convert(args.srcfile, args.targetidxfile, args.targetfile, args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_docs_train, max_sent_l,
                         max_doc_l, args.unkfilter)
    
    print("Max sent length (before dropping): {}".format(max_sent_l))    
    
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcvocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                " Rest are replaced with special UNK tokens.",
                                                type=int, default=50000)
    parser.add_argument('--targetvocabsize', help="Size of target vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                "Rest are replaced with special UNK tokens.",
                                                type=int, default=50000)
    parser.add_argument('--srcfile', help="Path to source training data, "
                                           "where each line represents a single "
                                           "source/target sequence.")
    parser.add_argument('--targetfile', help="Path to target training data, "
                                           "where each line represents a single "
                                           "source/target sequence.")
    parser.add_argument('--targetidxfile', help="Path to target index data, "
                                                "indicating whether a sentence "
                                                "is taken for the summary.")
    parser.add_argument('--srcvalfile', help="Path to source validation data.")
    parser.add_argument('--targetvalfile', help="Path to target txt validation data.")
    parser.add_argument('--targetidxvalfile', help="Path to target validation data.")
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=64)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str)
    parser.add_argument('--maxsentlength', help="For the character models, words are "
                                           "(if longer than maxwordlength) or zero-padded "
                                            "(if shorter) to maxwordlength", type=int, default=35)
    parser.add_argument('--srcvocabfile', help="If working with a preset vocab, "
                                          "then including this will ignore srcvocabsize and use the"
                                          "vocab provided here.",
                                          type = str, default='')
    parser.add_argument('--vocabfile', help="If working with a preset vocab, ", type=str, default='')
    parser.add_argument('--targetvocabfile', help="If working with a preset vocab, "
                                         "then including this will ignore targetvocabsize and "
                                         "use the vocab provided here.",
                                          type = str, default='')
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. "
                                       "Can be an absolute count limit (if > 1) "
                                       "or a proportional limit (0 < unkfilter < 1).",
                                          type = float, default = 0)
    args = parser.parse_args(arguments)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
