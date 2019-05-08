from __future__ import division
from __future__ import print_function
import nltk
import os
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
import glob
import math
import numpy
import random
import collections
import re
from nltk.corpus import stopwords

# 0=ham
# 1=spam

def perceptron_wo(iteration):
    print("\nIteration: ", iteration)
    etas = [0.001, 0.01, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    for eta in etas:
        all_words_weight = {}
        all_words_weight = stem("train/*/*", all_words_weight)
        for i in all_words_weight:
            all_words_weight[i] = 0.0
        for conv in range(iteration):
            all_words_weight = ptr("train/ham/*", 0.0, all_words_weight, eta)
            all_words_weight = ptr("train/spam/*", 1.0, all_words_weight, eta)
        test_file_count, wrong_decision = test(all_words_weight, "test-2/ham/*.txt", 0, 0, 0.0)
        test_file_count, wrong_decision = test(all_words_weight, "test-2/spam/*.txt", test_file_count, wrong_decision,
                                               1.0)
        print("Files interpreted correctly: ", (test_file_count - wrong_decision), "/", test_file_count)
        accuracy = (test_file_count - wrong_decision) / test_file_count
        print(eta, " : ", accuracy * 100)

def ptr(path, target, weights, eta):
    filepath = glob.glob(path + ".txt")
    for file in filepath:
        if not os.path.isfile(file):
            print("File path {} does not exist. Exiting...".format(file))
            sys.exit()

        doc_words = {}
        # print(file)
        file = file.rstrip(".txt")
        doc_words = stem(file, doc_words)
        # w_sum = weights["weight_zero"];
        w_sum = 1.0;
        for word in doc_words:
            if word not in weights:
                weights[word] = 0.0
            w_sum += (weights[word]*doc_words[word])
        if w_sum > 0:
            output = 1.0
        else:
            output = 0.0
        for word in doc_words:
            weights[word] += float(eta)*float(target - output)*float(doc_words[word])
    return weights

def test(weights, path, test_file_count, wrong_decision, target):
    filepath = glob.glob(path)
    test_file_count += len(filepath)
    for file in filepath:
        file = file.rstrip(".txt")
        doc_words = {}
        doc_words = stem(file, doc_words)
        # w_sum = weights["weight_zero"];
        w_sum = 1.0;
        for word in doc_words:
            if word not in weights:
                weights[word] = 0.0
            w_sum += (weights[word] * doc_words[word])
        if w_sum > 0:
            output = 1.0
        else:
            output = 0.0
        if (output != target) :
            wrong_decision += 1
    return test_file_count, wrong_decision

def stem(path, bag_of_words):
    filepath = glob.glob(path + ".txt")
    tokenizer = RegexpTokenizer("[a-zA-Z]+")
    stemmer = LancasterStemmer()
    stop_words = set(stopwords.words("english"))
    for file in filepath:
        if not os.path.isfile(file):
            print("File path {} does not exist. Exiting...".format(file))
            sys.exit()
        with open(file,'r') as fp:
            line = fp.read()
            tok = tokenizer.tokenize(line)
            tokens = []
            for t in tok:
                if t not in stop_words:
                    tokens.append(t)
            record_word_cnt(tokens, bag_of_words)
    return bag_of_words

def record_word_cnt(words, bag_of_words):
    for word in words:
        if word != '':
            if word.lower() in bag_of_words:
                bag_of_words[word.lower()] += 1
            else:
                bag_of_words[word.lower()] = 1

def pwo_m():
    iterations = [10, 150, 250]
    for iteration in iterations:
        perceptron_wo(iteration)