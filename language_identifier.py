#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle
from collections import defaultdict
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import sys
import math


lang_list=["bg","cs","da","de","el","en","es","et",
               "fi","fr","hu","it","lt","lv","nl",
               "pl","pt","ro","sk","sl","sv"]

lang_to_idx = dict(zip(lang_list,xrange(len(lang_list))))
idx_to_lang = dict(zip(xrange(len(lang_list)),lang_list))

# a function for the interactive query mode
def repeat():
    query = raw_input('Type in your query: ')
    prediction = MD.query(query)
    print 'Predicted language: ', idx_to_lang[prediction[0]]

def iter_over_documents(directory, chunk_size =1000, mode = 'train'):
    '''
    A function to iterate over files in the data directory
    :param directory: top directory with the corpora files
    :param mode: either 'train' or 'query': if it's 'train', the function will generate pairs (label, text) with
     text length equal to chunk_size. If it's 'query' mode, text files in the specified directory will be read as one string,
     without chunking.
    '''

    for root, dirs, files in os.walk(directory):
        for fname in files:
            path = os.path.join(directory, fname)

            if mode == 'train':

                language = lang_to_idx[fname[-2:]]
                with open(path) as filein:
                    data = (' ' + filein.read().strip() + ' ').decode('utf8')
                    document_length = len(data)
                    num_chunks = document_length/chunk_size
                    for i in xrange(num_chunks):
                        text = data[chunk_size*i : (i+1)*chunk_size]
                        label = language
                        yield (text,label) # extract a (string, label) pair

            elif mode == 'query':

                with open(path) as filein:
                    data = (' ' + filein.read().strip() + ' ').decode('utf8')
                    yield path, data


def preprocessing(directory,chunk_size, mode = 'train'):

    samples_generator = iter_over_documents(directory,chunk_size, mode = mode)
    data = []
    labels = []

    # populate the data array
    for text,language in samples_generator:
        data.append(text)
        labels.append(language)

    return data,labels

def data_generator(idx_array, X_array, Y_array):
    for idx in idx_array:
        yield (X_array[idx], Y_array[idx])

class MODEL():
    '''
    A language identifier class. The retrieved statistics (features and corresponding values)
     is saved in a dictionary form (see function 'safe')

    '''

    def __init__(self, ngram_size = 3, n_fold = 5):
        self.ngram_size = ngram_size
        self.num_fold = n_fold
        self.accuracies = {}

    def get_ngrams(self, text):
        '''
        Extract n-grams from a string
        :param text: a chunk of text as a string.
        :return: a list of n-grams
        '''
        text = text.lower()
        ngrams=[]
        for a in xrange(self.ngram_size):
            for i in xrange(len(text)-a):
                ngrams.append(text[i:i+1+a])

        return ngrams

    def feature_vector_normalisation(self, vector):
        """
        Perform feature vector normalisation.
        :param vector: a feature vector, which captures the n-gram frequencies.
        The vector is in the dictionary format, which is why we use (feature, value) indexing.

        It makes sense to give different weigths to n-grams depending on their length. The reasoning is simple:
        "a trigram in comparison with a unigram or bigram is more informative in terms of a string's language - let's
          give it a bigger weight then".

        """

        feature_weights = [0.05 + x*0.1 for x in range(self.ngram_size)]

        mag = math.sqrt(sum(value**2 for feature,value in vector.iteritems()))

        for feature,value in vector.iteritems():
            vector[feature] = (vector[feature]/mag)*feature_weights[len(feature)-1]

    def train(self, samples_array, labels_array):

        '''
        Main training subroutine. Training and evaluation are done using stratified k-fold cross validation.
        First extract necessary statistics from (text, label) pairs.
        Then perform feature vector normalisation.
        Do the same for test pairs and evaluate.
        :param samples_array: data array, which holds text samples
               labels_array: data array, which holds text labels
               k: number of folds.
        :return: trains the model and prints evaluation results

        '''

        skf = StratifiedKFold(labels_array, self.num_fold)

        i = 1 # fold counter

        print '\nStart training...'

        for train_idx, test_idx in skf:
            sys.stdout.write('Training, %d fold ... ' %(i))

            model_template = defaultdict(lambda: defaultdict(float))
            train_samples_generator = data_generator(train_idx, samples_array, labels_array)
            test_samples_generator = data_generator(test_idx, samples_array,labels_array)

            for text, lang in train_samples_generator:
                feat_sequence = self.get_ngrams(text)

                for feature in feat_sequence:
                    model_template[lang][feature] += 1

            # perform feature vector normalisation
            for feature in model_template.itervalues():
                self.feature_vector_normalisation(feature)

            self.model = dict(model_template)

            sys.stdout.write('evaluating the model ... ')

            self.evaluate(test_samples_generator)

            sys.stdout.write(' mean accuracy for %d fold is %f \n' %(i, np.mean(self.accuracies.values())))

            i += 1

        print '\nAccuracies of the model with %d-fold cross-validation: ' %(self.num_fold)

        for label,value in self.accuracies.iteritems():
            print idx_to_lang[label], "---",value

    def evaluate(self, samples_generator):
        '''
        Evaluation subroutine.
        :param samples_generator: test_samples generator, yielding (text,label) pairs
        :return: updates self.accuracies
        '''
        total = defaultdict(int)
        correct = defaultdict(int)

        for text, label in samples_generator:
            total[label] +=1
            predicted_language, score = self.query(text)
            if predicted_language == label:
                correct[label] += 1

        for language in total.keys():
            self.accuracies[language] = float(correct[language])/total[language] * 100

    def query(self, text):
        '''
        Predict the language of a query text
        :param text: input text (as a string)
        :return: language for which the model gives the highest score
        '''
        query_ch_sequence = self.get_ngrams(text)
        query_model = defaultdict(float)
        for ch in query_ch_sequence:
            query_model[ch] += 1

        self.feature_vector_normalisation(query_model)

        model_scores = []
        for language, feature_vector in self.model.iteritems():
            score = 0.0
            for ch in query_model:
                score += query_model[ch] * feature_vector.get(ch, 0.0)
            model_scores.append((language, score))

        return max(model_scores, key=lambda item: item[1])

    def load(self, filename = 'new_model.pkl'):
        print 'Loading the model from <----- %s ' %(filename)
        with open(filename) as file_in:
            self.model = cPickle.load(file_in)

    def save(self, filename = 'new_model.pkl'):
        print '\nSaving the model into -----> %s ' %(filename)
        with open(filename, 'w') as file_out:
            cPickle.dump(self.model, file_out)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Simple method of language identification')
    parser.add_argument('-q', help = 'Specify a file or a folder with files for prediction', nargs = 1)
    parser.add_argument('-tr', help = 'Train a model and evaluate it '
                                       'using the files from the specified folder', nargs = 1)
    parser.add_argument('-m', help = 'Specify a model file', nargs=1, required=True)
    parser.add_argument('-k', help = 'Specify the number of folds for cross validation', nargs=1)
    parser.add_argument('-ng', help = 'Specify the size of n-grams', nargs=1)
    parser.add_argument('-ch', help = 'Specify the chunksize', nargs=1)
    opts = parser.parse_args()

    if opts.ch:
        chunk_size = int(opts.ch[0])
    else:
        chunk_size = 100

    if opts.ng:
        ngram_size = int(opts.ng[0])
    else:
        ngram_size = 3


    if opts.k:
        num_folds = int(opts.k[0])
    else:
        num_folds = 5

    if opts.tr:

        MD = MODEL(ngram_size=ngram_size, n_fold=num_folds)
        print 'Preprocessing the data ... '
        X, Y = preprocessing(opts.tr[0], chunk_size=chunk_size)

        MD.train(X,Y)
        print '\nTraining finished!'
        MD.save(opts.m[0])


    if opts.q:
        MD = MODEL()
        MD.load(opts.m[0])

        if os.path.isfile(opts.q[0]):
            print 'Querying the file: ', opts.q[0]
            with open(opts.q[0]) as f:
                text = (' ' + f.read().strip() + ' ').decode('utf8')
                prediction = MD.query(text)
                print 'Predicted language is: ', idx_to_lang[prediction[0]]

        elif os.path.isdir(opts.q[0]):
            print 'Querying files in the specified folder'
            data = iter_over_documents(opts.q[0], mode = 'query')
            for filename, text in data:
                prediction = MD.query(text)
                print 'Prediction for file %s' %(filename), 'is: ', idx_to_lang[prediction[0]]

    elif not opts.q and not opts.tr:
        MD = MODEL()
        print 'Loading model from the specified file ...'
        MD.load(opts.m[0])
        i = 0
        while(True):

            print 'Do you want to run the program ? (y/n) : '
            str = raw_input()

            if str == 'y' or str == 'Y':
                i += 1
                repeat()
            elif str == 'n' or str == 'N':
                print 'Exiting program'
                break
            else:
                print 'Please answer y/n only.\n'