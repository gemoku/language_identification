# language_identification

###A Python implementation of a simple method to identify the language of a document

### Description

The underlying vector model is a variation of an n-gram frequency statistics model of [Marc Damashek (1995)] (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.7010&rep=rep1&type=pdf): 
- treat each text as a vector of the frequencies of its character n-grams;
- having a training set of documents in different languages, extract n-gram character frequency statistics for each document and store this statistics in a vector. We get a dictionary of the following form: *model[language][ngram][value]*. Thus, each document is represented as a vector whose components are the relative frequencies of its distinct constituent character n-grams;
- when a query comes in, extract ngram statistics for this query, build a query vector and compare it to each of the vectors in the trained model. In this program I use the 1-nearest neighbour algorithm and cosine similarity as distance metric to detect the closest vector and corresponding language.
- the highest scoring vector corresponds to the predicted language of the input text.

Despite the simplicity of the model, it achieves quite good results: accuracy above 96 % for 21 languages of Europarl corpus (see details below).

### How to run

I used the [Europarl corpus] (http://www.statmt.org/europarl/) for training. The total size of the decompressed text files is around 7.4G. In the *sample_data* folder you can find sample text chunks for each of the 21 languages - these are the first 10000 lines of the corpora text files. Use this data to reproduce the reported results of the experiment. The repository also contains a model (*10k_model*) trained on this data.
######NOTE_1: Europarl corpus contains 20 parallel sub-corpora: lang1-en, lang2-en and so on for the 20 non-English languages. After downloading you will get 20 non-English texts and 20 English texts. For the experiments I use all non-English texts and only one English - from the parallel French-English corpus (europarl-v7.fr-en.en). This is done to equalise the statistics for English in comparison with other languages.

1. First you will need to download the corpus using the following script:

```bashscript
./load_data.sh
```
This will create a folder *./data*, download all the corpora files and extract text files from the archives. The script also creates an *./experiment* folder, takes 10000 first lines of each of the downloaded text files (see NOTE_1) and stores them in this folder.

2. Train the model:
```bashscript
python language_identifier.py -tr experiment -m model -k 10 -ng 3 -ch 100
```
This will chunk document files from the *experiment* folder into pieces of size 100 (parameter *-ch*), extract trigram statistics (n-gram size is specified using the *-ng* option), train the model, evaluate it using stratified 10-fold cross-validation (parameter *-k* specifies the number of folds) and save the model into the file specified by the *-m* option. 

3. Query mode:
to query a file named *filename*:
```bashscript
python language_identifier.py -q filename -m model
```
---
to query several files in a specified folder named *foldername*:
```bashscript
python language_identifier.py -q foldername -m model
```
---
to make an interactive query:
```bashscript
python language_identifier.py -m model
```
###  Command line options

- -q  'Specify a file or a folder with files for prediction'
- -tr 'Train a model and evaluate it using the files from the specified folder'
- -m  'Specify a model file' (required)
- -k  'Specify the number of folds for cross-validation' 
- -ng 'Specify the size of n-grams'
- -ch 'Specify the chunksize'

### Known issues and caveats
1. I assume that languages under consideration are whitespace-delimited. The program should work for languages without spaces between words (like Japanese, Chinese) because I treat whitespace just like any other character and this information should also be captured when the n-gram statistics is extracted. However, the program was not tested on such languages.
2. I am not deleting punctuation signs and using only lowercasing for preprocessing - there are better ways to clean the text.
2. The chunksize I use for training and querying is 100 characters - this is enough to make quite an accurate prediction. I didn't perform any analysis on the accuarcy variance depending on this parameter, but the obvious thing to presume is: the shorter the text, the more difficult it is to detect it's language. See ["Language identification: the long and the short of the matter"](http://www.aclweb.org/anthology/N10-1027) for some valuable info on that account. 
3. The weigths for different n-grams are set according to the following reasoning: the longer the ngram, the more evidence it provides for determining whether the input belongs to some language (details in["Selecting and Weighting N-Grams to Identify 1100 1185 Languages"](http://www.cs.cmu.edu/~ralf/papers/brown-tsd13.pdf) by Ralf D. Brown). Here for simplicity I set them as
```python
[0.05 + x*0.1 for x in range(self.ngram_size)]
```
That is, we increase the weight depending on the ngram size. I guess normally you would use grid search or some other way to tune these weigths on a development set.
