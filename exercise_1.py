import numpy as np
from numpy.linalg import norm
from math import log
import string

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def text_preprocess(text, stop_words, stemmer):
    """
    pre-process text according to standard techniques
    used in IR for text pre processing, a good description
    of the techniques can be found in Chris D. Manning's book
    on Information Retrieval at chapter 2
    """

    #convert to lowercase
    text = text.lower()

    #break up text into words
    word_list = word_tokenize(text)

    #remove stopwords and punctuation
    word_list = [word for word in word_list if word not in string.punctuation and word not in stop_words]

    #apply a stemmer
    word_list = [stemmer.stem(word) for word in word_list]

    return word_list

def sent_word_tokenize(doc_text, doc_language='english'):
    """
    function which takes a list of document contents in string format and for each
    document splits it in sentences using sent_tokenize
    Furthermore for each sentence we apply text pre-processing and retrieve a word list

    returns:
    -doc_sentences is a 2 dimensional list, where the first dimension represents the
    documents and the second one the sentences for each document in string format
    -doc_sent_words is a 3 dimensional list, where the first dimension represents the
    documents, the second one the sentences for each document and the third one
    the words/terms for each sentence in string format
    """

    stop_words = stopwords.words(doc_language)
    stemmer = SnowballStemmer(doc_language)

    doc_sentences = []
    doc_sent_words = []

    for doc in doc_text:
        sentences = sent_tokenize(doc)
        sentence_words = [text_preprocess(sent, stop_words, stemmer) for sent in sentences]

        doc_sentences.append(sentences)
        doc_sent_words.append(sentence_words)

    return doc_sentences, doc_sent_words

def create_vocab(doc_sent_words, use_bigrams):
    """
    creates a vocabulary for all documents which consists
    of a list of all unique words, for each word we create
    a mapping that allows us to retrieve the index for a
    specific word, or the word for a specific index

    returns:
    -vocab is a list of strings which contains the words
    that appear in any of the documents
    it is important to mention these words might not all be
    dictionary words because of the stemming process applied
    on the text preprocessing step
    -vocab size is the number of unique words in all documents
    after applying text preprocessing
    -word_to_index is a dictionary that maps words to their
    corresponding indexes in the vocabulary
    -index_to_word is a dictionary that maps indexes to their
    corresponding words in the vocabulary
    """

    flat_list = []
    
    #create a list of words contained in all documents
    for doc in doc_sent_words:
        for sentences in doc:
            for word in sentences:
                flat_list.append(word)

        if (use_bigrams):
            bigrams = []
            for sentences in doc:
                for i in range(len(sentences)):
                    if i < len(sentences) - 1:
                        bigrams.append((sentences[i], sentences[i + 1]))
            vocab = list(set(flat_list + bigrams))
        else:
            # retrieve a list of all unique words
            vocab = list(set(flat_list))

    #retrieve a list of all unique words
    #vocab = list(set(flat_list))
    vocab_size = len(vocab)
    
    #create mapping between words and indexes
    word_to_index = {word:i for i, word in enumerate(vocab)}
    index_to_word = {i:word for i, word in enumerate(vocab)}
    
    return vocab, vocab_size, word_to_index, index_to_word
    

def calculate_idf(doc_sent_words, vocab, vocab_size, word_to_index):
    
    #idf is a (vocab_size,) vector where each position
    #represents the inverse document frequency of that term
    #idf is a global measure, as such it doesn't depend on a single sentence
    idf_vector = np.zeros((vocab_size))

    for word in vocab:
        df = 0
        num_sentences = 0

        #calculate document frequency(in truth it's actually sentence frequency)
        #iterate all sentences and count in how many a word occurs
        for doc in doc_sent_words:
            for sentences in doc:
                num_sentences += 1
                if word in sentences:
                    df += 1


        idf = log(num_sentences / df)
        idf_vector[word_to_index[word]] = idf

    return idf_vector
     
def calculate_tf(sentence_words, doc_words, vocab, vocab_size, word_to_index):
    

    #calculate term frequency for sentences
    #sent_tf_matrix is a (num_sentences, vocab_size) matrix with normalized tf scores
    #where each line represents a sentences and each column represents a term
    num_sentences = len(sentence_words)
    sent_tf_matrix = np.zeros((num_sentences, vocab_size))

    for sent_id, sentence in enumerate(sentence_words):
        for word in sentence:
            #print(word, word_to_index[word])
            sent_tf_matrix[sent_id, word_to_index[word]] += 1

    #normalizing tf scores
    sent_tf_matrix = sent_tf_matrix / sent_tf_matrix.max(axis=1, keepdims=True)

    #calculate term frequency for documents
    #very similar to the term frequency for sentences
    #doc_tf_vector is a (vocab_size,) vector with normalized tf scores
    doc_tf_vector = np.zeros((vocab_size))
    
    for word in doc_words:
        doc_tf_vector[word_to_index[word]] += 1
    
    #normalizing tf scores
    doc_tf_vector = doc_tf_vector / doc_tf_vector.max()

    return sent_tf_matrix, doc_tf_vector

def calculate_tf_idf(sent_tf_matrix, doc_tf_vector, idf_vector):
    #now to obtain tf-idf scores we multiply the tf_matrix for sentences
    #with the idf_vector, the same idf vector is used for all sentences
    #when multiplying which makes sense since it's a global measure

    #numpy broadcasting takes care of converting idf_vector to a matrix
    sent_tf_idf_matrix = sent_tf_matrix * idf_vector

    #tf-idf for document, the same idf vector used for sentences is used here
    doc_tf_idf_vector = doc_tf_vector * idf_vector

    return sent_tf_idf_matrix, doc_tf_idf_vector

def rank_sentences(sent_tf_idf_matrix, doc_tf_idf_vector, num_sentences_retrieved=3):

    #to simplify calculating cosine similarity we should first
    #normalize the tf_idf vectors
    sent_tf_idf_matrix /= norm(sent_tf_idf_matrix, axis=1, keepdims=True)
    doc_tf_idf_vector /= norm(doc_tf_idf_vector)

    #now calculating the scores becomes simply a matrix multiplication
    #ranked_sentences is a (num_sentences,) vector where each position
    #represents the score assigned to that sentence, higher values are better
    ranked_sentences = sent_tf_idf_matrix.dot(doc_tf_idf_vector)

    #retrieve indexes of highest scoring sentences
    relevant_sent_indexes = ranked_sentences.argsort()[-num_sentences_retrieved:][::-1]

    #sort list so that sentences appear as in the order they show up in the document
    relevant_sent_indexes.sort()
   
    return ranked_sentences, relevant_sent_indexes 


def main_ex1():

    #global variables
    doc_name = 'computer-science.txt'
    num_sentences_retrieved = 3
    doc_language = 'english'

    #document to apply automatic summarization to
    f = open(doc_name)
    doc_text = f.read()
    f.close()

    #apply sentence and word tokenization
    #our functions are general enough to work with a collection of documents so here
    #we pass doc_text as a list of one document
    doc_sentences, doc_sent_words = sent_word_tokenize([doc_text], doc_language)

    sent_words = doc_sent_words[0]
    sentences = doc_sentences[0]

    #retrieve vocabulary and word-index mappings
    vocab, vocab_size, word_to_index, index_to_word = create_vocab(doc_sent_words, False)

    #idf vector
    idf_vector = calculate_idf(doc_sent_words, vocab, vocab_size, word_to_index) 

    #sentence tf matrix and document tf vector
    doc_words = [word for sentence in sent_words for word in sentence]
    sent_tf_matrix, doc_tf_vector = calculate_tf(sent_words, doc_words, vocab, vocab_size, word_to_index)

    #calculate tf-idf
    sent_tf_idf_matrix, doc_tf_idf_vector = calculate_tf_idf(sent_tf_matrix, doc_tf_vector, idf_vector)

    #ranked retrieval
    ranked_sentences, relevant_sent_indexes = rank_sentences(sent_tf_idf_matrix, doc_tf_idf_vector, num_sentences_retrieved)

    print('Showing', num_sentences_retrieved, 'most relevant sentences of', doc_name)
    for sent_index in relevant_sent_indexes:
        print('Score:', ranked_sentences[sent_index], '-', sentences[sent_index])

if __name__ == '__main__':
    main_ex1()

