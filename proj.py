from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball
import numpy as np
from numpy.linalg import norm
from math import log
import string

doc_language = 'english'
#doc_language = 'portuguese'
num_sentences = 5

stop_words = stopwords.words(doc_language)

def text_preprocess(text, stop_words=stop_words):
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

    #remove stopwords
    word_list = [word for word in word_list if word not in stop_words]

    #remove stopwords and punctuation
    word_list = [word for word in word_list if word not in string.punctuation and word not in stop_words]

    stemmer = snowball.PortugueseStemmer(ignore_stopwords=False)

    word_list = list(map(stemmer.stem, word_list))

    return word_list

     
def calculate_tf_idf(sentence_tokens, doc_tokens, vocab, vocab_size, word_to_index):
    
    #idf is a (vocab_size,) vector where each position
    #represents the inverse document frequency of that term
    #idf is a global measure, as such it doesn't depend on a single sentence
    idf_vector = np.zeros((vocab_size))
    num_sentences = len(sentence_tokens)

    for word in vocab:

        #calculate document frequency(in truth it's actually sentence frequency)
        #iterate all sentences and count in how many a word occurs
        df = 0
        for sentence in sentence_tokens:
            if word in sentence:
                df += 1

        idf = log(num_sentences / df)
        idf_vector[word_to_index[word]] = idf

    #calculate term frequency for sentences
    #sent_tf_matrix is a (num_sentences, vocab_size) matrix with normalized tf scores
    #where each line represents a sentences and each column represents a term
    sent_tf_matrix = np.zeros((num_sentences, vocab_size))

    for sent_id, sentence in enumerate(sentence_tokens):
        for word in sentence:
            sent_tf_matrix[sent_id, word_to_index[word]] += 1

    #normalizing tf scores
    sent_tf_matrix = sent_tf_matrix / sent_tf_matrix.max(axis=1, keepdims=True)

    #calculate term frequency for documents
    #very similar to the term frequency for sentences
    #doc_tf_vector is a (vocab_size,) vector with normalized tf scores
    doc_tf_vector = np.zeros((vocab_size))

    for word in doc_tokens:
        doc_tf_vector[word_to_index[word]] += 1
    
    #normalizing tf scores
    doc_tf_vector = doc_tf_vector / doc_tf_vector.max()


    #now to obtain tf-idf scores we multiply the tf_matrix for sentences
    #with the idf_vector, the same idf vector is used for all sentences
    #when multiplying which makes sense since it's a global measure

    #numpy broadcasting takes care of converting idf_vector to a matrix
    sent_tf_idf_matrix = sent_tf_matrix * idf_vector

    #tf-idf for document, the same idf vector used for sentences is used here
    doc_tf_idf_vector = doc_tf_vector * idf_vector

    return sent_tf_idf_matrix, doc_tf_idf_vector

def rank_sentences(sent_tf_idf_matrix, doc_tf_idf_vector, num_sentences):

    #to simplify calculating cosine similarity we should first
    #normalize the tf_idf vectors
    sent_tf_idf_matrix /= norm(sent_tf_idf_matrix, axis=1, keepdims=True)
    doc_tf_idf_vector /= norm(doc_tf_idf_vector)

    #now calculating the scores becomes simply a matrix multiplication
    #ranked_sentences is a (num_sentences,) vector where each position
    #represents the score assigned to that sentence, higher values are better
    ranked_sentences = sent_tf_idf_matrix.dot(doc_tf_idf_vector)

    #retrieve indexes of highest scoring sentences
    relevant_sent_indexes = ranked_sentences.argsort()[-num_sentences:][::-1]

    return ranked_sentences, relevant_sent_indexes 

f = open('doc.txt')
doc = f.read()
f.close()


#break up into sentences
sentences = sent_tokenize(doc)

sentence_tokens = [text_preprocess(sent) for sent in sentences]
doc_tokens = text_preprocess(doc)

#vocab size - number of unique words in the document
vocab = list(set(doc_tokens))
vocab_size = len(vocab)

#create mapping between words and indexes
word_to_index = {word:i for i, word in enumerate(vocab)}
index_to_word = {i:word for i, word in enumerate(vocab)}

sent_tf_idf_matrix, doc_tf_idf_vector = calculate_tf_idf(sentence_tokens, doc_tokens, vocab, vocab_size, word_to_index)

ranked_sentences, relevant_sent_indexes = rank_sentences(sent_tf_idf_matrix, doc_tf_idf_vector, num_sentences)

print('Showing', num_sentences, 'most relevant sentences')
for sent_index in relevant_sent_indexes:
    print('Score:', ranked_sentences[sent_index], '-', sentences[sent_index])


