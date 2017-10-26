#import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
from math import log

doc_language = 'english'
#doc_language = 'portuguese'

stop_words = stopwords.words(doc_language)

def text_preprocess(text, stop_words=stop_words):
    #convert to lowercase
    text = text.lower()

    #remove punctuation - anything other than words or whitespace
    text = re.sub(r'[^\w\s]', '', text)

    #break up textences into words
    word_list = word_tokenize(text)

    #remove stopwords
    word_list = [word for word in word_list if word not in stop_words]
    #print(word_list)
    return word_list


# we probably don't need an inverted index for exercise 1
"""
def build_inv_index(sentence_tokens, doc_tokens):
    inv_index = {}

    #index both the document and sentences
    #doc will have an id of 0
    docs = sentence_tokens.copy()
    docs.insert(0, doc_tokens)

    for i, tokens in enumerate(docs):
        word_count = count_words(tokens)
        max_count = max(word_count.values())

        for word in word_count:
            #store normalized term frequency in index
            norm_tf = word_count[word] / max_count
            word_freqs = (i, norm_tf)

            if word in inv_index:
                inv_index[word].append(word_freqs)
            else:
                inv_index[word] = [word_freqs]

    return inv_index
"""   
     
def calculate_tf_idf(sentence_tokens, vocab, vocab_size, word_to_index):
	
	#idf is a (vocab_size,) vector where each position
	#represents the inverse document frequency of that term
	#idf is a global measure, as such it doesn't depend on a single document
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

	#calculate term_frequency
	#tf is a (num_sentences, vocab_size) matrix with normalized tf scores
	#for each sentence
	tf_matrix = np.zeros((num_sentences, vocab_size))

	for sent_id, sentence in enumerate(sentence_tokens):
		for word in sentence:
			tf_matrix[sent_id, word_to_index[word]] += 1

	#normalizing tf scores
	tf_matrix = tf_matrix / tf_matrix.max(axis=1, keepdims=True)

	return tf_matrix



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

a=calculate_tf_idf(sentence_tokens, vocab, vocab_size, word_to_index)
