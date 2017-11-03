from exercise_1 import *
from exercise_2 import *

import re

def sent_word_tokenize_advanced(doc_text, doc_language='english', n_grams=1, use_pos_tags=False, pos_regex=''):
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
    
    if use_pos_tags:
        single_pos_pattern = re.compile('<.*?>')
        regex_pattern = re.compile(pos_regex)

    for doc in doc_text:
        sentences = sent_tokenize(doc)
        doc_sentences.append(sentences)
        
        sentence_words = []
        for sentence in sentences:
            words = text_preprocess(sentence, stop_words, stemmer)
        
            if n_grams > 1:
                n_gram_list = []
                for n in range(2, n_grams+1):
                    sentence_ngrams = zip(*[words[i:] for i in range(n)])
                    n_gram_list += sentence_ngrams
                    

            if use_pos_tags:
                pos_list = []
                
                pos_tags = ['<'+tag[1]+'>' for tag in nltk.pos_tag(words)]
                pos_str = ''.join(pos_tags)

                for match in re.finditer(regex_pattern, pos_str):
                    
                    #deal with empty matches
                    if match.start() == match.end():
                        continue
                    
                    #calculate start and end word indexes
                    start_word_index = len(re.findall(single_pos_pattern, pos_str[:match.start()]))
                    end_word_index = start_word_index + \
                                     len(re.findall(single_pos_pattern, pos_str[match.start():match.end()]))

                    #avoid matches of single words since they already occur once
                    #on the sentence
                    if start_word_index+1 == end_word_index:
                        continue
                    pos_list.append(tuple(words[start_word_index:end_word_index]))
                
                
            if n_grams > 1:
                words += n_gram_list
                
            if use_pos_tags:
                words += pos_list
            
            sentence_words.append(words)
        
        doc_sent_words.append(sentence_words)
        
    return doc_sentences, doc_sent_words



def calculate_tf_bm25(sentence_words, doc_words, vocab, vocab_size, word_to_index, k1=5, b=0.75):
    
    
    #sent_tf_matrix is a (num_sentences, vocab_size) matrix with normalized tf scores
    #where each line represents a sentences and each column represents a term
    num_sentences = len(sentence_words)
    sent_tf_matrix = np.zeros((num_sentences, vocab_size))

    for sent_id, sentence in enumerate(sentence_words):
        for word in sentence:
            sent_tf_matrix[sent_id, word_to_index[word]] += 1
                  
    word_count = sent_tf_matrix.sum(axis=1)
    avg_sent_length = word_count.mean()
    
    sent_tf_matrix = ((sent_tf_matrix * (k1+1)) /
                     (sent_tf_matrix + k1 * (1 - b + b *(num_sentences/ avg_sent_length))))
    
    doc_tf_vector = np.zeros((vocab_size))
    
    for word in doc_words: 
        doc_tf_vector[word_to_index[word]] += 1
                     
    doc_tf_vector = ((doc_tf_vector * (k1+1)) /
                     (doc_tf_vector + k1 * (1 - b + b *(num_sentences/ avg_sent_length))))
                         
    return sent_tf_matrix, doc_tf_vector

def calculate_idf_bm25(doc_sent_words, vocab, vocab_size, word_to_index):
    
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


        idf = log((num_sentences - df + 0.5) / (df + 0.5))
        idf_vector[word_to_index[word]] = idf

    return idf_vector


if __name__ == '__main__':
    print('Ex 2 without BM25')
    main_ex2(use_bm25=False)
    
    print('\nEx 2 with BM25')
    main_ex2(use_bm25=True)
    
    print('\nEx 2 with BM25 and bigrams')
    main_ex2(use_bm25=True, n_grams=2)
    
    print('\nEx 2 with BM25, bigrams and parts of speech regex')
    pos_regex = '(((<JJ>)*(<NN>)+<IN>)?(<JJ>)*(<NN>)*)+'
    main_ex2(use_bm25=True, n_grams=2, use_pos_tags=True, pos_regex=pos_regex)
