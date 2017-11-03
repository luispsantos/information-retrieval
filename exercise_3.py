from exercise_1 import *
from exercise_2 import *

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

def create_vocab_with_bigrams(doc_sent_words):
    bigrams = []
    for doc in doc_sent_words:
        for sentences in doc:
            for i in range(len(sentences)):
                if i < len(sentences) - 1:
                    bigrams.append((sentences[i], sentences[i + 1]))

    vocab, vocab_size, word_to_index, index_to_word = create_vocab(doc_sent_words)
    vocab+=bigrams
    vocab_size+=len(bigrams)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for i, word in enumerate(vocab)}

    return vocab, vocab_size, word_to_index, index_to_word


if __name__ == '__main__':
    print('Ex 2 without BM25')
    main_ex2(use_bm25=False)
    
    print('\nEx 2 with BM25')
    main_ex2(use_bm25=True)
