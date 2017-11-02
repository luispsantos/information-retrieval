from proj import *

def read_docs(dir_path, encoding='latin-1'):
    """
    reads all documents from dir_path and produces a list of documents' name
    and a list of documents' content
    """

    doc_name = []
    doc_text = []
    
    arr_files = os.listdir(dir_path)
    for filename in arr_files:
        if filename.endswith(".txt"):
            doc_name.append(filename)
            
            f = open(os.path.join(dir_path, filename), 'r', encoding=encoding)
            doc_text.append(f.read())
            f.close()
            
    return doc_name, doc_text


def main():

    #global variables
    dir_path = 'temario/test'
    num_sentences_retrieved = 5
    doc_language = 'portuguese'

    #read source texts from a directory
    doc_name, doc_text = read_docs(dir_path)

    #apply sentence and word tokenization
    doc_sentences, doc_sent_words = sent_word_tokenize(doc_text, doc_language)

    #retrieve vocabulary and word-index mappings
    vocab, vocab_size, word_to_index, index_to_word = create_vocab(doc_sent_words)

    #idf vector
    idf_vector = calculate_idf(doc_sent_words, vocab, vocab_size, word_to_index) 

    #now we iterate over all documents and perform ranked retrieval for each one
    for sentences, sent_words in zip(doc_sentences, doc_sent_words):

        #sentence tf matrix and document tf vector
        doc_words = [word for sentence in sent_words for word in sentence]
        sent_tf_matrix, doc_tf_vector = calculate_tf(sent_words, doc_words, vocab, vocab_size, word_to_index)

        #calculate tf-idf
        sent_tf_idf_matrix, doc_tf_idf_vector = calculate_tf_idf(sent_tf_matrix, doc_tf_vector, idf_vector)

        #ranked retrieval
        ranked_sentences, relevant_sent_indexes = rank_sentences(sent_tf_idf_matrix, doc_tf_idf_vector, num_sentences_retrieved)

        print()
        for sent_index in relevant_sent_indexes:
            print('Score:', ranked_sentences[sent_index], '-', sentences[sent_index])

    #globals().update(locals())

if __name__ == '__main__':
    main()

