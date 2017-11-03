from exercise_1 import *

def read_docs(dir_path, retrieve_files=-1, encoding='latin-1'):
    """
    reads all documents from dir_path and produces a list of documents' name
    and a list of documents' content, the lists are ordered by file name
    retrieve_files is the number of files to retrieve, can be useful to retrieve
    just a subset of the files for faster completion time, the default -1 retrieves
    all files
    """

    doc_name = []
    doc_text = []
    
    file_name_list = [file_name for file_name in os.listdir(dir_path) if file_name.endswith('.txt')]
    file_name_list.sort()
    if retrieve_files != -1:
        file_name_list = file_name_list[:retrieve_files]

    for file_name in file_name_list:
        doc_name.append(file_name)
        
        f = open(os.path.join(dir_path, file_name), 'r', encoding=encoding)
        doc_text.append(f.read())
        f.close()
            
    return doc_name, doc_text


def main():

    #global variables
    source_text_path = 'temario/textos-fonte'
    summaries_path = 'temario/sumarios'
    retrieve_files = 2
    num_sentences_retrieved = 5
    doc_language = 'portuguese'

    #read source texts and summaries
    doc_name, doc_text = read_docs(source_text_path, retrieve_files)
    summaries_name, summaries_text = read_docs(summaries_path, retrieve_files)
    num_docs = len(doc_name)

    #check to see if each source_text has a corresponding summary
    assert len(doc_name) == len(summaries_name), 'Source texts and summaries directory\
should have the same number of text files'

    #make sure files names match if we remove the initial prefix
    #a simple for loop should work since files are ordered by file name
    for doc_file_name, summary_file_name in zip(doc_name, summaries_name):

        #ignoring prefix 'St-' for docs and 'Ext-' for summaries
        assert doc_file_name[3:] == summary_file_name[4:], 'Files ' + doc_file_name + ' and ' + summary_file_name + ' do not match'

    #apply sentence and word tokenization
    doc_sentences, doc_sent_words = sent_word_tokenize(doc_text, doc_language)

    #retrieve vocabulary and word-index mappings
    vocab, vocab_size, word_to_index, index_to_word = create_vocab(doc_sent_words)

    #idf vector
    idf_vector = calculate_idf(doc_sent_words, vocab, vocab_size, word_to_index) 
    globals().update(locals())

    #now we iterate over all documents and perform ranked retrieval for each one
    for doc_id in range(num_docs):

        #sentence tf matrix and document tf vector
        doc_words = [word for sentence in doc_sent_words[doc_id] for word in sentence]
        sent_tf_matrix, doc_tf_vector = calculate_tf(doc_sent_words[doc_id], doc_words, vocab, vocab_size, word_to_index)

        #calculate tf-idf
        sent_tf_idf_matrix, doc_tf_idf_vector = calculate_tf_idf(sent_tf_matrix, doc_tf_vector, idf_vector)

        #ranked retrieval
        ranked_sentences, relevant_sent_indexes = rank_sentences(sent_tf_idf_matrix, doc_tf_idf_vector, num_sentences_retrieved)

        #tokenize summary file into sentences and convert it to sentence indexes
        summary_sentences = sent_tokenize(summaries_text[doc_id])

        print([doc_sentences[doc_id].index(summary_sentence) for summary_sentence in summary_sentences])

        print()
        for sent_index in relevant_sent_indexes:
            print('Score:', ranked_sentences[sent_index], '-', doc_sentences[doc_id][sent_index])

    globals().update(locals())

if __name__ == '__main__':
    main()

