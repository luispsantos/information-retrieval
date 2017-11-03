from exercise_1 import *
from exercise_3 import *
import os

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

def calculate_metrics(retrieved_sent_indexes, relevant_sent_indexes, epsilon=0.01):
    """
    given sorted lists of retrieved and relevant sentences indexes
    calculate common measures used in Information Retrieval
    """
    
    #calculate list intersection
    intersection_len = len(set(retrieved_sent_indexes).intersection(relevant_sent_indexes))
    
    recall = intersection_len / len(relevant_sent_indexes)
    precision = intersection_len / len(retrieved_sent_indexes)
    
    #we add epsilon to the denominator to avoid division by 0
    f1_score = 2*((recall * precision)/(recall + precision + epsilon))
    
    avg_precision = 0
    relevant_seen = 0
    for i, sent_id in enumerate(retrieved_sent_indexes):
        if sent_id in relevant_sent_indexes:
            relevant_seen += 1
            avg_precision += relevant_seen / (i+1)
    
    return precision, recall, f1_score, avg_precision

def main_ex2(use_bm25=False, n_grams=1, use_pos_tags=False, pos_regex=''):

    #global variables
    source_text_path = 'temario/textos-fonte'
    summaries_path = 'temario/sumarios'
    retrieve_files = 25
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
    doc_sentences, doc_sent_words = sent_word_tokenize_advanced(doc_text, doc_language, n_grams, use_pos_tags, pos_regex)
 
    #retrieve vocabulary and word-index mappings
    vocab, vocab_size, word_to_index, index_to_word = create_vocab(doc_sent_words)

    #idf vector
    if use_bm25:
        idf_vector = calculate_idf_bm25(doc_sent_words, vocab, vocab_size, word_to_index)
    else:
        idf_vector = calculate_idf(doc_sent_words, vocab, vocab_size, word_to_index) 

    #now we iterate over all documents and perform ranked retrieval for each one
    mean_avg_precision = 0
    for doc_id in range(num_docs):

        #sentence tf matrix and document tf vector
        doc_words = [word for sentence in doc_sent_words[doc_id] for word in sentence]
        
        if use_bm25:
            sent_tf_matrix, doc_tf_vector = calculate_tf_bm25(doc_sent_words[doc_id], doc_words, vocab, vocab_size, word_to_index)
        else:
            sent_tf_matrix, doc_tf_vector = calculate_tf(doc_sent_words[doc_id], doc_words, vocab, vocab_size, word_to_index)

        #calculate tf-idf
        sent_tf_idf_matrix, doc_tf_idf_vector = calculate_tf_idf(sent_tf_matrix, doc_tf_vector, idf_vector)

        #ranked retrieval
        ranked_sentences, retrieved_sent_indexes = rank_sentences(sent_tf_idf_matrix, doc_tf_idf_vector, num_sentences_retrieved)

        #tokenize summary file into sentences
        summary_sentences = sent_tokenize(summaries_text[doc_id])

        relevant_sent_indexes = [doc_sentences[doc_id].index(summary_sentence) for summary_sentence in summary_sentences]
        relevant_sent_indexes.sort()
        
        precision, recall, f1_score, avg_precision = calculate_metrics(retrieved_sent_indexes, relevant_sent_indexes)
        mean_avg_precision += avg_precision
        
        print('Doc name:', doc_name[doc_id][3:], 'Pr:', round(precision, 3), 'Re:', round(recall, 3), 'F1 score:', round(f1_score, 3))
    
    print('Mean average precision:', mean_avg_precision / num_docs)



if __name__ == '__main__':
    main_ex2(use_bm25=False)

