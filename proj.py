#import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

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

def count_words(tokens):
    word_count = {}

    print(tokens)
    for word in tokens:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    return word_count

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
        

f = open('doc.txt')
doc = f.read()
f.close()


#break up into sentences
sentences = sent_tokenize(doc)

sentence_tokens = [text_preprocess(sent) for sent in sentences]
doc_tokens = text_preprocess(doc)

#vocab size - number of unique words in the document
vocab = set(doc_tokens)
vocab_size = len(vocab)

inv_index = build_inv_index(sentence_tokens, doc_tokens)
print(inv_index)

#calculate tf-idf vectors
#for i, word in enumerate(vocab):

