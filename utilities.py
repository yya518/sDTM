import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sys

def read_vector(filename):
    wordVectors = []
    vocab = []
    fileObject = open(filename, 'r')
    for i, line in enumerate(fileObject):
        if i==0 or i==1: # first line is a number (vocab size)
            continue
        line = line.strip()
        word = line.split()[0]
        vocab.append(word)
        wv_i = []
        for j, vecVal in enumerate(line.split()[1:]):
            wv_i.append(float(vecVal))
        wordVectors.append(wv_i)
    wordVectors = np.asarray(wordVectors)
    vocab_dict = dict(zip(vocab, range(1, len(vocab)+1))) # no 0 id; saved for padding
    print("Vectors read from: "+filename)
    return wordVectors, vocab_dict, vocab

'''
def read_rnn_data(path, vocab_dict):
    myFile= open(path, "rU")
    labels = []
    docIDs = []
    for i, aRow in enumerate(myFile):
        x = [text.split(',') for text in aRow.split()]
        ids = []
        for w in x[1]:
            if w in vocab_dict:
                ids.append(vocab_dict[w])
            else:
                ids.append(vocab_dict["[UNK]"])
        if len(ids)>=5:
            labels.append(class_dict[x[0][0]])
            docIDs.append(ids)
    myFile.close()
    num_docs = len(labels)
    print(num_docs, "docs in total")
    y = np.zeros((num_docs, num_classes), dtype=np.int32)
    x = np.zeros((num_docs, MAX_NUM_WORD), dtype=np.int32)
    for i in range(num_docs):
        y[i][labels[i]] = 1
        if len(docIDs[i])>MAX_NUM_WORD:
            x[i, :] = docIDs[i][:MAX_NUM_WORD]
        else:
            x[i, :len(docIDs[i])] = docIDs[i]
    return x, y, num_docs
'''

def decrease_vocab(path, vocab, target_vocab_size=2000):
    _vocab = [w for w in vocab if w not in stopWords]
    myFile= open(path, "rU")
    docs = []
    for i, aRow in enumerate(myFile):
        x = aRow.split()[1]
        docs.append(x)
    myFile.close()
    vectorizer = TfidfVectorizer(vocabulary = _vocab)
    X = vectorizer.fit_transform(docs)
    word_importance = np.sum(X, axis = 0) # shape: [1, vocab_size], a numpy matrix!
    sorted_vocab_idx = np.squeeze(np.asarray(np.argsort(word_importance), dtype=np.int32)) # shape: [vocab_size, ], a numpy array
    vocab_idx_wanted = np.flip(sorted_vocab_idx)[:target_vocab_size] # decending order, int
    new_vocab = [_vocab[i] for i in vocab_idx_wanted]
    new_vocab_dict = dict(zip(new_vocab, range(target_vocab_size)))
    with open("topic_model_vocab.txt", 'w') as w_f:
        w_f.write('\n'.join(new_vocab))
    return new_vocab, new_vocab_dict

def read_topic_data(path, reduced_vocab):
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    myFile= open(path, "rU" )
    d = []
    for i, aRow in enumerate(myFile):
        x = aRow.split()[1]
        d.append(x)
    myFile.close()
    counts = count_vect.fit_transform(d).toarray()
    doc_word_sum = np.sum(counts, axis=1)
    valid_idx = doc_word_sum>5
    x = counts[valid_idx]
    return x, valid_idx

def read_topical_atten_data(path, vocab_dict, reduced_vocab, class_dict):
    num_classes = len(class_dict.keys())

    myFile= open(path, "rU")
    labels = []
    docIDs = []
    docs = []
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    
    reduced_vocab_size = len(reduced_vocab)
    
    for i, aRow in enumerate(myFile):
        line = aRow.strip().split()
        ids = []
        NOT_EMPTY = False
        for w in line[1].split(','):
            if w in vocab_dict:
                ids.append(vocab_dict[w])
                if vocab_dict[w] < reduced_vocab_size: ## we want to make sure the doc is not empty, because we will restrict the vocab to the reduced_vocab. make sure vocab is within the 2000 list 
                    NOT_EMPTY = True
        if len(ids)>0:#5 and NOT_EMPTY:
            labels.append(class_dict[line[0]])
            docIDs.append(ids)
            docs.append(line[1])
    myFile.close()
    num_docs = len(labels)
    print(num_docs, "docs in total")
    y = np.zeros((num_docs, num_classes), dtype=np.int32)
    x = np.zeros((num_docs, MAX_NUM_WORD), dtype=np.int32)
    for i in range(num_docs):
        y[i][labels[i]] = 1
        if len(docIDs[i])>MAX_NUM_WORD:
            x[i, :] = docIDs[i][:MAX_NUM_WORD]
        else:
            x[i, :len(docIDs[i])] = docIDs[i]
    counts = count_vect.fit_transform(docs).toarray()
    return x, y, counts, num_docs    

def load_yelp():
    path = './'
    return load_dataset(path + 'data/yelp/embedding/yelpVec.txt', 
                         path + 'data/yelp/topic_model_vocab.txt',
                         path + "data/yelp/full-processed.tab",
                        path + "data/yelp/full-processed.tab",
                         path + "data/yelp/full-processed.tab",
                         CLASSES= ['0','1'])

def load_data(dataset_name):
    path = './'
    if dataset_name == 'yelp':
        return load_dataset(path + 'data/yelp/embedding/yelpVec.txt', 
                         path + 'data/yelp/topic_model_vocab.txt',
                         path + "data/yelp/train-processed.tab",
                        path + "data/yelp/test-processed.tab",
                         path + "data/yelp/test-processed.tab",
                         CLASSES= ['0','1'])
    
    if dataset_name == 'yelp_helpful':
        return load_dataset(path + 'data/yelp/embedding/yelpVec.txt', 
                         path + 'data/yelp/topic_model_vocab.txt',
                         path + "data/yelp/train-review.helpful.tab",
                        path + "data/yelp/valid-review.helpful.tab",
                         path + "data/yelp/test-review.helpful.tab",
                         CLASSES= ['0','1'])
    
    if dataset_name == '20news':
        return load_dataset(path + 'data/20news/embedding/20newsVec.txt', 
                            path + 'data/20news/topic_model_vocab.txt',
                            path + "data/20news/train-processed.tab",
                            path + "data/20news/valid-processed.tab",
                            path + "data/20news/test-processed.tab",
                            CLASSES = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
                                       'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                                       'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
                                       'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 
                                       'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 
                                       'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
                           )
    
    if dataset_name == 'stackexchange':
        return load_dataset(path + 'data/stackexchange/embedding/stackexchangeVec.txt', 
                            path + 'data/stackexchange/topic_model_vocab.txt',
                            path + "data/stackexchange/train-processed.tab",
                            path + "data/stackexchange/test-processed.tab",
                            path + "data/stackexchange/test-processed.tab",
                            CLASSES=['anime', 'english', 'gaming', 'history', 'outdoors', 'politics','travel', 'writers']
                           )
    
def load_dataset(embedding_path, vocab_path, train_file, valid_file, test_file, CLASSES):
    num_classes = len(CLASSES)
    class_dict = dict(zip(CLASSES, range(num_classes)))    

    # load my data
    wv_matrix, vocab_dict, old_vocab = read_vector(embedding_path)
    old_vocab_size = len(old_vocab)
    print("vocab size: ", old_vocab_size)
    vocab_size=2000
    print("reduced vocab size: ", vocab_size)
    new_vocab = []
    with open(vocab_path) as r_f:
        for line in r_f:
            new_vocab.append(line.strip())
    print('start loading data')
    train_x_rnn, train_y, train_x_bow, num_train_docs = read_topical_atten_data(train_file, vocab_dict, new_vocab, class_dict)
    valid_x_rnn, valid_y, valid_x_bow, num_valid_docs = read_topical_atten_data(valid_file, vocab_dict, new_vocab, class_dict)
    test_x_rnn, test_y, test_x_bow, num_test_docs = read_topical_atten_data(test_file, vocab_dict, new_vocab, class_dict)
    
    print(train_x_bow.shape, test_x_bow.shape)

    return train_x_rnn, train_y, train_x_bow, num_train_docs, valid_x_rnn, valid_y, valid_x_bow, num_valid_docs, test_x_rnn, test_y, test_x_bow, num_test_docs, new_vocab, wv_matrix, vocab_dict


def load_document(docs, wv_matrix, new_vocab, vocab_dict): ## given a list of documents in string 'a b c d', get required data format
    ids = []
    docIDs = []
    count_vect = CountVectorizer(vocabulary=new_vocab)
    for doc in docs:
        for w in doc.split(' '):
            if w in vocab_dict:
                ids.append(vocab_dict[w])
        docIDs.append(ids)
    num_docs = len(docs)
    x = np.zeros((num_docs, MAX_NUM_WORD), dtype=np.int32)
    for i in range(num_docs):
        if len(docIDs[i])>MAX_NUM_WORD:
            x[i, :] = docIDs[i][:MAX_NUM_WORD]
        else:
            x[i, :len(docIDs[i])] = docIDs[i]
    counts = count_vect.fit_transform(docs).toarray()
    return x, counts, num_docs
    
stopWords = set(stopwords.words('english'))

MAX_NUM_WORD = 500
EMBEDDING_SIZE = 100


from scipy import spatial
def cosine_similarity(u, v):
    return 1 - spatial.distance.cosine(u, v)



