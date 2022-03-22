from attention_model import *
import numpy as np
from lexicon import *

def print_topic_words(topic_word, vocab, top_k, rank=False):
    if rank:
        top_words = get_topic_words_rank(topic_word, vocab, top_k)
    else:
        top_words = get_topic_words(topic_word, vocab, top_k)
    for i, t in enumerate(top_words):
        print("top {} words in {}-th topic:".format(top_k, i))
        print(' '.join(t))
        print('')
        
        
def get_topic_words_rank(topic_word, vocab, top_k):
    vocab_dict = {key: i for i, key in enumerate(vocab)}

    n_topics = topic_word.shape[0]
    n_words = topic_word.shape[1]
    prob_w = np.zeros(n_words)
    for k in range(n_topics):
        for w in range(n_words):
            prob_w[w] += topic_word[k][w]

    top_words = []
    for topic_idx, topic in enumerate(topic_word): #for each topic
        topic_ranked = np.array([topic[i] / prob_w[i] for i in range(n_words)])
        temp_sum = sum(topic_ranked) 
        topic_ranked = [float(i)/temp_sum for i in topic_ranked]
        
        term_idx = np.argsort(topic_ranked)
        topKwords = []
        for j in np.flip(term_idx[-top_k:]):
            topKwords.append( (reverse_stem(vocab[j]), topic_ranked[j], topic_word[topic_idx][vocab_dict[vocab[j]]] ) )
        #print('{}, {}'.format(topic_idx, ' '.join(topKwords)))
        top_words.append(topKwords)
    return top_words

def get_topic_words(topic_word, vocab, top_k):
    top_words = []
    for i, t in enumerate(topic_word):
        term_idx = np.argsort(t)
        topKwords = []
        for j in np.flip(term_idx[-top_k:]):
            topKwords.append( reverse_stem(vocab[j]) )
        top_words.append(topKwords)
    return top_words

def load_GSM_beta(ckpt_path): #topic-word distribution
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        with tf.variable_scope('variational_topic_model'):
            model = VariationalTopicModel(vocab_size=2000,
                                latent_dim = 64,
                                num_topic = 25,
                                embedding_size=100,
                                dropout_keep_proba=0.8
                                )
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver.restore(sess, ckpt_path)
        topic_word = sess.run(model.beta, {model.is_training: False})
    print('topic-word distribution:', topic_word.shape)
    return topic_word

def load_GSM_theta(ckpt_path, x_bow): #doc-topic distribution
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        with tf.variable_scope('variational_topic_model'):
            model = VariationalTopicModel(vocab_size=2000,
                                latent_dim = 64,
                                num_topic = 25,
                                embedding_size=100,
                                dropout_keep_proba=0.8
                                )
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver.restore(sess, ckpt_path)
        
        feed_dict = {
            model.x: x_bow,
            model.is_training:False
        }        
            
        doc_topic, perpl = sess.run([model.topic, model.perp], feed_dict)

    print('doc-topic distribution:', doc_topic.shape)
    mean_perp = np.mean(perpl)
    print("perplexity {:g} ".format(mean_perp))
    return doc_topic        


def load_sDTM_beta(ckpt_path, wv_matrix): #topic-word distribution
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        model = Topical_attention_model(
            reduced_vocab_size=2000,
            num_topic=25,
            num_classes=2,
            pretrained_embed=wv_matrix,
            embedding_size=100,
            RNN_hidden_size=64,
            topic_hidden_size=64,
            dropout_keep_proba=0.8,
            max_word_num=500,
            threshold=0.1
        )
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver.restore(sess, ckpt_path)
        
        topic_word, topic_embed = sess.run([model.vtm.beta, model.vtm.topic_embed], {model.vtm.is_training: False})
        #topic_word = sess.run(model.beta, {model.is_training: False})
    print('topic-word distribution:', topic_word.shape)
    return topic_word

def load_sDTM_theta(ckpt_path, x_rnn, x_bow, y, wv_matrix): #doc-topic distribution
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        model = Topical_attention_model(
            reduced_vocab_size=2000,
            num_topic=25,
            num_classes=2,
            pretrained_embed=wv_matrix,
            embedding_size=100,
            RNN_hidden_size=64,
            topic_hidden_size=64,
            dropout_keep_proba=0.8,
            max_word_num=500,
            threshold=0.1
        )
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver.restore(sess, ckpt_path)
        
        feed_dict = {
            model.input_x: x_rnn,
            model.vtm.x: x_bow,
            model.input_y: y,
            model.is_training:False,
            model.vtm.is_training: False
        }
            
        preds, labels, pred_prob, perpl, weight, doc_topic = sess.run([model.predict, model.label, model.predict_prob, model.vtm.perp, model.weights, model.vtm.topic], feed_dict)

    print('doc-topic distribution:', doc_topic.shape)
    mean_perp = np.mean(perpl)
    print("perplexity {:g} ".format(mean_perp))
    return doc_topic  