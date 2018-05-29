import numpy as np
# import fasttext
import tensorflow as tf

def get_embedding_weights(nb_words, tokenizer, w2v, embedding_dim=300):

    embedding_weights = np.zeros((nb_words, embedding_dim))
    for w, i in tokenizer.word_index.items():
        # if type(w2v) == fasttext.model.WordVectorModel or w in w2v:
        if w in w2v:
            embedding_weights[i, :] = w2v[w]
        else:
            embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, embedding_dim)

    return embedding_weights

def load_word2vec(fin, w2v=None):
    import tensorflow as tf
    if w2v is None:
        w2v = dict()

    fin.readline()
    for line in fin:
        data = line.split()
        word = data[0]
        vec = [float(v) for v in data[1:]]
        if word not in w2v:
            w2v[word] = vec
        else:
            # tf.logging.info('{} is already in the model!'.format(word))
            pass

    return w2v

def transitions_model(dim=[1000]):
    # a = tf.constant(input_a, dtype=tf.float32, shape=list(input_a.shape), name='a')
    # b = tf.constant(input_b, dtype=tf.float32, shape=list(input_b.shape), name='b')

    a = tf.placeholder(np.float32, [None] + dim, 'a')
    b = tf.placeholder(np.float32, [None] + dim, 'b')

    match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

    return a, b, p_ab, p_ba, p_aba

def get_max_cycle(start_idx, prob_ab, prob_ba):
    unlabeled_idx = prob_ab[start_idx].argmax()
    return_idx = prob_ba[unlabeled_idx].argmax()

    return [start_idx, unlabeled_idx, return_idx]
