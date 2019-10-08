import tensorflow as tf
def cross_entropy_loss(inputs, true_w):

    # ==========================================================================
    #
    # inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    # true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].
    #
    # Write the code that calculate A = log(exp({u_o}^T v_c))

    A = tf.log(tf.exp(tf.reduce_sum(tf.multiply(inputs, true_w),axis=1)) + 1e-10)


    # And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, tf.transpose(true_w))), axis=1)+ 1e-10)

    # ==========================================================================

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weights: Weights for nce loss. Dimension is [Vocabulary, embeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimension is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    #getting Batch size,Vocab Size and number of -ve samples
    batch_size = labels.shape[0]
    vocab_size = biases.shape[0]
    sample_size = float(sample.shape[0])

    #Converting vectors to tensors
    labels = tf.reshape(labels, [batch_size])
    biases = tf.reshape(biases, [vocab_size, 1])
    samples = tf.convert_to_tensor(sample)
    unigram_prob = tf.reshape(tf.convert_to_tensor(unigram_prob), [vocab_size, 1])

    #Looking up weight embeddings for labels and -ve samples
    weight_labels = tf.gather(weights,labels)
    weight_samples = tf.gather(weights,samples)

    #Looking up bias embeddings for labels and -ve samples
    bias_labels = tf.gather(biases,labels)
    bias_samples = tf.gather(biases,samples)

    #Looking up unigram probabilities for labels and -ve samples
    uniprob_labels = tf.gather(unigram_prob,labels)
    uniprob_samples = tf.gather(unigram_prob,samples)

    #Taking dot product of weights (labels,samples) and inputs.
    temp_o = tf.reshape(tf.diag_part(tf.matmul(inputs, tf.transpose(weight_labels))), [batch_size, 1])
    temp_x = tf.matmul(weight_samples,tf.transpose(inputs))

    #Adding biases to temp_o and temp_x
    func_s_o = tf.add(bias_labels,temp_o)
    func_s_x = tf.add(bias_samples,temp_x)

    #Finding log of unigram probabilities for labels and samples
    log_prob_labels = tf.log(tf.multiply(sample_size,uniprob_labels) + 1e-10)
    log_prob_samples = tf.log(tf.multiply(sample_size,uniprob_samples) + 1e-10)

    #Finding log_sigmoids for true(labels) and negative samples.
    log_sigmoid_o = tf.log_sigmoid(tf.subtract(func_s_o,log_prob_labels))
    sigmoid_x = tf.sigmoid(tf.subtract(func_s_x, log_prob_samples))
    log_sigmoid_x = tf.log(tf.subtract(tf.ones(sigmoid_x.shape),sigmoid_x) + 1e-10)

    summation_log_sigmoid_x = tf.reduce_sum(log_sigmoid_x,axis=1)

    net_loss = tf.negative(tf.add(log_sigmoid_o,summation_log_sigmoid_x))
    return net_loss


