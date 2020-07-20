# ###############################################
# ###############################################

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time


# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    #input_dim = input_tensor.shape[0]
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations




# Functions for deep neural network structure construction
def multilayer_perceptron(x, input_dim, hidden_dims, output_dim, act_list):
    #input_dim = x.shape[0]

    datat = x
    dimt  = input_dim
    for i, dim in enumerate(hidden_dims):
        layer_name = 'layer' + str(i+1)
        datat      = nn_layer(datat, dimt, dim, layer_name, act_list[i])
        dimt       = dim

    out_layer = nn_layer(datat, dimt, output_dim, 'output_layer', act_list[i+1])
    return out_layer

# Functions for deep neural network training
def train(I, O, 
    n_hidden=[2000,1000,1000], 
    act_list=[tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid], 
    location='./log', 
    training_epochs=300, 
    batch_size=1000, 
    LR=0.001, 
    traintestsplit=0.01, 
    LRdecay=0):

    num_total = I.shape[1]                              # number of total samples
    num_val   = int(num_total * traintestsplit)         # number of validation samples
    num_train = num_total - num_val                     # number of training samples
    n_input   = I.shape[0]                              # input size
    n_output  = O.shape[0]                              # output size
    I_train   = np.transpose(I[:, 0:num_train])         # training data
    O_train   = np.transpose(O[:, 0:num_train])         # training label
    I_val     = np.transpose(I[:, num_train:num_total]) # validation data
    O_val     = np.transpose(O[:, num_train:num_total]) # validation label

    input         = tf.placeholder("float", [None, n_input])
    output        = tf.placeholder("float", [None, n_output])
    is_train      = tf.placeholder("bool")
    learning_rate = tf.placeholder(tf.float32, shape=[])
    total_batch   = int(num_total / batch_size)
    print('Train: %d ' % num_train, 'validation: %d ' % num_val)
    print('Input dim: %d ' % n_input, ', output dim: %d ' % n_output)

    input_keep_prob  = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)

    pred      = multilayer_perceptron(input, n_input, n_hidden, n_output, act_list)
    cost      = tf.reduce_mean(tf.square(pred - output))    # cost function: MSE

    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost) # training algorithms: RMSprop
    init      = tf.global_variables_initializer()
    saver     = tf.train.Saver()

    MSETime   = np.zeros((training_epochs,3))
    LRt = LR
    with tf.Session() as sess:
        sess.run(init)
        merged       = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(location, sess.graph)
        start_time   = time.time()
        for epoch in range(training_epochs):
            run_metadata = tf.RunMetadata()
            if LRdecay == 1:
                LRt = LR/(epoch+1)
            for i in range(total_batch):
                idx = np.random.randint(num_train,size=batch_size)

                _, c, summary= sess.run([optimizer, cost, merged], feed_dict={input: I_train[idx, :],
                                                                output: O_train[idx, :],
                                                                input_keep_prob: 1,
                                                                hidden_keep_prob: 1,
                                                                learning_rate: LRt,
                                                                is_train: True})
            MSETime[epoch, 0] = c
            MSETime[epoch, 1] = sess.run(cost, feed_dict={input: I_val,
                                                          output: O_val,
                                                          input_keep_prob: 1,
                                                          hidden_keep_prob: 1,
                                                          is_train: False})
            MSETime[epoch, 2] = time.time() - start_time
            if epoch%(10) == 0:        #epoch%(int(training_epochs/10)) == 0:
                print('Epoch:%d, '%epoch, 'train:%0.2f%%, '%(c*100), 'validation:%0.2f%%.'%(MSETime[epoch, 1]*100))
            train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
            train_writer.add_summary(summary, epoch)

        print("Training time: %0.2f s" % (time.time() - start_time))
        sio.savemat('MSETime_%d_%d_%d' % (n_output, batch_size, LR*10000), {'train': MSETime[:,0], 'validation': MSETime[:,1], 'time': MSETime[:,2]})
        saver.save(sess, location+"/model")
        writer = tf.summary.FileWriter(location)
        writer.add_graph(sess.graph)
        writer.close()
        train_writer.close()
    return 0


# Functions for deep neural network testing
def test(I, n_input, n_output, 
    n_hidden=[2000,1000,1000], 
    act_list=[tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid], 
    model_location='./log', 
    save_name='Prediction', 
    binary=0):

    print('Begin testing...')

    tf.reset_default_graph()
    input                = tf.placeholder("float", [None, n_input])
    is_train         = tf.placeholder("bool")
    input_keep_prob  = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)
    pred             = multilayer_perceptron(input, n_input, n_hidden, n_output, act_list)
    saver            = tf.train.Saver()
    #print('test input dim: %d ' % n_input, ', output dim: %d ' % n_output)
    with tf.Session() as sess:
        saver.restore(sess, model_location+"/model")
        start_time  = time.time()
        w_pred      = sess.run(pred, feed_dict={input: np.transpose(I),
                                                input_keep_prob: 1,
                                                hidden_keep_prob: 1,
                                                is_train: False})
        testtime    = time.time() - start_time
        # print("testing time: %0.2f s" % testtime)
        if binary==1:
            w_pred[w_pred >= 0.5] = 1
            w_pred[w_pred <  0.5] = 0
        w_pred = np.transpose(w_pred)
        sio.savemat(save_name, {'pred': w_pred})
    return testtime
