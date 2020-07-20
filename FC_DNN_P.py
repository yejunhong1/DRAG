from __future__ import print_function
import tensorflow as tf
import numpy as np
"""A extended version of FC_DNN.py with an output list of network parameters."""


# Don't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    #with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
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
        #with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('Wx_plus_b', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)

        layer_paras = [weights, biases]
        return activations, layer_paras

# Functions for deep neural network structure construction
def multilayer_perceptron(x, input_dim, hidden_dims, output_dim, act_list, pre=""):
    """Build a multi-layer full-connected network with the inputed configuration. 
    The return is the network output and a list of network parameters (weights and bias of layer 1 to n)
    """
    datat = x
    dimt  = input_dim
    net_paras =[]
    for i, dim in enumerate(hidden_dims):
        layer_name = pre+'Layer' + str(i+1)
        datat, layer_paras = nn_layer(datat, dimt, dim, layer_name, act_list[i])
        dimt       = dim
        net_paras.extend(layer_paras)

    out_layer, layer_paras = nn_layer(datat, dimt, output_dim, pre+'Output', act_list[i+1])
    net_paras.extend(layer_paras)
    return out_layer, net_paras


def write_var( sess, varls, varname= ["W1","B1","W2","B2","W3","B3"], filename="variables" ):
    pars    = sess.run( varls )
    thefile = open( filename + '.txt', 'w' )
    for i in range(len(varls)):
        thefile.write( "%s\n" % varname[i] )
        thefile.write( "%s\n" %  pars[i] )
        thefile.write( "\n"  )
    thefile.close()