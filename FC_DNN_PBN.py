from __future__ import print_function
import tensorflow as tf
import numpy as np
from batch_norm import *
"""A extended version of FC_DNN.py with an output list of network parameters and batch norm parameters.
To be finished...
"""


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
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            variable_summaries(preactivate)
        with tf.name_scope('activation'):    
            activations = act(preactivate)
            variable_summaries(activations)

        layer_paras = [weights, biases]
        return activations, layer_paras

def nn_bn_layer(input_tensor, input_dim, output_dim, layer_name, act, is_training, sess, parForTarget=None):
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
        with tf.name_scope('pre_bn'):
            PBN = tf.matmul(input_tensor, weights)
            variable_summaries( PBN )
        with tf.name_scope('pre_activation'):
            BN  = batch_norm( PBN, output_dim, is_training, sess, parForTarget=parForTarget )
        with tf.name_scope('activation'):
            activations = act(BN.bnorm)
            variable_summaries( activations )

        layer_paras = [weights]
        return activations, layer_paras, [BN]

# Functions for deep neural network structure construction
def multilayer_perceptron(x, input_dim, hidden_dims, output_dim, act_list, is_training, sess, parForTarget=None, pre="", start=0, endwithout=1):
    """Build a multi-layer full-connected network with the inputed configuration. 
    The return is the network output and a list of network parameters (weights and bias of layer 1 to n)
    """
    datat = x
    dimi  = input_dim
    net_paras = []
    bn_paras  = []
    for i, dimo in enumerate(hidden_dims):
        if parForTarget == None:
            parForTargett = None
        else:
            parForTargett = parForTarget[i]
        layer_name = pre+'Layer' + str(i+1+start)
        datat, layer_paras, layer_bn_paras = nn_bn_layer(datat, dimi, dimo, layer_name, act_list[i], is_training, sess, parForTarget=parForTargett)
        dimi = dimo
        net_paras.extend(layer_paras)
        bn_paras.extend(layer_bn_paras)
    end_layer = datat
    if endwithout==1:
        end_layer, layer_paras = nn_layer(datat, dimi, output_dim, pre+'Output', act_list[i+1])
        net_paras.extend(layer_paras)

    return end_layer, net_paras, bn_paras


def write_var( sess, varls, varname= ["W1","B1","W2","B2","W3","B3"], filename="variables" ):
    pars    = sess.run( varls )
    thefile = open( filename + '.txt', 'w' )
    for i in range(len(varls)):
        thefile.write( "%s\n" % varname[i] )
        thefile.write( "%s\n" %  pars[i] )
        thefile.write( "\n"  )
    thefile.close()


def init_summary(net_path, sess):
    saver = tf.train.Saver()
    saver.save(sess, net_path + "/net.ckpt")
    writer = tf.summary.FileWriter( net_path )
    writer.add_graph(sess.graph)
    writer.close()
    merged       = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( net_path, sess.graph )
    run_metadata = tf.RunMetadata()
    return saver, merged, train_writer, run_metadata