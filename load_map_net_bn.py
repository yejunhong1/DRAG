import numpy as np
import tensorflow as tf
import math
from config import *
import FC_DNN_PBN as DNN


class LoadMapNet_bn:
    """ Load mapping network for the SBS activation problem

    Input: arrival rate and action

    Output: estimated load

    ar_size: size of the history arrival rate vector/tensor

    action_size: number of the SBSs

    load_size: size of the predicted arrival rate vector/tensor

    write_sum: key for if to write summary data to file
    """
    def __init__(self, ar_size, action_size, load_size, write_sum = 0, net_size_scale=1 ):
        self.ar_size     = ar_size
        self.action_size = action_size
        self.ar_action_size = ar_size + action_size
        self.load_size   = load_size
        self.counter     = 0
        self.write_sum   = write_sum

        self.n_hidden    = [int(n*net_size_scale) for n in LMN_N_HIDDENS]
        self.activations = LMN_ACTS
        #print("n hiddens: "+str(self.n_hidden))
        self.g = tf.Graph()
        with self.g.as_default():   #
            self.sess = tf.InteractiveSession()

            self.ar_action_in  = tf.placeholder( "float32", [None, self.ar_action_size], name="AR_action" )
            self.real_load     = tf.placeholder( "float32", [None, self.load_size],      name="Real_load" )
            self.learning_rate = tf.placeholder( "float32", shape=[],                    name="LR" )
            self.is_training   = tf.placeholder( tf.bool,   [],                          name="is_training" )
              
            self.mapped_load, self.layer_paras, self.bn_paras = DNN.multilayer_perceptron( self.ar_action_in, 
                self.ar_action_size, self.n_hidden, self.load_size, self.activations, self.is_training, self.sess )

            self.lm_parameters_name = ["W1","B1","W2","B2","W3","B3"]
            self.cost      = tf.reduce_mean( tf.square( self.mapped_load - self.real_load) )
            self.optimizer = tf.train.RMSPropOptimizer( self.learning_rate ).minimize( self.cost )

            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )
            
            self.num_bn   = len(self.bn_paras)
            self.train_op = [ self.optimizer ]
            for i in range(self.num_bn):
                self.train_op.append( self.bn_paras[i].train_mean )
                self.train_op.append( self.bn_paras[i].train_var  )

            net_path   = "./model/lmn"
            if self.write_sum > 0:
                self.saver = tf.train.Saver()
                self.saver.save(self.sess, net_path + "/net.ckpt")
                writer = tf.summary.FileWriter( net_path )
                writer.add_graph(self.sess.graph)
                writer.close()
                self.merged       = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter( net_path, self.sess.graph )
                self.run_metadata = tf.RunMetadata()
        
    def evaluate_load_map(self, arrival_rate, action):
        ar_action_in = np.concatenate([arrival_rate, action], axis=1)
        mapped_load = self.sess.run( self.mapped_load, feed_dict={ self.ar_action_in:ar_action_in, self.is_training: False } )
        mapped_load = np.clip( mapped_load, 0, 0.998 )
        return mapped_load
        
        
    def train_load_map(self, ar_action_in, real_load_in, learning_rate=0.0001):
        self.sess.run( self.train_op, feed_dict={ self.ar_action_in: ar_action_in, 
            self.real_load: real_load_in, self.learning_rate: learning_rate, self.is_training: True } )
        
        error = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            error, summary = self.sess.run( [self.cost, self.merged], feed_dict={ self.ar_action_in: ar_action_in, 
                self.real_load: real_load_in, self.is_training: False } )

            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
            #print("error: "+str(error)+", counter: "+str(self.counter))
        self.counter += 1
        return error

    def save_load_map_net(self):
        pass
        
    def close_all( self ):
        self.train_writer.close()
        #DNN.write_var( self.sess, self.layer_paras, self.lm_parameters_name, filename="lm_var" )


