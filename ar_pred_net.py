import numpy as np
import tensorflow as tf
import math
from config import *
import FC_DNN_P as DNN


class ARPredNet:
    """ Arrival Rate prediction network for the SBS activation problem
    his_ar_size:  size of the history arrival rate vector/tensor
    pred_ar_size: size of the predicted arrival rate vector/tensor
    write_sum:    key for if to write summary data to file
    """
    def __init__(self, his_ar_size, pred_ar_size, write_sum = 0, net_size_scale=1 ):
        self.counter      = 0
        self.write_sum    = write_sum        
        self.his_ar_size  = his_ar_size
        self.pred_ar_size = pred_ar_size

        self.n_hidden     = [int(n*net_size_scale) for n in ARPN_N_HIDDENS]
        self.activations  = ARPN_ACTS

        self.g = tf.Graph()
        with self.g.as_default():   #
            self.sess = tf.InteractiveSession()

            self.his_ar_in     = tf.placeholder( "float32", [None, self.his_ar_size],  name="History_ar" )
            self.next_ar       = tf.placeholder( "float32", [None, self.pred_ar_size], name="Real_ar" )
            self.learning_rate = tf.placeholder( "float32", shape=[],                  name="LR" )

            self.pred_ar, self.arp_layer_parls = DNN.multilayer_perceptron( self.his_ar_in, 
                self.his_ar_size, self.n_hidden, self.pred_ar_size, self.activations )

            self.paras_name = ["W1","B1","W2","B2","W3","B3"]
            self.cost       = tf.reduce_mean( tf.square( self.pred_ar - self.next_ar) )
            self.optimizer  = tf.train.RMSPropOptimizer( self.learning_rate ).minimize( self.cost )

            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )

            net_path   = "./model/arpn"
            if self.write_sum > 0:
                self.saver = tf.train.Saver()
                self.saver.save(self.sess, net_path + "/net.ckpt")
                writer = tf.summary.FileWriter( net_path )
                writer.add_graph(self.sess.graph)
                writer.close()
                self.merged       = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter( net_path, self.sess.graph )
                self.run_metadata = tf.RunMetadata()
        
    def evaluate_ar_pred(self, his_ar):
        pred_ar = self.sess.run( self.pred_ar, feed_dict={ self.his_ar_in: his_ar } )
        pred_ar = np.clip(pred_ar, 0, 1)
        return pred_ar
        
        
    def train_ar_pred(self, his_ar_in, next_ar_in, learning_rate=0.0001):
        self.sess.run( self.optimizer, feed_dict={ self.his_ar_in: his_ar_in, 
            self.next_ar: next_ar_in, self.learning_rate: learning_rate } )
        
        error = self.sess.run(self.cost, feed_dict={ self.his_ar_in: his_ar_in, self.next_ar: next_ar_in } )
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            summary = self.sess.run([self.merged], 
                feed_dict={ self.his_ar_in: his_ar_in, self.next_ar: next_ar_in } )

            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
            #print("error: "+str(error)+", counter: "+str(self.counter))
            
        self.counter += 1
        return error

    def save_ar_pred_net(self):
        pass
        
    def close_all( self ):
        self.train_writer.close()
        #DNN.write_var( self.sess, self.arp_layer_parls, self.paras_name, filename="arp_var" )


