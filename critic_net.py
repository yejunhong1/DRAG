import numpy as np
import tensorflow as tf
import math
from config import *
import FC_DNN_P as DNN



class CriticNet:
    """ Critic Q value network of the DDPG algorithm 
    state_size:  size of the state vector/tensor
    action_size: size of the action vector/tensor
    TAU:         update rate of target network parameters
    write_sum:   key/interval for writing summary data to file
    """

    def __init__(self, state_size, action_size, TAU = 0.001, write_sum = 0, net_size_scale=1 ):
        self.hidden_dims = [int(n*net_size_scale) for n in CN_N_HIDDENS]
        self.activations = CN_ACTS      # the last one is the linear activation function
        self.input_size  = state_size + action_size
        self.counter     = 0
        self.write_sum   = write_sum
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            self.input         = tf.placeholder( "float32", [None, self.input_size], name="state_action" )
            self.t_input       = tf.placeholder( "float32", [None, self.input_size], name="Target_state_action" )
            self.learning_rate = tf.placeholder( "float32", shape=[],  name="learning_rate" )
            self.Q_obj         = tf.placeholder( "float32", [None, 1], name="q_obj" ) #supervisor

            self.Q_value,   self.parameters    = DNN.multilayer_perceptron( self.input,   self.input_size, self.hidden_dims, 1, self.activations )
            self.t_Q_value, self.t_parameters  = DNN.multilayer_perceptron( self.t_input, self.input_size, self.hidden_dims, 1, self.activations, "Target_" )
            self.parameters_name = ["W1","B1","W2","B2","W3","B3"]
            #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1)+tf.nn.l2_loss(self.W2)+ tf.nn.l2_loss(self.W2_action) + tf.nn.l2_loss(self.W3)+tf.nn.l2_loss(self.B1)+tf.nn.l2_loss(self.B2)+tf.nn.l2_loss(self.B3) 
            #self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.W2,2))+ 0.0001*tf.reduce_sum(tf.pow(self.B2,2))             
            #self.cost      = tf.pow( self.Q_value - self.Q_obj,2 )/AC_BATCH_SIZE #+ self.l2_regularizer_loss#/tf.to_float(tf.shape(self.Q_obj)[0])
            self.cost      = tf.reduce_mean( tf.square( self.Q_value - self.Q_obj ) )
            self.optimizer = tf.train.AdamOptimizer( self.learning_rate ).minimize( self.cost )
            
            #action gradient to be used in actor network:
            self.grad_to_input    = tf.gradients( self.Q_value, self.input )
            self.act_grad_v       = self.grad_to_input[0][:,state_size:]
            self.action_gradients = [ self.act_grad_v / tf.to_float( tf.shape(self.act_grad_v[0])[0] ) ] #this is just divided by batch size
            
            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )
            
            #To make sure critic and target have same parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_parameters[0].assign( self.parameters[0] ),
                self.t_parameters[1].assign( self.parameters[1] ),
                self.t_parameters[2].assign( self.parameters[2] ),
                self.t_parameters[3].assign( self.parameters[3] ),
                self.t_parameters[4].assign( self.parameters[4] ),
                self.t_parameters[5].assign( self.parameters[5] ) ])

            self.update_target_critic_op = [
                self.t_parameters[0].assign( TAU*self.parameters[0] + (1-TAU)*self.t_parameters[0] ),
                self.t_parameters[1].assign( TAU*self.parameters[1] + (1-TAU)*self.t_parameters[1] ),
                self.t_parameters[2].assign( TAU*self.parameters[2] + (1-TAU)*self.t_parameters[2] ),
                self.t_parameters[3].assign( TAU*self.parameters[3] + (1-TAU)*self.t_parameters[3] ),
                self.t_parameters[4].assign( TAU*self.parameters[4] + (1-TAU)*self.t_parameters[4] ),
                self.t_parameters[5].assign( TAU*self.parameters[5] + (1-TAU)*self.t_parameters[5] ) ]

            #for i in range( 0, len(self.t_parameters) ):
            #    self.t_parameters[i].assign( self.parameters[i] )   # may need modify with .assign
            
            net_path   = "./model/cn"
            if self.write_sum > 0:
                self.saver = tf.train.Saver()
                self.saver.save(self.sess, net_path + "/net.ckpt")
                writer = tf.summary.FileWriter( net_path )
                writer.add_graph(self.sess.graph)
                writer.close()
                self.merged       = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter( net_path, self.sess.graph )
                self.run_metadata = tf.RunMetadata()

    def train_critic(self, state_batch, action_batch, tq_batch, learning_rate=0.0001 ):
        input = np.concatenate([state_batch, action_batch], axis=1)
        self.sess.run( self.optimizer, feed_dict={self.input: input, self.t_input: input, self.Q_obj: tq_batch, self.learning_rate: learning_rate} )       
        
        cerror = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            cerror, summary = self.sess.run( [self.cost, self.merged], feed_dict={self.input: input, self.t_input: input, self.Q_obj: tq_batch, self.learning_rate: learning_rate} )
            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
        self.counter += 1
        return cerror

    def evaluate_critic(self, state_1, action_1):
        input = np.concatenate([state_1, action_1], axis=1)
        return self.sess.run( self.Q_value, feed_dict={self.input: input} )

    def evaluate_target_critic(self, state, action ):
        input = np.concatenate([state, action], axis=1)
        return self.sess.run( self.t_Q_value, feed_dict={self.t_input: input} )
        
    def compute_grad_Q2a(self, state_t, action_t):
        input = np.concatenate([state_t, action_t], axis=1)
        return self.sess.run( self.action_gradients, feed_dict={self.input: input} )

    def update_target_critic(self):
        #print("target critic is updated.......")
        self.sess.run( self.update_target_critic_op )
        
        #for i in range( 0, len(self.t_parameters) ):
        #    self.t_parameters[i].assign( TAU*self.parameters[i] + (1-TAU)*self.t_parameters[i] )

    def close_all( self ):
        self.train_writer.close()
        DNN.write_var( self.sess, self.parameters, self.parameters_name, filename="critic_var" )

