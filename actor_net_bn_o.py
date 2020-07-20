import tensorflow as tf
import math
from batch_norm import *
import numpy as np
from config import *
import FC_DNN_P as DNN

N_HIDDEN_1 = AN_N_HIDDENS[0]
N_HIDDEN_2 = AN_N_HIDDENS[1]


class ActorNet_bn:
    """ Actor Network with batch normalization of DDPG Algorithm 
    state_size:  size of the state vector/tensor
    action_size: size of the action vector/tensor
    TAU:         update rate of target network parameters
    write_sum:   key/interval for writing summary data to file
    """
    def __init__(self, state_size, action_size, TAU = 0.001, write_sum = 0):
        tf.reset_default_graph()
        self.counter     = 0
        self.write_sum   = write_sum        # if to write the summary file
        ntanh = lambda x, name=[]: ( tf.nn.tanh( x ) + 1 )/2
        self.activations = [tf.nn.softplus, tf.nn.relu, ntanh]
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #actor network model parameters:
            self.state       = tf.placeholder( "float32", [None, state_size], name="state" )
            self.is_training = tf.placeholder( tf.bool,   [],                 name="is_training" )
            
            with tf.name_scope("Layer_1"):
                with tf.name_scope('weights'):
                    self.W1   = tf.Variable( tf.random_uniform( [state_size,N_HIDDEN_1], -1/math.sqrt(state_size), 1/math.sqrt(state_size) ) )
                    DNN.variable_summaries(self.W1)
                with tf.name_scope('biases'):
                    self.B1   = tf.Variable( tf.random_uniform( [N_HIDDEN_1], -1/math.sqrt(state_size), 1/math.sqrt(state_size) ) )
                    DNN.variable_summaries(self.B1)
                with tf.name_scope('pre_bn'):
                    self.PBN1 = tf.matmul( self.state, self.W1 )
                    DNN.variable_summaries(self.PBN1)
                with tf.name_scope('pre_activation'):
                    self.BN1  = batch_norm( self.PBN1, N_HIDDEN_1, self.is_training, self.sess )
                    DNN.variable_summaries(self.BN1.bnorm)
                with tf.name_scope('activation'):
                    self.A1   = self.activations[0]( self.BN1.bnorm )     # + self.B1
                    DNN.variable_summaries(self.A1)
            
            with tf.name_scope("Layer_2"):
                with tf.name_scope('weights'):
                    self.W2   = tf.Variable( tf.random_uniform( [N_HIDDEN_1,N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1) ) )
                    DNN.variable_summaries(self.W2)
                with tf.name_scope('biases'):
                    self.B2   = tf.Variable( tf.random_uniform( [N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1) ) )
                    DNN.variable_summaries(self.B2)
                with tf.name_scope('pre_bn'):
                    self.PBN2 = tf.matmul(  self.A1, self.W2 )
                    DNN.variable_summaries(self.PBN2)
                with tf.name_scope('pre_activation'):
                    self.BN2  = batch_norm( self.PBN2, N_HIDDEN_2, self.is_training, self.sess )
                    DNN.variable_summaries(self.BN2.bnorm)
                with tf.name_scope('activation'):
                    self.A2   = self.activations[1]( self.BN2.bnorm )     # + self.B2
                    DNN.variable_summaries(self.A2)

            with tf.name_scope("Output_layer"):
                with tf.name_scope('weights'):
                    self.W3     = tf.Variable( tf.random_uniform( [N_HIDDEN_2,action_size], -0.003, 0.003 ) )
                    DNN.variable_summaries(self.W3)
                with tf.name_scope('biases'):
                    self.B3     = tf.Variable( tf.random_uniform( [action_size], -0.003, 0.003 ) )
                    DNN.variable_summaries(self.B3)
                with tf.name_scope('activation'):
                    self.action = self.activations[2]( tf.matmul( self.A2, self.W3 ) + self.B3 )
                    DNN.variable_summaries(self.action)

            

            #target actor network model parameters:
            self.t_state       = tf.placeholder( "float32", [None,state_size], name="t_state" )

            with tf.name_scope("T_Layer_1"):
                with tf.name_scope('weights'):
                    self.t_W1   = tf.Variable( tf.random_uniform( [state_size,N_HIDDEN_1],  -1/math.sqrt(state_size), 1/math.sqrt(state_size) ) )
                with tf.name_scope('biases'):
                    self.t_B1   = tf.Variable( tf.random_uniform( [N_HIDDEN_1], -1/math.sqrt(state_size), 1/math.sqrt(state_size) ) )
                with tf.name_scope('pre_bn'):
                    self.t_PBN1 = tf.matmul( self.t_state, self.t_W1 )
                with tf.name_scope('pre_activation'):
                    self.t_BN1  = batch_norm( self.t_PBN1, N_HIDDEN_1, self.is_training, self.sess, self.BN1 )
                with tf.name_scope('activation'):
                    self.t_A1   = self.activations[0]( self.t_BN1.bnorm )   # + self.t_B1

            with tf.name_scope("T_Layer_2"):
                with tf.name_scope('weights'):
                    self.t_W2   = tf.Variable( tf.random_uniform( [N_HIDDEN_1, N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1) ) )
                with tf.name_scope('biases'):
                    self.t_B2   = tf.Variable( tf.random_uniform( [N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1) ) )
                with tf.name_scope('pre_bn'):
                    self.t_PBN2 = tf.matmul(  self.t_A1, self.t_W2 )
                with tf.name_scope('pre_activation'):
                    self.t_BN2  = batch_norm( self.t_PBN2, N_HIDDEN_2, self.is_training, self.sess, self.BN2 )
                with tf.name_scope('activation'):
                    self.t_A2   = self.activations[1]( self.t_BN2.bnorm )   # + self.t_B2

            with tf.name_scope("T_Output_layer"):
                with tf.name_scope('weights'):
                    self.t_W3     = tf.Variable( tf.random_uniform( [N_HIDDEN_2, action_size], -0.003, 0.003 ) )
                with tf.name_scope('biases'):
                    self.t_B3     = tf.Variable( tf.random_uniform( [action_size], -0.003, 0.003 ) )
                with tf.name_scope('activation'):
                    self.t_action = self.activations[2]( tf.matmul( self.t_A2, self.t_W3 ) + self.t_B3 )


            self.learning_rate    = tf.placeholder( "float32", shape=[],           name="learning_rate" )
            self.obj_action       = tf.placeholder( "float32", [None,action_size], name="obj_action" )
            self.q_gradient_input = tf.placeholder( "float32", [None,action_size], name="q_gradient_input" ) #gets input from action_gradient computed in critic network file
            
            #cost of actor network:
            with tf.name_scope('cost'):
                self.cost             = tf.reduce_mean( tf.square( tf.round(self.action) - self.obj_action ) )
            self.actor_parameters     = [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3, self.BN1.scale, self.BN1.beta, self.BN2.scale, self.BN2.beta]
            self.parameters_gradients = tf.gradients( self.action, self.actor_parameters, -self.q_gradient_input )#/BATCH_SIZE) changed -self.q_gradient to -
            self.optimizer            = tf.train.AdamOptimizer( self.learning_rate ).apply_gradients( zip(self.parameters_gradients, self.actor_parameters) )  
            
            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )    
            
            #To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
				self.t_W1.assign( self.W1 ),
				self.t_B1.assign( self.B1 ),
				self.t_W2.assign( self.W2 ),
				self.t_B2.assign( self.B2 ),
				self.t_W3.assign( self.W3 ),
				self.t_B3.assign( self.B3 ) ] )

            self.update_target_actor_op = [
                self.t_W1.assign( TAU*self.W1 + (1-TAU)*self.t_W1 ),
                self.t_B1.assign( TAU*self.B1 + (1-TAU)*self.t_B1 ),  
                self.t_W2.assign( TAU*self.W2 + (1-TAU)*self.t_W2 ),
                self.t_B2.assign( TAU*self.B2 + (1-TAU)*self.t_B2 ),  
                self.t_W3.assign( TAU*self.W3 + (1-TAU)*self.t_W3 ),
                self.t_B3.assign( TAU*self.B3 + (1-TAU)*self.t_B3 ),
                self.t_BN1.updateTarget,
                self.t_BN2.updateTarget
            ]
            
            net_path   = "./model/an"
            self.saver = tf.train.Saver()
            self.saver.save(self.sess, net_path + "/net.ckpt")
            writer = tf.summary.FileWriter( net_path )
            writer.add_graph(self.sess.graph)
            writer.close()
            self.merged       = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter( net_path, self.sess.graph )
            self.run_metadata = tf.RunMetadata()
        
    def evaluate_actor( self, state ):
        return self.sess.run( self.action, feed_dict={self.state: state, self.is_training: False} )        
        
        
    def evaluate_target_actor( self, t_state ):
        return self.sess.run( self.t_action, feed_dict={ self.t_state: t_state, self.is_training: False } )
        
    def train_actor( self, state, obj_actioin, q_gradient_input, learning_rate=0.0001 ):
        self.sess.run( [ self.optimizer, self.BN1.train_mean, self.BN1.train_var, self.BN2.train_mean, self.BN2.train_var, 
            self.t_BN1.train_mean, self.t_BN1.train_var, self.t_BN2.train_mean, self.t_BN2.train_var], 
            feed_dict={ self.state: state, self.t_state: state, self.q_gradient_input: q_gradient_input, 
            self.learning_rate: learning_rate, self.is_training: True } )
        
        aerror = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            summary, aerror = self.sess.run( [self.merged, self.cost], 
            feed_dict={ self.state: state, self.t_state: state, self.obj_action: obj_actioin, 
            self.q_gradient_input: q_gradient_input, self.learning_rate: learning_rate, self.is_training: False } )
            
            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
        self.counter += 1
        return aerror

    def update_target_actor(self):
        self.sess.run( self.update_target_actor_op )    
        
    def close_all(self):
        self.train_writer.close()
        #DNN.write_var( self.sess, self.parameters, varname= self.parameters_name, filename="actor_var" )
