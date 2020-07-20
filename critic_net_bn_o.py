import tensorflow as tf
import math
from batch_norm import *
import numpy as np
from config import *
import FC_DNN_P as DNN

N_HIDDEN_1 = CN_N_HIDDENS[0]
N_HIDDEN_2 = CN_N_HIDDENS[1]

class CriticNet_bn:
    """ Critic Q value network with batch normalization of the DDPG algorithm 
    state_size:  size of the state vector/tensor
    action_size: size of the action vector/tensor
    TAU:         update rate of target network parameters
    write_sum:   key/interval for writing summary data to file
    """
    def __init__(self, state_size, action_size, TAU = 0.001, write_sum = 0):
        self.counter     = 0
        self.write_sum   = write_sum
        self.activations = [tf.nn.softplus, tf.nn.relu, lambda x, name=[]: x]

        tf.reset_default_graph()
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #Critic Q Network:
            self.state       = tf.placeholder( "float32", [None, state_size],  name="state"  )
            self.action      = tf.placeholder( "float32", [None, action_size], name="action"  ) 
            self.is_training = tf.placeholder( tf.bool,   [],                  name="is_training" )
            
            with tf.name_scope("Layer_1"):
                with tf.name_scope('weights'):
                    self.W1   = tf.Variable( tf.random_uniform( [state_size, N_HIDDEN_1], -1/math.sqrt(state_size), 1/math.sqrt(state_size)))
                    DNN.variable_summaries(self.W1)
                with tf.name_scope('biases'):
                    self.B1   = tf.Variable( tf.random_uniform( [N_HIDDEN_1], -1/math.sqrt(state_size), 1/math.sqrt(state_size)))
                    DNN.variable_summaries(self.B1)
                with tf.name_scope('pre_bn'):
                    self.PBN1 = tf.matmul( self.state, self.W1 )
                    DNN.variable_summaries(self.PBN1)
                with tf.name_scope('pre_activation'):
                    self.BN1  = batch_norm( self.PBN1, N_HIDDEN_1, self.is_training, self.sess )
                    DNN.variable_summaries(self.BN1.bnorm)
                with tf.name_scope('activation'):
                    self.A1   = self.activations[0]( self.BN1.bnorm ) #+ self.B1 
                    DNN.variable_summaries(self.A1)

            with tf.name_scope("Layer_2"):
                with tf.name_scope('weights'):
                    self.W2   = tf.Variable( tf.random_uniform( [N_HIDDEN_1, N_HIDDEN_2],  -1/math.sqrt(N_HIDDEN_1+action_size), 1/math.sqrt(N_HIDDEN_1+action_size)))  
                    DNN.variable_summaries(self.W2)
                with tf.name_scope('biases'):
                    self.B2   = tf.Variable( tf.random_uniform( [N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1+action_size), 1/math.sqrt(N_HIDDEN_1+action_size)))
                    DNN.variable_summaries(self.B2)
                with tf.name_scope('action_weights'):
                    self.W2_action = tf.Variable( tf.random_uniform( [action_size, N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1+action_size), 1/math.sqrt(N_HIDDEN_1+action_size)))
                    DNN.variable_summaries(self.W2_action)
                with tf.name_scope('pre_bn'):
                    self.PBN2 = tf.matmul(  self.A1, self.W2) + tf.matmul( self.action, self.W2_action )
                    DNN.variable_summaries(self.PBN2)
                with tf.name_scope('pre_activation'):
                    self.BN2  = batch_norm( self.PBN2, N_HIDDEN_2, self.is_training, self.sess )
                    DNN.variable_summaries(self.BN2.bnorm)
                with tf.name_scope('activation'):
                    self.A2   = self.activations[1]( self.BN2.bnorm ) #+ self.B2
                    DNN.variable_summaries(self.A2)

            with tf.name_scope("Output_layer"):
                with tf.name_scope('weights'):
                    self.W3 = tf.Variable( tf.random_uniform( [N_HIDDEN_2, 1],  -0.003,  0.003 ) )
                    DNN.variable_summaries(self.W3)
                with tf.name_scope('biases'):
                    self.B3 = tf.Variable( tf.random_uniform( [1], -0.003, 0.003 ) )
                    DNN.variable_summaries(self.B3)
                with tf.name_scope('activation'):
                    self.Q_value = self.activations[2]( tf.matmul(  self.A2, self.W3 ) + self.B3 )
                    DNN.variable_summaries( self.Q_value )


           # Target Critic Q Network:
            self.t_state  = tf.placeholder( "float32", [None,state_size] , name="t_state"  )
            self.t_action = tf.placeholder( "float32", [None,action_size], name="t_action" )
            
            with tf.name_scope("T_Layer_1"):
                with tf.name_scope('weights'):
                    self.t_W1   = tf.Variable( tf.random_uniform( [state_size,N_HIDDEN_1], -1/math.sqrt(state_size), 1/math.sqrt(state_size) ) )
                with tf.name_scope('biases'):
                    self.t_B1   = tf.Variable( tf.random_uniform( [N_HIDDEN_1], -1/math.sqrt(state_size), 1/math.sqrt(state_size) ) )
                with tf.name_scope('pre_bn'):    
                    self.t_PBN1 = tf.matmul( self.t_state, self.t_W1 )
                with tf.name_scope('pre_activation'):    
                    self.t_BN1  = batch_norm( self.t_PBN1, N_HIDDEN_1, self.is_training, self.sess, self.BN1 )        
                with tf.name_scope('activation'):    
                    self.t_A1   = self.activations[0]( self.t_BN1.bnorm ) #+ self.t_B1
            
            with tf.name_scope("T_Layer_2"):
                with tf.name_scope('weights'):
                    self.t_W2   = tf.Variable( tf.random_uniform( [N_HIDDEN_1, N_HIDDEN_2],  -1/math.sqrt(N_HIDDEN_1+action_size), 1/math.sqrt(N_HIDDEN_1+action_size) ) )  
                    self.t_W2_action = tf.Variable( tf.random_uniform( [action_size, N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1+action_size), 1/math.sqrt(N_HIDDEN_1+action_size) ) )
                with tf.name_scope('biases'):
                    self.t_B2   = tf.Variable( tf.random_uniform( [N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1+action_size), 1/math.sqrt(N_HIDDEN_1+action_size) ) )
                with tf.name_scope('pre_bn'):    
                    self.t_PBN2 = tf.matmul( self.t_A1, self.t_W2) + tf.matmul( self.t_action, self.t_W2_action )
                with tf.name_scope('pre_activation'):    
                    self.t_BN2  = batch_norm( self.t_PBN2, N_HIDDEN_2, self.is_training, self.sess, self.BN2 )
                with tf.name_scope('activation'):    
                    self.t_A2   = self.activations[1]( self.t_BN2.bnorm ) #+ self.t_B2 
            
            with tf.name_scope("T_Output_layer"):
                with tf.name_scope('weights'):
                    self.t_W3 = tf.Variable( tf.random_uniform( [N_HIDDEN_2,1], -0.003, 0.003 ) )
                with tf.name_scope('biases'):
                    self.t_B3 = tf.Variable( tf.random_uniform( [1], -0.003, 0.003 ) )
                with tf.name_scope('activation'):
                    self.t_Q_value = self.activations[2]( tf.matmul( self.t_A2, self.t_W3 ) + self.t_B3 )
            
            
            self.Q_obj         = tf.placeholder( "float32", [None,1], name="obj_Q" ) #supervisor
            self.learning_rate = tf.placeholder( "float32", shape=[], name="learning_rate" )

            #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1)+tf.nn.l2_loss(self.W2)+ tf.nn.l2_loss(self.W2_action) + tf.nn.l2_loss(self.W3)+tf.nn.l2_loss(self.B1)+tf.nn.l2_loss(self.B2)+tf.nn.l2_loss(self.B3) 
            #self.l2_regularizer_loss = 0.0001*tf.reduce_mean( tf.square( self.W2 ) )
            with tf.name_scope("cost"):
                #self.cost           = tf.pow( self.Q_value - self.Q_obj, 2 )/AC_BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.Q_obj)[0])
                self.cost            = tf.reduce_mean( tf.square( self.Q_value - self.Q_obj ) )
            self.optimizer           = tf.train.AdamOptimizer( self.learning_rate ).minimize(self.cost)
            self.act_grad_v          = tf.gradients( self.Q_value, self.action )
            self.action_gradients    = [ self.act_grad_v[0]/tf.to_float( tf.shape(self.act_grad_v[0])[0] ) ] #this is just divided by batch size
            #from simple actor net:
            self.check_fl            = self.action_gradients             
            
            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )
            
            #To initialize critic and target with the same values:
            # copy target parameters
            self.sess.run([
				self.t_W1.assign( self.W1 ),
				self.t_B1.assign( self.B1 ),
				self.t_W2.assign( self.W2 ),
				self.t_W2_action.assign( self.W2_action ),
				self.t_B2.assign( self.B2 ),
				self.t_W3.assign( self.W3 ),
				self.t_B3.assign( self.B3 )
			])
            
            self.update_target_critic_op = [
                self.t_W1.assign( TAU*self.W1 + (1-TAU)*self.t_W1 ),
                self.t_B1.assign( TAU*self.B1 + (1-TAU)*self.t_B1 ),  
                self.t_W2.assign( TAU*self.W2 + (1-TAU)*self.t_W2 ),
                self.t_W2_action.assign( TAU*self.W2_action + (1-TAU)*self.t_W2_action ),
                self.t_B2.assign( TAU*self.B2 + (1-TAU)*self.t_B2 ),
                self.t_W3.assign( TAU*self.W3 + (1-TAU)*self.t_W3 ),
                self.t_B3.assign( TAU*self.B3 + (1-TAU)*self.t_B3 ),
                self.t_BN1.updateTarget,
                self.t_BN2.updateTarget
            ]

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



    def train_critic( self, state_batch, action_batch, q_obj, learning_rate=0.0001 ):
        self.sess.run( [ self.optimizer, self.BN1.train_mean, self.BN1.train_var, self.BN2.train_mean, self.BN2.train_var, self.t_BN1.train_mean, self.t_BN1.train_var, self.t_BN2.train_mean, self.t_BN2.train_var ], feed_dict={ self.state: state_batch, self.t_state: state_batch, self.action: action_batch, self.t_action: action_batch, self.learning_rate: learning_rate, self.Q_obj: q_obj, self.is_training: True } )
        
        cerror = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            cerror, summary = self.sess.run( [self.cost, self.merged], feed_dict={self.state: state_batch, self.action: action_batch, self.t_state: state_batch, self.t_action: action_batch, self.Q_obj: q_obj, self.learning_rate: learning_rate, self.is_training: True} )
            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
        self.counter += 1
        return cerror

    def evaluate_critic( self, state, action ):
        return self.sess.run(self.Q_value, feed_dict={self.state: state, self.action: action, self.is_training: False} )    
        

    def evaluate_target_critic( self, t_state, t_action ):
        return self.sess.run(self.t_Q_value, feed_dict={self.t_state: t_state, self.t_action: t_action, self.is_training: False} )    
        
        
    def compute_grad_Q2a( self, state, action ):
        return self.sess.run(self.action_gradients, feed_dict={self.state: state, self.action: action, self.is_training: False} )

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)
         
    def close_all(self):
        self.train_writer.close()

