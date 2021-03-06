import tensorflow as tf
import math
from batch_norm import *
import numpy as np
from config import *
import FC_DNN_PBN as DNN


class CriticNet_bn:
    """ Critic Q value network with batch normalization of the DDPG algorithm 
    This network takes the state and action as inputs in the first layer
    state_size:  size of the state vector/tensor
    action_size: size of the action vector/tensor
    TAU:         update rate of target network parameters
    write_sum:   key/interval for writing summary data to file
    """
    def __init__(self, state_size, action_size, TAU = 0.001, write_sum = 0):
        self.counter     = 0
        self.state_size  = state_size
        self.action_size = action_size
        self.write_sum   = write_sum
        self.input_size  = state_size + action_size

        self.hidden_dims = CN_N_HIDDENS
        self.activations = CN_ACTS

        tf.reset_default_graph()
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #Critic Q Network:
            self.input       = tf.placeholder( "float32", [None, self.input_size], name="state_action" )
            self.t_input     = tf.placeholder( "float32", [None, self.input_size], name="Target_state_action" )
            self.is_training = tf.placeholder( tf.bool,   [],                      name="is_training" )
            

            self.Q_value,   self.parameters,   self.bn_paras    = DNN.multilayer_perceptron( self.input,   self.input_size, self.hidden_dims, 1, self.activations, self.is_training, self.sess )
            self.t_Q_value, self.t_parameters, self.t_bn_paras  = DNN.multilayer_perceptron( self.t_input, self.input_size, self.hidden_dims, 1, self.activations, self.is_training, self.sess, self.bn_paras, pre="T_" )

            self.parameters_name    = ["W1","W2","W3","B3"]
            self.bn_parameters_name = ["BN1","BN2"]
            self.num_paras = len(self.parameters)
            self.num_bn    = len(self.bn_paras)

            self.Q_obj         = tf.placeholder( "float32", [None,1], name="obj_Q" ) #supervisor
            self.learning_rate = tf.placeholder( "float32", shape=[], name="learning_rate" )
            
            with tf.name_scope("cost"):
                #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_state)+tf.nn.l2_loss(self.W2)+ tf.nn.l2_loss(self.W2_action) + tf.nn.l2_loss(self.W3)+tf.nn.l2_loss(self.B1)+tf.nn.l2_loss(self.B2)+tf.nn.l2_loss(self.B3) 
                #self.l2_regularizer_loss = 0.0001*tf.reduce_mean( tf.square( self.W2 ) )             
                #self.cost                = tf.pow( self.Q_value - self.Q_obj, 2 )/AC_BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.Q_obj)[0])
                self.cost         = tf.reduce_mean( tf.square( self.Q_value - self.Q_obj ) )
            with tf.name_scope("Grad_min_cost"):
                self.optimizer        = tf.train.AdamOptimizer( self.learning_rate, name="critic_grad" ).minimize(self.cost, name="min_cost")
            
            #action gradient to be used in actor network:
            with tf.name_scope("Grad_Q2a"):
                self.grad_to_input    = tf.gradients( self.Q_value, self.input, name="Grad_to_action")
                self.act_grad_v       = self.grad_to_input[0][:,state_size:]
                self.action_gradients = [ self.act_grad_v / tf.to_float( tf.shape(self.act_grad_v[0])[0] ) ] #this is just divided by batch size
                      
            
            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )

            self.init_target_op          = []   # To initialize critic and target with the same values: copy target parameters
            self.update_target_critic_op = []   # Operations to update target network, including BN parameters
            for i in range(self.num_paras):
                self.init_target_op.append( self.t_parameters[i].assign( self.parameters[i] ) )
                self.update_target_critic_op.append( self.t_parameters[i].assign( TAU*self.parameters[i] + (1-TAU)*self.t_parameters[i] ) )

            for i in range(self.num_bn):
                self.update_target_critic_op.append( self.t_bn_paras[i].updateTarget )

            self.sess.run( self.init_target_op )
            """
            self.sess.run([
				self.t_W1_state.assign(  self.W1_state  ),
                self.t_W1_action.assign( self.W1_action ),
				self.t_B1.assign( self.B1 ),
				self.t_W2.assign( self.W2 ),
				self.t_B2.assign( self.B2 ),
				self.t_W3.assign( self.W3 ),
				self.t_B3.assign( self.B3 )
			])
            
            self.update_target_critic_op = [
                self.t_W1_state.assign(  TAU*self.W1_state  + (1-TAU)*self.t_W1_state  ),
                self.t_W1_action.assign( TAU*self.W1_action + (1-TAU)*self.t_W1_action ),
                self.t_B1.assign( TAU*self.B1 + (1-TAU)*self.t_B1 ),
                self.t_W2.assign( TAU*self.W2 + (1-TAU)*self.t_W2 ),
                self.t_B2.assign( TAU*self.B2 + (1-TAU)*self.t_B2 ),
                self.t_W3.assign( TAU*self.W3 + (1-TAU)*self.t_W3 ),
                self.t_B3.assign( TAU*self.B3 + (1-TAU)*self.t_B3 ),
                self.t_BN1.updateTarget,
                self.t_BN2.updateTarget
            ]
            """
            # operations to be evaluated during training, including apply gradient and update statistics in BN layers
            self.train_op = [ self.optimizer ]
            for i in range(self.num_bn):
                self.train_op.append( self.bn_paras[i].train_mean )
                self.train_op.append( self.bn_paras[i].train_var  )
                self.train_op.append( self.t_bn_paras[i].train_mean )
                self.train_op.append( self.t_bn_paras[i].train_var  )

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
        input = np.concatenate( [state_batch, action_batch], axis=1 )
        self.sess.run( self.train_op, feed_dict={ self.input: input, self.t_input: input, self.learning_rate: learning_rate, self.Q_obj: q_obj, self.is_training: True } )
        cerror = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            cerror, summary = self.sess.run( [ self.cost, self.merged ], feed_dict={ self.input: input, self.t_input: input, self.Q_obj: q_obj, self.learning_rate: learning_rate, self.is_training: True } )
            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
        self.counter += 1
        return cerror

    def evaluate_critic( self, state, action ):
        input = np.concatenate( [state, action], axis=1 )
        return self.sess.run(self.Q_value, feed_dict={ self.input: input, self.is_training: False } )    
        

    def evaluate_target_critic( self, t_state, t_action ):
        input = np.concatenate([t_state, t_action], axis=1)
        return self.sess.run(self.t_Q_value, feed_dict={ self.t_input: input, self.is_training: False } )    
        
        
    def compute_grad_Q2a( self, state, action ):
        input = np.concatenate( [state, action], axis=1 )
        return self.sess.run(self.action_gradients, feed_dict={ self.input: input, self.is_training: False } )

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)
         
    def close_all(self):
        self.train_writer.close()

