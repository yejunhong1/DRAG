import tensorflow as tf
import math
from batch_norm import *
import numpy as np
from config import *
import FC_DNN_PBN as DNN


class CriticNet_bn:
    """ Critic Q value network with batch normalization of the DDPG algorithm 
    
    Modularized version of the original one

    state_size:  size of the state vector/tensor

    action_size: size of the action vector/tensor

    TAU:         update rate of target network parameters

    write_sum:   key/interval for writing summary data to file
    """
    def __init__(self, state_size, action_size, TAU = 0.001, write_sum = 0, net_size_scale=1 ):
        self.counter     = 0
        self.state_size  = state_size
        self.action_size = action_size
        self.output_size = 1
        self.write_sum   = write_sum
        self.input_size  = state_size + action_size

        self.hidden_dims = [int(n*net_size_scale) for n in CN_N_HIDDENS]
        self.activations = CN_ACTS
        #print("n hiddens: "+str(self.hidden_dims))
        tf.reset_default_graph()
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #Critic Q Network:
            self.state       = tf.placeholder( "float32", [None, state_size],  name="state"  )
            self.action      = tf.placeholder( "float32", [None, action_size], name="action"  )
            self.t_state     = tf.placeholder( "float32", [None, state_size] , name="t_state"  )
            self.t_action    = tf.placeholder( "float32", [None, action_size], name="t_action" )
            self.is_training = tf.placeholder( tf.bool,   [],                  name="is_training" )
            
            # Critic network
            self.A1, self.para1, self.bn_para1 = DNN.nn_bn_layer( self.state, self.state_size, 
                self.hidden_dims[0], "Layer_1", self.activations[0], self.is_training, self.sess )
            
            with tf.name_scope("Layer_2"):
                with tf.name_scope('weights'):
                    self.W2   = tf.Variable( tf.random_uniform( [self.hidden_dims[0], self.hidden_dims[1]],  -1/math.sqrt(self.hidden_dims[0]+action_size), 1/math.sqrt(self.hidden_dims[0]+action_size) ) )  
                    DNN.variable_summaries(self.W2)
                with tf.name_scope('biases'):
                    self.B2   = tf.Variable( tf.random_uniform( [self.hidden_dims[1]], -1/math.sqrt(self.hidden_dims[0]+action_size), 1/math.sqrt(self.hidden_dims[0]+action_size) ) )
                    DNN.variable_summaries(self.B2)
                with tf.name_scope('action_weights'):
                    self.W2_action = tf.Variable( tf.random_uniform( [action_size, self.hidden_dims[1]], -1/math.sqrt(self.hidden_dims[0]+action_size), 1/math.sqrt(self.hidden_dims[0]+action_size) ) )
                    DNN.variable_summaries(self.W2_action)
                with tf.name_scope('pre_bn'):
                    self.PBN2 = tf.matmul(  self.A1, self.W2) + tf.matmul( self.action, self.W2_action )
                    DNN.variable_summaries(self.PBN2)
                with tf.name_scope('pre_activation'):
                    self.BN2  = batch_norm( self.PBN2, self.hidden_dims[1], self.is_training, self.sess )
                    DNN.variable_summaries(self.BN2.bnorm)
                with tf.name_scope('activation'):
                    self.A2   = self.activations[1]( self.BN2.bnorm ) #+ self.B2
                    DNN.variable_summaries(self.A2)

            self.Q_value, self.out_paras = DNN.nn_layer( self.A2, self.hidden_dims[1], self.output_size, "Output", self.activations[2] )

            self.parameters = [self.para1[0], self.W2, self.W2_action ]
            self.parameters.extend( self.out_paras )
            self.bn_paras = [ self.bn_para1[0], self.BN2 ]

            # Target critic network
            self.t_A1, self.t_para1, self.t_bn_para1 = DNN.nn_bn_layer( self.t_state, self.state_size, 
                self.hidden_dims[0], "T_Layer_1", self.activations[0], self.is_training, self.sess, self.bn_paras[0] )

            with tf.name_scope("T_Layer_2"):
                with tf.name_scope('weights'):
                    self.t_W2   = tf.Variable( tf.random_uniform( [self.hidden_dims[0], self.hidden_dims[1]],  -1/math.sqrt(self.hidden_dims[0]+action_size), 1/math.sqrt(self.hidden_dims[0]+action_size) ) )  
                    DNN.variable_summaries(self.t_W2)
                with tf.name_scope('biases'):
                    self.t_B2   = tf.Variable( tf.random_uniform( [self.hidden_dims[1]], -1/math.sqrt(self.hidden_dims[0]+action_size), 1/math.sqrt(self.hidden_dims[0]+action_size) ) )
                    DNN.variable_summaries(self.t_B2)
                with tf.name_scope('action_weights'):
                    self.t_W2_action = tf.Variable( tf.random_uniform( [action_size, self.hidden_dims[1]], -1/math.sqrt(self.hidden_dims[0]+action_size), 1/math.sqrt(self.hidden_dims[0]+action_size) ) )
                    DNN.variable_summaries(self.t_W2_action)
                with tf.name_scope('pre_bn'):
                    self.t_PBN2 = tf.matmul(  self.t_A1, self.t_W2) + tf.matmul( self.t_action, self.t_W2_action )
                    DNN.variable_summaries(self.t_PBN2)
                with tf.name_scope('pre_activation'):
                    self.t_BN2  = batch_norm( self.t_PBN2, self.hidden_dims[1], self.is_training, self.sess, self.bn_paras[1] )
                    DNN.variable_summaries(self.t_BN2.bnorm)
                with tf.name_scope('activation'):
                    self.t_A2   = self.activations[1]( self.t_BN2.bnorm ) #+ self.B2
                    DNN.variable_summaries(self.t_A2)

            self.t_Q_value, self.t_out_paras = DNN.nn_layer( self.t_A2, self.hidden_dims[1], self.output_size, "T_Output", self.activations[2] )

            self.t_parameters = [self.t_para1[0], self.t_W2, self.t_W2_action ]
            self.t_parameters.extend( self.t_out_paras )
            self.t_bn_paras = [ self.t_bn_para1[0], self.t_BN2 ]

            self.parameters_name    = ["W1","W2","W2_action","W3","B3"]
            self.bn_parameters_name = ["BN1","BN2"]
            self.num_paras = len(self.parameters)
            self.num_bn    = len(self.bn_paras)

            self.Q_obj         = tf.placeholder( "float32", [None,1], name="obj_Q" ) #supervisor
            self.learning_rate = tf.placeholder( "float32", shape=[], name="learning_rate" )
            
            with tf.name_scope("cost"):
                #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_state)+tf.nn.l2_loss(self.W2)+ tf.nn.l2_loss(self.W2_action) + tf.nn.l2_loss(self.W3)+tf.nn.l2_loss(self.B1)+tf.nn.l2_loss(self.B2)+tf.nn.l2_loss(self.B3) 
                #self.l2_regularizer_loss = 0.0001*tf.reduce_mean( tf.square( self.W2 ) )             
                #self.cost                = tf.pow( self.Q_value - self.Q_obj, 2 )/AC_BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.Q_obj)[0])
                self.cost = tf.reduce_mean( tf.square( self.Q_value - self.Q_obj ) )
            with tf.name_scope("Grad_min_cost"):
                self.optimizer = tf.train.AdamOptimizer( self.learning_rate, name="critic_grad" ).minimize(self.cost, name="min_cost")
            
            #action gradient to be used in actor network:
            with tf.name_scope("Grad_Q2a"):
                self.act_grad_v        = tf.gradients( self.Q_value, self.action, name="Grad_to_action")
                self.action_gradients  = [ self.act_grad_v[0]/tf.to_float( tf.shape(self.act_grad_v[0])[0] ) ] #this is just divided by batch size
                      
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

            # operations to be evaluated during training, including apply gradient and update statistics in BN layers
            self.train_op = [ self.optimizer ]
            for i in range(self.num_bn):
                self.train_op.append( self.bn_paras[i].train_mean )
                self.train_op.append( self.bn_paras[i].train_var  )
                self.train_op.append( self.t_bn_paras[i].train_mean )
                self.train_op.append( self.t_bn_paras[i].train_var  )
            
            net_path = "./model/cn"
            if self.write_sum > 0:
                self.saver = tf.train.Saver()
                self.saver.save(self.sess, net_path + "/net.ckpt")
                writer = tf.summary.FileWriter( net_path )
                writer.add_graph(self.sess.graph)
                writer.close()
                self.merged       = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter( net_path, self.sess.graph )
                self.run_metadata = tf.RunMetadata()
                #self.saver, self.merged, self.train_writer, self.run_metadata = DNN.init_summary(net_path, self.sess)


    def train_critic( self, state_batch, action_batch, q_obj, learning_rate=0.0001 ):
        self.sess.run( self.train_op, feed_dict={ self.state: state_batch, self.t_state: state_batch, self.action: action_batch, 
            self.t_action: action_batch, self.learning_rate: learning_rate, self.Q_obj: q_obj, self.is_training: True } )

        cerror = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            cerror, summary = self.sess.run( [ self.cost, self.merged ], feed_dict={ self.state: state_batch, 
                self.t_state: state_batch, self.action: action_batch, self.t_action: action_batch, self.Q_obj: q_obj, 
                self.learning_rate: learning_rate, self.is_training: False } )

            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
        self.counter += 1
        return cerror

    def evaluate_critic( self, state, action ):
        return self.sess.run(self.Q_value, feed_dict={ self.state: state, self.action: action, self.is_training: False } )    
        

    def evaluate_target_critic( self, t_state, t_action ):
        return self.sess.run(self.t_Q_value, feed_dict={ self.t_state: t_state, self.t_action: t_action, self.is_training: False } )    
        
        
    def compute_grad_Q2a( self, state, action ):
        return self.sess.run(self.action_gradients, feed_dict={ self.state: state, self.action: action, self.is_training: False } )

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)
         
    def close_all(self):
        self.train_writer.close()

