import tensorflow as tf
import math
from batch_norm import *
import numpy as np
from config import *
import FC_DNN_PBN as DNN


class ActorNet_bn:
    """ Actor Network with batch normalization of DDPG Algorithm 

    state_size:  size of the state vector/tensor

    action_size: size of the action vector/tensor

    TAU:         update rate of target network parameters

    write_sum:   key/interval for writing summary data to file
    """
    def __init__(self, state_size, action_size, TAU = 0.001, write_sum = 0, net_size_scale=1):
        tf.reset_default_graph()
        self.counter     = 0
        self.state_size  = state_size
        self.action_size = action_size
        self.write_sum   = write_sum        # if to write the summary file

        self.hidden_dims = [int(n*net_size_scale) for n in AN_N_HIDDENS]
        self.activations = AN_ACTS
        #print("n hiddens: "+str(self.hidden_dims))
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #actor network model parameters:
            self.state       = tf.placeholder( "float32", [None, state_size], name="state" )
            self.t_state     = tf.placeholder( "float32", [None, state_size], name="t_state" )
            self.is_training = tf.placeholder( tf.bool,   [],                 name="is_training" )
            
            self.action,   self.parameters,   self.bn_paras   = DNN.multilayer_perceptron( self.state,   self.state_size, 
                self.hidden_dims, self.action_size, self.activations, self.is_training, self.sess )
            self.t_action, self.t_parameters, self.t_bn_paras = DNN.multilayer_perceptron( self.t_state, self.state_size, 
                self.hidden_dims, self.action_size, self.activations, self.is_training, self.sess, parForTarget=self.bn_paras, pre="Target_" )
            
            self.parameters_name    = ["W1","W2","W3","B3"]
            self.bn_parameters_name = ["BN1","BN2"]
            self.num_paras = len(self.parameters)
            self.num_bn    = len(self.bn_paras)

            self.learning_rate    = tf.placeholder( "float32", shape=[],                 name="LR" )
            self.obj_action       = tf.placeholder( "float32", [None, self.action_size], name="obj_action" )
            self.q_gradient_input = tf.placeholder( "float32", [None, self.action_size], name="q_gradient_input" ) #gets input from action_gradient computed in critic network file
            
            #cost of actor network:
            with tf.name_scope('cost'):
                self.cost = tf.reduce_mean( tf.square( tf.round(self.action) - self.obj_action ) )
            #self.all_parameters = [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3, self.BN1.scale, self.BN1.beta, self.BN2.scale, self.BN2.beta]
            self.all_parameters = self.parameters
            for i in range(self.num_bn):
                self.all_parameters.append( self.bn_paras[i].scale )
                self.all_parameters.append( self.bn_paras[i].beta  )

            self.parameters_gradients = tf.gradients( self.action, self.all_parameters, -self.q_gradient_input )#/BATCH_SIZE) changed -self.q_gradient to -
            self.optimizer = tf.train.AdamOptimizer( self.learning_rate, name="action_grad" ).apply_gradients( zip(self.parameters_gradients, self.all_parameters), name="apply_gradient" )  
            
            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )    
            
            #To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.init_target_op         = []
            self.update_target_actor_op = []
            for i in range(self.num_paras):
                self.init_target_op.append( self.t_parameters[i].assign( self.parameters[i] ) )
                self.update_target_actor_op.append( self.t_parameters[i].assign( TAU*self.parameters[i] + (1-TAU)*self.t_parameters[i] ) )
            """
            self.sess.run([
                self.t_parameters[0].assign( self.parameters[0] ),
                self.t_parameters[1].assign( self.parameters[1] ),
                self.t_parameters[2].assign( self.parameters[2] ),
                self.t_parameters[3].assign( self.parameters[3] ) ] )

            self.update_target_actor_op = [
                self.t_parameters[0].assign( TAU*self.parameters[0] + (1-TAU)*self.t_parameters[0] ),
                self.t_parameters[1].assign( TAU*self.parameters[1] + (1-TAU)*self.t_parameters[1] ),
                self.t_parameters[2].assign( TAU*self.parameters[2] + (1-TAU)*self.t_parameters[2] ),
                self.t_parameters[3].assign( TAU*self.parameters[3] + (1-TAU)*self.t_parameters[3] ),
                self.t_bn_paras[0].updateTarget,
                self.t_bn_paras[1].updateTarget
            ]
            """
            # operations to be evaluated during training, including apply gradient and update statistics in BN layers
            self.train_op = [ self.optimizer ]
            for i in range(self.num_bn):
                self.train_op.append( self.bn_paras[i].train_mean )
                self.train_op.append( self.bn_paras[i].train_var  )
                self.train_op.append( self.t_bn_paras[i].train_mean )
                self.train_op.append( self.t_bn_paras[i].train_var  )
                self.update_target_actor_op.append( self.t_bn_paras[i].updateTarget )

            self.sess.run( self.init_target_op )
            
            net_path   = "./model/an"
            if self.write_sum > 0:
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
        self.sess.run( self.train_op, feed_dict={ self.state: state, self.t_state: state, 
            self.q_gradient_input: q_gradient_input, self.learning_rate: learning_rate, self.is_training: True } )
        
        aerror = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            summary, aerror = self.sess.run( [self.merged, self.cost], feed_dict={ self.state: state, self.t_state: state, 
                self.obj_action: obj_actioin, self.q_gradient_input: q_gradient_input, self.learning_rate: learning_rate, self.is_training: False } )
            
            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
        self.counter += 1
        return aerror

    def update_target_actor(self):
        self.sess.run( self.update_target_actor_op )    
        
    def close_all(self):
        self.train_writer.close()
        #DNN.write_var( self.sess, self.parameters, varname= self.parameters_name, filename="actor_var" )
