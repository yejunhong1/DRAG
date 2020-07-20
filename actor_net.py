import numpy as np
import tensorflow as tf
import math
from config import *
import FC_DNN_P as DNN
#from FC_DNN_P import *



class ActorNet:
    """ Actor Network Model of DDPG Algorithm 
    state_size:  size of the state vector/tensor
    action_size: size of the action vector/tensor
    TAU:         update rate of target network parameters
    write_sum:   key/interval for writing summary data to file
    """
    
    def __init__(self, state_size, action_size, TAU = 0.001, write_sum = 0, net_size_scale=1 ):
        self.hidden_dims = [int(n*net_size_scale) for n in AN_N_HIDDENS]
        self.activations = AN_ACTS
        self.counter     = 0
        self.write_sum   = write_sum        # if to write the summary file
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            self.state     = tf.placeholder( "float32", [None,state_size], name="state" )
            self.t_state   = tf.placeholder( "float32", [None,state_size], name="Target_state" )
            self.learning_rate = tf.placeholder( "float32", shape=[] )
            self.obj_action    = tf.placeholder( "float32", [None,action_size], name="Obj_action" )

            self.action,   self.parameters   = DNN.multilayer_perceptron( self.state,   state_size, self.hidden_dims, action_size, self.activations)
            self.t_action, self.t_parameters = DNN.multilayer_perceptron( self.t_state, state_size, self.hidden_dims, action_size, self.activations, "Target_")
            self.a_parameters_name = ["W1","B1","W2","B2","W3","B3"]
            self.action   = (self.action + 1 )/2
            self.t_action = (self.t_action + 1 )/2
            #cost of actor network:
            self.q_gradient_input     = tf.placeholder( "float32", [None, action_size], name="Grad_Q2a" ) #gets input from action_gradient computed in critic network file

            self.parameters_gradients = tf.gradients(self.action, self.parameters, - self.q_gradient_input )#/A_BATCH_SIZE) 
            self.optimizer            = tf.train.AdamOptimizer( self.learning_rate ).apply_gradients( zip( self.parameters_gradients, self.parameters ) )  
            self.cost                 = tf.reduce_mean( tf.square( tf.round( self.action ) - self.obj_action ) )
            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )

            # To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_parameters[0].assign( self.parameters[0] ),
                self.t_parameters[1].assign( self.parameters[1] ),
                self.t_parameters[2].assign( self.parameters[2] ),
                self.t_parameters[3].assign( self.parameters[3] ),
                self.t_parameters[4].assign( self.parameters[4] ),
                self.t_parameters[5].assign( self.parameters[5] ) ])
            
            self.update_target_actor_op = [
                self.t_parameters[0].assign( TAU*self.parameters[0] + (1-TAU)*self.t_parameters[0] ),
                self.t_parameters[1].assign( TAU*self.parameters[1] + (1-TAU)*self.t_parameters[1] ),
                self.t_parameters[2].assign( TAU*self.parameters[2] + (1-TAU)*self.t_parameters[2] ),
                self.t_parameters[3].assign( TAU*self.parameters[3] + (1-TAU)*self.t_parameters[3] ),
                self.t_parameters[4].assign( TAU*self.parameters[4] + (1-TAU)*self.t_parameters[4] ),
                self.t_parameters[5].assign( TAU*self.parameters[5] + (1-TAU)*self.t_parameters[5] ) ]
            #for i in range( 0, len(self.t_parameters) ):
            #    self.t_parameters[i].assign( self.parameters[i] )   # may need modify with .assign
            
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
        return self.sess.run( self.action, feed_dict={ self.state: state } )        
        
    def evaluate_target_actor( self, t_state ):
        return self.sess.run( self.t_action, feed_dict={ self.t_state: t_state } )
        
    def train_actor( self, state, obj_actioin, q_gradient_input, learning_rate=0.0001 ):
        self.sess.run( self.optimizer, feed_dict={ self.state: state, self.t_state: state, self.obj_action: obj_actioin, self.q_gradient_input: q_gradient_input, self.learning_rate: learning_rate} )        
        
        aerror = 1
        if (self.write_sum >0 ) and (self.counter%self.write_sum == 0):
            summary, aerror = self.sess.run( [self.merged, self.cost], feed_dict={ self.state: state, self.t_state: state, self.obj_action: obj_actioin, self.q_gradient_input: q_gradient_input, self.learning_rate: learning_rate} )
            self.train_writer.add_run_metadata( self.run_metadata, 'step%03d' % self.counter )
            self.train_writer.add_summary( summary, self.counter )
        self.counter += 1
        return aerror



    def update_target_actor( self):
        self.sess.run(self.update_target_actor_op)

        #for i in range( 0, len(self.t_parameters) ):
        #    self.t_parameters[i].assign( TAU*self.parameters[i] + (1-TAU)*self.t_parameters[i] )

    def close_all( self ):
        self.train_writer.close()
        DNN.write_var( self.sess, self.parameters, varname= self.a_parameters_name, filename="actor_var" )

    