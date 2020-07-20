import numpy as np
import tensorflow as tf
import math
from config import *


N_HIDDEN_1 = AN_N_HIDDENS[0]
N_HIDDEN_2 = AN_N_HIDDENS[1]


class ActorNet:
    """ Actor Network Model of DDPG Algorithm """
    
    def __init__(self, num_states, num_actions):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
           
            #actor network model parameters:
            self.a_W1, self.a_B1, self.a_W2, self.a_B2, self.a_W3, self.a_B3,\
            self.a_state_in,  self.actor_model   = self.create_actor_net( num_states, num_actions )
            
                                   
            #target actor network model parameters:
            self.ta_W1, self.ta_B1, self.ta_W2, self.ta_B2, self.ta_W3, self.ta_B3,\
            self.ta_state_in, self.t_actor_model = self.create_actor_net( num_states, num_actions )
            
            #cost of actor network:
            self.q_gradient_input     = tf.placeholder( "float", [None, num_actions] ) #gets input from action_gradient computed in critic network file
            self.learning_rate        = tf.placeholder(tf.float32, shape=[])
            self.actor_parameters     = [self.a_W1, self.a_B1, self.a_W2, self.a_B2, self.a_W3, self.a_B3]
            self.parameters_gradients = tf.gradients(self.actor_model, self.actor_parameters, - self.q_gradient_input )#/AC_BATCH_SIZE) 
            self.optimizer            = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.parameters_gradients, self.actor_parameters ) )  
            #initialize all tensor variable parameters:
            self.sess.run( tf.global_variables_initializer() )    

            #To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
				self.ta_W1.assign(self.a_W1),
				self.ta_B1.assign(self.a_B1),
				self.ta_W2.assign(self.a_W2),
				self.ta_B2.assign(self.a_B2),
				self.ta_W3.assign(self.a_W3),
				self.ta_B3.assign(self.a_B3) ])

            self.update_target_actor_op = [
                self.ta_W1.assign( TAU*self.a_W1 + (1-TAU)*self.ta_W1 ),
                self.ta_B1.assign( TAU*self.a_B1 + (1-TAU)*self.ta_B1 ),
                self.ta_W2.assign( TAU*self.a_W2 + (1-TAU)*self.ta_W2 ),
                self.ta_B2.assign( TAU*self.a_B2 + (1-TAU)*self.ta_B2 ),
                self.ta_W3.assign( TAU*self.a_W3 + (1-TAU)*self.ta_W3 ),
                self.ta_B3.assign( TAU*self.a_B3 + (1-TAU)*self.ta_B3 )]
        
            net_path   = "./model/an"
            self.saver = tf.train.Saver()
            self.saver.save(self.sess, net_path + "/net.ckpt")
            writer = tf.summary.FileWriter( net_path )
            writer.add_graph(self.sess.graph)
            writer.close()
            self.merged       = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter( net_path, self.sess.graph )
            self.run_metadata = tf.RunMetadata()

    def create_actor_net(self, num_states=4, num_actions=1):
        """ Network that takes states and return action """

        a_state_in = tf.placeholder( "float", [None,num_states] )    
        a_W1 = tf.Variable( tf.random_uniform( [num_states,N_HIDDEN_1], -1/math.sqrt(num_states), 1/math.sqrt(num_states) ) )
        a_B1 = tf.Variable( tf.random_uniform( [N_HIDDEN_1],            -1/math.sqrt(num_states), 1/math.sqrt(num_states) ) )
        a_W2 = tf.Variable( tf.random_uniform( [N_HIDDEN_1,N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1) ) )
        a_B2 = tf.Variable( tf.random_uniform( [N_HIDDEN_2],            -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1) ) )
        a_W3 = tf.Variable( tf.random_uniform( [N_HIDDEN_2,num_actions],-0.003,                   0.003 ) )
        a_B3 = tf.Variable( tf.random_uniform( [num_actions],           -0.003,                   0.003 ) )
    
        a_H1        = tf.nn.softplus(  tf.matmul( a_state_in, a_W1 ) + a_B1 )
        a_H2        = tf.nn.relu(      tf.matmul( a_H1,       a_W2 ) + a_B2 )
        actor_model = tf.nn.sigmoid( tf.matmul( a_H2,       a_W3 ) + a_B3   )#
        return a_W1, a_B1, a_W2, a_B2, a_W3, a_B3, a_state_in, actor_model
        
        
    def evaluate_actor(self, state):
        return self.sess.run(self.actor_model, feed_dict={self.a_state_in:state})        
        
        
    def evaluate_target_actor(self, state):
        return self.sess.run(self.t_actor_model, feed_dict={self.ta_state_in: state})
        
    def train_actor(self, a_state_in, q_gradient_input, learning_rate=0.0001):
        self.sess.run(self.optimizer, feed_dict={ self.a_state_in: a_state_in, self.q_gradient_input: q_gradient_input, self.learning_rate: learning_rate})
    
    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)    

        