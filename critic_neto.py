import numpy as np
import tensorflow as tf
import math
from config import *


N_HIDDEN_1 = CN_N_HIDDENS[0]
N_HIDDEN_2 = CN_N_HIDDENS[1]

class CriticNet:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self, num_states, num_actions):
        
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #c_q_model parameters:
            self.c_W1, self.c_B1, self.c_W2, self.c_W2_action, self.c_B2, self.c_W3, self.c_B3,\
            self.c_q_model,  self.c_state_in,  self.c_action_in  = self.create_critic_net(num_states, num_actions)
                                   
            #create target_q_model:
            self.tc_W1, self.tc_B1, self.tc_W2, self.tc_W2_action, self.tc_B2, self.tc_W3, self.tc_B3,\
            self.tc_q_model, self.tc_state_in, self.tc_action_in = self.create_critic_net(num_states, num_actions)
            
            self.q_value_in = tf.placeholder("float",[None,1]) #supervisor
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            #self.l2_regularizer_loss = tf.nn.l2_loss(self.c_W1)+tf.nn.l2_loss(self.c_W2)+ tf.nn.l2_loss(self.c_W2_action) + tf.nn.l2_loss(self.c_W3)+tf.nn.l2_loss(self.c_B1)+tf.nn.l2_loss(self.c_B2)+tf.nn.l2_loss(self.c_B3) 
            self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.c_W2,2))+ 0.0001*tf.reduce_sum(tf.pow(self.c_B2,2))             
            self.cost      = tf.pow( self.c_q_model-self.q_value_in,2 )/AC_BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.q_value_in)[0])
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            
            #action gradient to be used in actor network:
            #self.action_gradients=tf.gradients(self.c_q_model,self.c_action_in)
            #from simple actor net:
            self.act_grad_v = tf.gradients(self.c_q_model, self.c_action_in)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])] #this is just divided by batch size
            #from simple actor net:
            self.check_fl = self.action_gradients             
            
            #initialize all tensor variable parameters:
            self.sess.run(tf.global_variables_initializer())
            
            #To make sure critic and target have same parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
				self.tc_W1.assign(self.c_W1),
				self.tc_B1.assign(self.c_B1),
				self.tc_W2.assign(self.c_W2),
				self.tc_W2_action.assign(self.c_W2_action),
				self.tc_B2.assign(self.c_B2),
				self.tc_W3.assign(self.c_W3),
				self.tc_B3.assign(self.c_B3)
			])
            
            self.update_target_critic_op = [
                self.tc_W1.assign( TAU*self.c_W1+(1-TAU)*self.tc_W1 ),
                self.tc_B1.assign( TAU*self.c_B1+(1-TAU)*self.tc_B1 ),
                self.tc_W2.assign( TAU*self.c_W2+(1-TAU)*self.tc_W2 ),
                self.tc_W2_action.assign( TAU*self.c_W2_action+(1-TAU)*self.tc_W2_action ),
                self.tc_B2.assign( TAU*self.c_B2+(1-TAU)*self.tc_B2 ),
                self.tc_W3.assign( TAU*self.c_W3+(1-TAU)*self.tc_W3 ),
                self.tc_B3.assign( TAU*self.c_B3+(1-TAU)*self.tc_B3 )
            ]
            
    def create_critic_net(self, num_states=4, num_actions=1):

        c_state_in = tf.placeholder("float",[None,num_states])
        c_action_in = tf.placeholder("float",[None,num_actions])
    
        c_W1 = tf.Variable( tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)) )
        c_B1 = tf.Variable( tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)) )
        c_W2 = tf.Variable( tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)) )    
        c_W2_action = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)) )    
        c_B2 = tf.Variable( tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)) )
        c_W3 = tf.Variable( tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003) )
        c_B3 = tf.Variable( tf.random_uniform([1],-0.003,0.003) )
    
        c_H1 = tf.nn.softplus( tf.matmul( c_state_in , c_W1 ) + c_B1 )
        c_H2 = tf.nn.relu( tf.matmul( c_H1,c_W2 ) + tf.matmul( c_action_in,c_W2_action ) + c_B2 )
        c_q_model = tf.matmul( c_H2 , c_W3 ) + c_B3

        return c_W1, c_B1, c_W2, c_W2_action, c_B2, c_W3, c_B3, c_q_model, c_state_in, c_action_in
    
    def train_critic(self, state_batch, action_batch, tq_batch, learning_rate=0.0001):
        self.sess.run(self.optimizer, feed_dict={self.c_state_in:state_batch, self.c_action_in:action_batch, self.q_value_in:tq_batch, self.learning_rate: learning_rate})
             
    def evaluate_critic(self, state_1, action_1):
        return self.sess.run(self.c_q_model, feed_dict={self.c_state_in: state_1, self.c_action_in: action_1})

    def evaluate_target_critic(self,state,action):
        return self.sess.run(self.tc_q_model, feed_dict={self.tc_state_in: state, self.tc_action_in: action})
        
    def compute_delQ_a(self,state_t,action_t):
#        print '\n'
#        print 'check grad number'        
#        ch= self.sess.run(self.check_fl, feed_dict={self.c_state_in: state_t,self.c_action_in: action_t})
#        print len(ch)
#        print len(ch[0])        
#        raw_input("Press Enter to continue...")
        return self.sess.run(self.action_gradients, feed_dict={self.c_state_in: state_t,self.c_action_in: action_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)


