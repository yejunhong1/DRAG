#-*-coding:utf-8-*-
import numpy as np
import random
import copy
import math
from collections import deque
from tensorflow_grad_inverter import grad_inverter
from config import *
from environment import ENV


from actor_net import ActorNet
from critic_net import CriticNet
from actor_net_bn import ActorNet_bn
from critic_net_bn import CriticNet_bn
from critic_net_bn_3 import CriticNet_bn_3
from ar_pred_net import ARPredNet
from load_map_net import LoadMapNet
from ar_pred_net_bn import ARPredNet_bn
from load_map_net_bn import LoadMapNet_bn

class DDPG:

    """ Deep Deterministic Policy Gradient Algorithm

    hisar_size: size of the history ar vector/tensor

    action_size: size of the action vector/tensor

    TAU: update rate of target network parameters

    is_batch_norm: if apply batch norm

    write_sum: key/interval for writing summary data to file
    """
    def __init__( self, hisar_size, ar_size, action_size, TAU = 0.001, is_batch_norm = 0, write_sum = 0, net_size_scale=1, max_load=1, beta0=beta):
        self.hisar_size  = hisar_size
        self.load_size   = action_size + 1
        self.ar_size     = ar_size
        self.state_size  = action_size * 2
        self.action_size = action_size
        self.ar_action_size = ar_size + action_size

        #print("net_size_scale: "+str(net_size_scale))
        if is_batch_norm:
            if len(CN_N_HIDDENS)==2:
                self.critic_net   = CriticNet_bn(  self.state_size, self.action_size, TAU, write_sum, net_size_scale  )
            else:
                self.critic_net   = CriticNet_bn_3(  self.state_size, self.action_size, TAU, write_sum, net_size_scale  )
            self.actor_net    = ActorNet_bn(   self.state_size, self.action_size, TAU, write_sum, net_size_scale  )
            self.ar_pred_net  = ARPredNet_bn(  self.hisar_size, self.ar_size,     write_sum, net_size_scale )           # arrival rate prediction network
            self.load_map_net = LoadMapNet_bn( self.ar_size,    self.action_size, self.load_size, write_sum, net_size_scale )           # load mapping network
        else:
            self.critic_net   = CriticNet(  self.state_size, self.action_size, TAU, write_sum, net_size_scale )
            self.actor_net    = ActorNet(   self.state_size, self.action_size, TAU, write_sum, net_size_scale )
            self.ar_pred_net  = ARPredNet(  self.hisar_size, self.ar_size,     write_sum, net_size_scale )           # arrival rate prediction network
            self.load_map_net = LoadMapNet( self.ar_size,    self.action_size, self.load_size, write_sum, net_size_scale )           # load mapping network

        self.env = ENV( action_size, max_load=max_load, beta0=beta0 )

        #self.k_nearest_neighbors = int(max_actions * k_ratio )
        #Initialize Network Buffers:
        self.replay_memory_ac  = deque()
        self.replay_memory_arp = deque()
        self.replay_memory_lm  = deque()

        #Intialize time step:
        self.time_step = 0
        self.counter   = 0
        
        action_max    = np.ones(  ( self.action_size ) ).tolist()
        action_min    = np.zeros( ( self.action_size ) ).tolist()
        action_bounds = [action_max, action_min] 
        self.grad_inv = grad_inverter( action_bounds )
        
    def construct_state( self, pred_ar, pre_action=[] ):
        """Construct a state with the predicted ar and previous action
        """
        num_sbs    = np.max( pred_ar.shape )
        pred_ar    = np.reshape( np.array( pred_ar  ),   (1, num_sbs) )
        pre_action = np.reshape( np.array( pre_action ), (1, num_sbs) )
        state      = np.concatenate( (pred_ar, pre_action), axis=1 )
        return state.tolist()

    def evaluate_actor( self, state_t ):
        """Evaluate the actor network to get an action
        """
        p_action = self.actor_net.evaluate_actor( state_t )
        return p_action
    
    def add_experience_ac( self, state, next_state, action, reward ):
        """Add data sample of the Actor-Critic network
        """
        self.state      = state
        self.next_state = next_state
        self.action     = action
        self.reward     = reward
        #if reward>0:
        self.replay_memory_ac.append( (self.state, self.next_state, self.action, self.reward) )
        
        self.time_step = self.time_step + 1
        if( len(self.replay_memory_ac) > AC_REPLAY_MEMORY_SIZE ):
            self.replay_memory_ac.popleft()
            
  
    def add_experience_arp( self, his_ar, pred_ar ):
        """Add data sample of the arrival rate prediction network
        """
        self.replay_memory_arp.append( (his_ar, pred_ar) )
        if( len(self.replay_memory_arp) > ARP_REPLAY_MEMORY_SIZE ):
            self.replay_memory_arp.popleft()

    def add_experience_lm( self, ar_action, mapped_load ):
        """Add data sample of the load mapping network
        """
        self.replay_memory_lm.append( (ar_action, mapped_load) )
        if( len(self.replay_memory_lm) > LM_REPLAY_MEMORY_SIZE ):
            self.replay_memory_lm.popleft()

    def refine_action(self, state, action_in, imp = 0):
        """ round up the action to [0,1], then if imp>0,
        get the p_action's nearest neighbors, return the one with the max metric value, 
        imp==1, metric = Q value; 
        imp==2, metric = Q value + reward; 
        imp==3, metric = reward.
        """
        action0 = np.round( action_in )
        action  = np.clip(  action0, 0, 1 )
        #print("in refine action: "+str(action))
        if imp>0:
            action  = self.improve_action(state, action, imp)
        return action

    def improve_action(self, state, p_action, greedy=1):
        """ get the p_action's nearest neighbors, return the one with the max metric value
        greedy==1, metric = Q value
        greedy==2, metric = Q value + reward
        greedy==3, metric = reward
        """
        state0   = state[0]
        ac_size  = np.max(p_action.shape)
        ar_size  = len(state0)-ac_size
        p_action = np.array(p_action)

        # if the action would cause outage, greedily modify the action
        pred_ar     = state0[0:ar_size]      # predicted ar 
        pred_ar     = np.reshape( pred_ar,    [1, ar_size] )
        prev_action = state0[ac_size: ]
        #print("p_action: "+str(p_action))
        reward = -1
        while reward < 0:
            map_load    = self.load_map_net.evaluate_load_map( pred_ar, [p_action] )
            map_load[0][-1] += 0.05                      # conservatively estimate the load of the mbs
            reward, _, _, _, _ = self.env.find_reward( map_load[0], p_action, prev_action )
            #print("est reward: "+str(reward))
            if reward < 0:
                t_ar = [a*(1-b) for a,b in zip(pred_ar[0], p_action)]
                if max(t_ar)==0:
                    #print("---------------------tried best, still negative reward, break...")
                    break
                max_index  = np.argmax( t_ar )
                p_action[max_index] = 1
                #print("---------------------negative reward, change the "+str(max_index)+" action to 1")

        # find the nearest neighbors
        actions = np.zeros( (ac_size+1, ac_size) )
        for i in range(0, ac_size):
            t_action    = copy.deepcopy(p_action)
            t_action[i] = 1-t_action[i]
            actions[ i] = t_action
        actions[ac_size] = copy.deepcopy(p_action)
        
        metrics = np.zeros( ( ac_size+1 ) )
        if greedy <=2:
            # make all the (state, action) pairs for the critic
            states   = np.tile(state, [len(actions), 1])
            # evaluate each pair through the critic
            q_values = self.critic_net.evaluate_critic(states, actions)
            # find the index of the pair with the maximum value
            metrics += np.reshape( q_values, ( ac_size+1 ) )
            #print("q values: "+str(metrics))
        if greedy >=2:

            rewards = np.zeros( ( ac_size+1 ) )
            for i in range(0,ac_size+1):
                taction  = np.reshape( actions[i], [1, ac_size] )
                map_load = self.load_map_net.evaluate_load_map( pred_ar, taction )
                map_load[0][-1] += 0.02                      # conservatively estimate the load of the mbs
                rewards[i], _, _, _, _ = self.env.find_reward( map_load[0], actions[i], prev_action )
            metrics = rewards + GAMMA*metrics

        max_index  = np.argmax( metrics )   # 
        action_out = actions[max_index]
        #if max_index != ac_size:
        #    print("Improve "+str(p_action)+" to "+str(action_out))
        # return the best action
        return action_out




    
    def get_minibatch_ac(self):
        """Get mini batch for training of actor-critic network
        """
        batch = random.sample( self.replay_memory_ac, AC_BATCH_SIZE )
        #state t
        self.batch_states = [item[0] for item in batch]
        self.batch_states = np.reshape( np.array(self.batch_states), (AC_BATCH_SIZE, self.state_size ) )
        #state t+1        
        self.batch_next_states = [item[1] for item in batch]
        self.batch_next_states = np.reshape( np.array( self.batch_next_states), (AC_BATCH_SIZE, self.state_size ) )
        
        self.batch_actions = [item[2] for item in batch]
        self.batch_actions = np.reshape( np.array( self.batch_actions), [len(self.batch_actions), self.action_size] )
        
        self.batch_rewards = [item[3] for item in batch]
        self.batch_rewards = np.array( self.batch_rewards )


    
    def get_minibatch_arp(self):
        """Get mini batch for training of arrival rate prediction network
        """
        batch = random.sample( self.replay_memory_arp, ARP_BATCH_SIZE )
        #history ars
        self.his_ars = [item[0] for item in batch]
        #print("his_ars: "+str(np.array(self.his_ars)))
        self.his_ars = np.reshape( np.array(self.his_ars), (ARP_BATCH_SIZE, self.hisar_size) )
        #state t+1
        self.next_ars = [item[1] for item in batch]
        self.next_ars = np.reshape( np.array( self.next_ars), (ARP_BATCH_SIZE, self.ar_size ) )
    
    def get_minibatch_lm(self):
        """Get mini batch for training of load mapping network
        """
        batch = random.sample( self.replay_memory_lm, LM_BATCH_SIZE )
        #history ars
        self.ar_action = [item[0] for item in batch]
        self.ar_action = np.reshape( np.array(self.ar_action), (LM_BATCH_SIZE, self.ar_action_size) )
        #state t+1
        self.mapped_load = [item[1] for item in batch]
        self.mapped_load = np.reshape( np.array( self.mapped_load), (LM_BATCH_SIZE, self.load_size ) )
    
    
    def train_ac( self, learning_rate=[0.0001, 0.0001], update_target=0):
        """Train actor-critic network with a minibatch from the replay memory
        """
        cerror = 1
        aerror = 1
        if( len( self.replay_memory_ac ) > AC_BATCH_SIZE ):
            # Sample a random minibatch of N transitions from R
            self.get_minibatch_ac()
            self.batch_next_taction = self.actor_net.evaluate_target_actor( self.batch_next_states )

            # Q'(s_i+1,a_i+1)        
            batch_next_tQ = self.critic_net.evaluate_target_critic( self.batch_next_states, self.batch_next_taction ) 
            
            # r + gamma*Q'(s_i+1,a_i+1)     
            self.batch_next_obj_Q = []
            for i in range(0,AC_BATCH_SIZE):
                self.batch_next_obj_Q.append( self.batch_rewards[i] + GAMMA*batch_next_tQ[i][0] )        
        
            self.batch_next_obj_Q = np.array(   self.batch_next_obj_Q )
            self.batch_next_obj_Q = np.reshape( self.batch_next_obj_Q, [len(self.batch_next_obj_Q),1] )
            #print("tQ: "+str(np.reshape( batch_next_tQ, [1, len(batch_next_tQ)]) ) )
            # Update critic by minimizing the loss
            cerror = self.critic_net.train_critic(self.batch_states, self.batch_actions, self.batch_next_obj_Q, learning_rate[1])

            # Find gradients from the Q values (critic network) to the actions
            action_for_grad_Q2a = self.evaluate_actor(self.batch_states)
            if is_grad_inverter:
                self.grad_Q2a = self.critic_net.compute_grad_Q2a( self.batch_states, action_for_grad_Q2a )#/AC_BATCH_SIZE            
                self.grad_Q2a = self.grad_inv.invert( self.grad_Q2a, action_for_grad_Q2a )
            else:
                self.grad_Q2a = self.critic_net.compute_grad_Q2a( self.batch_states, action_for_grad_Q2a )[0]#/AC_BATCH_SIZE

            # Train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
            aerror = self.actor_net.train_actor(self.batch_states, self.batch_actions, self.grad_Q2a, learning_rate[0])
            #print("aerror: "+str(aerror))
            if update_target == 1:
                self.update_target_net()
        return cerror, aerror

    def update_target_net( self ):
        # Update target Critic and Actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()

    def train_arp( self, learning_rate=0.0001):
        """Train the arrival rate prediction network with a minibatch from the replay memory
        """
        lrm = len( self.replay_memory_arp )
        arperror = 1
        if( lrm >= ARP_BATCH_SIZE ):
            #print('in train_arp, lrm: '+str(lrm))
            # Sample a random minibatch of N transitions from R
            self.get_minibatch_arp()
            # Train ar prediction network

            arperror = self.ar_pred_net.train_ar_pred(self.his_ars, self.next_ars, learning_rate)
            #print('in train_arp, arperror: '+str(arperror))
        return arperror

    def train_lm( self, learning_rate=0.0001):
        """Train the load mapping network with a minibatch from the replay memory
        """
        lrm = len( self.replay_memory_lm )
        lmerror = 1
        if( lrm >= LM_BATCH_SIZE ):
            # Sample a random minibatch of N transitions from R
            self.get_minibatch_lm()
            # Train load mapping network
            lmerror = self.load_map_net.train_load_map(self.ar_action, self.mapped_load, learning_rate)
        return lmerror

    def find_action_neigh( self, state, p_action ):
        # get the proto_action's k nearest neighbors
        actions = self.action_space.search_point(p_action, self.k_nearest_neighbors)[0]
        # make all the state, action pairs for the critic
        states = np.tile(state, [len(actions), 1])
        # evaluate each pair through the critic
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        # find the index of the pair with the maximum value
        max_index = np.argmax(actions_evaluation)
        # return the best action
        return actions[max_index]


    def close_all(self):
        self.actor_net.close_all()
        self.critic_net.close_all()
        self.ar_pred_net.close_all()
        self.load_map_net.close_all()

    
    def decay(self, i, minr, maxr, istep=1, estep=5000, method=1):
        """
        method=1: log
        method=2: linear
        method=3: inverse
        """
        if method == 1:     #log
            shift = (estep)**(-minr/maxr)
            scale = maxr*math.log(istep+shift)
            a = scale/(math.log(i +shift))
        if method == 2:     #linear
            scale = (maxr-minr)/(istep-estep)
            shift = maxr-scale*istep
            a = scale*i + shift
        if method == 3:      #inverse
            shift = (estep*minr-istep*maxr)/(maxr-minr)
            scale = maxr*(istep+shift)
            a = scale/(i+shift)
        return max(0, a)