#-*-coding:utf-8-*-
# An actor-critic reinforcement learning framewrok to control the on/off mode of small base stations in heterogeneous network
# 
# Based on the implementation of "Deep Deterministic Gradient with Tensor Flow" 
# by Steven Spielberg Pon Kumar (github.com/stevenpjg)
# Author: YE Junhong @IE, CUHK, 17, May 2018
import os
import shutil
import math
import random
import numpy as np

from agent import DDPG
from ou_noise import OUNoise
from config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Don't limit print width
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

# Specify parameters here:
is_batch_norm = True            # key for batch normalization
num_his_ar    = 4
eps_greedy    = 0
epsilon       = 0.3

def dqn_bsa( AR, ac_ref=4, write_sum=0, net_scale=1, funname='', beta0=beta ):

    num_sbs, num_ts = AR.shape

    num_mbs     = 1
    ts_per_day  = 48

    ar_size     = num_sbs
    his_ar_size = ar_size * num_his_ar
    load_size   = num_sbs + num_mbs
    action_size = num_sbs
    state_size  = ar_size + action_size
    print( "Size of history ar: "  + str(his_ar_size) )
    print( "Size of action: "      + str(action_size) )
    print( "Number of timeslots: " + str(num_ts) )
    
    rewards      = np.zeros( (num_ts) )            # reward of each timeslot
    sum_powers   = np.zeros( (num_ts) )
    switch_powers = np.zeros( (num_ts) ) 
    qos_costs    = np.zeros( (num_ts) ) 
    throughputs  = np.zeros( (num_ts) )
    prev_action  = np.ones(  (num_sbs) )    

    arp_errors  = [1]            # average error in the predicted arrival rate
    lm_errors   = [1]            # average error in the mapped load
    c_errors    = [1]            # average error in the Q values (critic network output)
    a_errors    = [1]            # average error in the action output

    # Randomly initialize critic, actor, target critic, target actor and load prediction network and replay buffer, in the agent
    agent = DDPG( his_ar_size, ar_size, action_size, TAU, is_batch_norm, write_sum, net_size_scale=net_scale, beta0=beta0 )
    exploration_noise = OUNoise( num_sbs )

    for i in range( num_his_ar, num_ts ):
        his_ar  = np.reshape( AR[:,i-num_his_ar:i], (1, his_ar_size) , order='F' )
        pred_ar = agent.ar_pred_net.evaluate_ar_pred( his_ar )

        # Generate a state_ac of the AC network
        state_ac  = agent.construct_state( pred_ar, prev_action )    #

        if eps_greedy:
            # epsilon-greedy based exploration
            if random.uniform(0, 1) < epsilon/i:#math.log(i+2):     #
                sigmai = 0.3#/math.log(i+1)
                action = exploration_noise.noisei( 0.0, sigmai )
            else:
                action = agent.evaluate_actor( state_ac )
                action = action[0]
        else:
            # noise-based exploration
            action = agent.evaluate_actor( state_ac )[0]
            sigmai = agent.decay(i, 0.01, 0.5, num_his_ar, num_ts/2, 2)
            noise  = exploration_noise.noisei( 0, sigmai )      #0.5/math.log(i+2)
            action = action + noise

        # Refine the action, including rounding to 0 or 1, and greedy exploration
        if ac_ref<=3:
            action = agent.refine_action( state_ac, action, ac_ref )
        else:   # hybrid method
            if random.uniform(0, 1) < agent.decay(i, 0, 3, num_his_ar, num_ts*0.75, 2):
                action = agent.refine_action( state_ac, action, 3 )       # refine the action
            else:
                action = agent.refine_action( state_ac, action, 2 )

        # after taking the action and the env reacts
        #pred_load = agent.load_map_net.evaluate_load_map( pred_ar, np.reshape( action, [1, action_size] ) )
        real_ar   = AR[:,i]
        real_load = agent.env.measure_load( real_ar, action )
        reward, sum_power, switch_power, qos_cost, throughput = agent.env.find_reward( real_load, action, prev_action )   #

        next_his_ar   = np.reshape( AR[:, i-num_his_ar+1:i+1], (1, his_ar_size) , order='F' )
        next_pred_ar  = agent.ar_pred_net.evaluate_ar_pred( next_his_ar )
        next_state_ac = agent.construct_state( next_pred_ar, action )    #

        # Add s_t, s_t+1, action, reward to experience memory
        ar_action = np.concatenate([real_ar, action])
        agent.add_experience_ac(  state_ac,  next_state_ac, action, reward )
        agent.add_experience_arp( his_ar,    real_ar )
        agent.add_experience_lm(  ar_action, real_load )

        # Train critic and actor network, maybe multiple minibatches per step
        a_lr = max(A_LR_MIN, agent.decay(i, A_LR_MIN, A_LR_MAX, num_his_ar, 8000, 2) ) #max( AC_LR_MIN[0], AC_LR_MAX[0]/math.log2(i+1) )
        c_lr = max(C_LR_MIN, agent.decay(i, C_LR_MIN, C_LR_MAX, num_his_ar, 8000, 2) ) #max( AC_LR_MIN[1], AC_LR_MAX[1]/math.log2(i+1) )
        learning_rate = [ a_lr, c_lr ]

        cerror = 1
        aerror = 1
        ac_train_times = min(16, max(1, int(i/500)) )
        for j in range( 0, ac_train_times ):    #                  #between 1 and 5
            cerrort, aerrort = agent.train_ac( learning_rate, 1 )
            if cerrort !=1:
                cerror = cerrort
                aerror = aerrort
        
        # Train ar prediction network, after many num_ts, one minibatch is enough for each step
        arp_error = 1
        arp_train_times = min(10, max(1, int(i/ARP_BATCH_SIZE)) ) #if i<1000 else 5
        lr = max(ARP_LR_MIN, agent.decay(i, ARP_LR_MIN, ARP_LR_MAX, num_his_ar, 8000, 2) )
        for j in range( 0, arp_train_times ):
            arp_errort = agent.train_arp( lr )     #/math.log(i+2)
            if arp_errort !=1:
                arp_error = arp_errort
        
        # Train load mapping network, after many num_ts, one minibatch is enough for each step
        lm_error = 1
        lm_train_times = min(10, max(1, int(i/LM_BATCH_SIZE)) ) #if i<1000 else 20
        lr = max(LM_LR_MIN, agent.decay(i, LM_LR_MIN, LM_LR_MAX, num_his_ar, 8000, 2) )
        for j in range( 0, lm_train_times ):
            lm_errort = agent.train_lm( lr )   #
            if lm_errort !=1:
                lm_error = lm_errort

        if arp_error !=1:
            arp_errors.append( math.sqrt( arp_error ) )
        if lm_error !=1:
            lm_errors.append( math.sqrt( lm_error ) )
        if cerror !=1:
            c_errors.append( math.sqrt( cerror ) )
        if aerror !=1:
            a_errors.append( aerror*num_sbs )         # hamming distance error

        prev_action = action
        rewards[i]  = reward
        sum_powers[i]  = sum_power
        throughputs[i]  = throughput
        switch_powers[i]  = switch_power
        qos_costs[i]  = qos_cost

        if i%(ts_per_day) == 0:
            mrt = np.mean( rewards[i-ts_per_day:i] )
            if write_sum>0:
                print(funname + " ------- i: %5d, arp-e: %1.5f, lm-e: %1.5f, a-e: %1.5f, c-e: %1.5f, d-reward: %1.5f \n" % ( i, arp_errors[-1], lm_errors[-1], a_errors[-1], c_errors[-1], mrt ) )
            else:
                print(funname + " ------- i: %5d, mean reward: %1.5f \n" % ( i, mrt ) )

    return rewards, sum_powers, switch_powers, qos_costs, throughputs

def writetext( array, name, ismatrix=0 ):
    """Write list or arrays/matrices into .txt files.
    """
    thefile = open( name + '.txt', 'w' )
    for item in array:
        s=str(item)
        if ismatrix==1:
            thefile.writelines( "%s\n" %  s[1:-1] )     # Delete the brackets [ and ]
        else:
            thefile.write( "%s\n" % s )
    thefile.close()