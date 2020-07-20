#-*-coding:utf-8-*-
# An actor-critic reinforcement learning framewrok to control the on/off mode of small base stations in heterogeneous network
# 
# Based on the implementation of "Deep Deterministic Gradient with Tensor Flow" 
# by Steven Spielberg Pon Kumar (github.com/stevenpjg)
# Author: YE Junhong @IE, CUHK, 04, Feb. 2018
import os
import shutil
import math
import random
import time

import tensorflow as tf
import scipy.io as sio
import numpy as np
#import matplotlib.pyplot as plt

from statistics import mean
from agent import DDPG
from ou_noise import OUNoise
from config import *

shutil.rmtree('./model/cn',   ignore_errors=True)
shutil.rmtree('./model/an',   ignore_errors=True)
shutil.rmtree('./model/arpn', ignore_errors=True)
shutil.rmtree('./model/lmn',  ignore_errors=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Don't limit print width
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

# Specify parameters here:
is_batch_norm = True            # key for batch normalization
days          = 2000
num_his_ar    = 4
eps_greedy    = 0
epsilon       = 0.3
write_sum     = 100              # key/interval for writing summary data to file
soft_update   = 1               # If use soft update
ac_ref1 = 3                     # initial action refinement method
ac_ref2 = 3                     # final action refinement method
#noise_method = 2                # 0: no noise, 1: independent noise, 2: correlated noise, 3: scaling noise
#pattern_type = 1                # 1: stationary, 2: slowly varying, 3: suddently varying
directory = "./net_scale/"

def runs(net_size_scale):
    patterns   = ['sp','svp','fvp']
    noise_para = ['0','1_0.05','2_0.05_0.03','3_0.05']
    data_file =  directory + 'AR_sp_n_30_d_2000_nm_2_0.05_0.03' # 'Simulated_arrival_rates_no_noise_' + str(days)   #
    samples   = sio.loadmat( data_file )
    arrival_rates = samples['AR']

    num_sbs, steps = arrival_rates.shape
    steps = 10000                                 # The number of steps (can freely modify)

    if soft_update == 1 :
        tui  = 1                     # Target network update interval
        TAUt = TAU
    else:
        tui  = 100
        TAUt = 1
    ar_size       = num_sbs
    his_ar_size   = ar_size * num_his_ar
    load_size     = ar_size + 1
    action_size   = num_sbs
    state_size    = num_sbs * 2
    print( "Size of history ar: "  + str(his_ar_size) )
    print( "Size of action: "      + str(action_size) )
    print( "Number of timeslots: " + str(steps) )
    
    rewards     = np.zeros( (steps) )            # reward of each timeslot
    mean_reward = np.zeros( ( int(steps/48) + 1 ) )
    actions     = np.zeros( (steps, num_sbs) )   # refined action
    actions_o   = np.zeros( (steps, num_sbs) )   # original/raw output action of the actor network
    prev_action = np.ones(  (num_sbs) )    
    pred_ars    = np.zeros( (steps, num_sbs) )   # predicted arrival rates of the next timeslot
    real_loads  = np.zeros( (steps, num_sbs+1) )
    pred_loads  = np.zeros( (steps, num_sbs+1) )

    arp_errors  = [1]            # average error in the predicted arrival rate
    lm_errors   = [1]            # average error in the mapped load
    c_errors    = [1]            # average error in the Q values (critic network output)
    a_errors    = [1]            # average error in the action output

    
    # Randomly initialize critic, actor, target critic, target actor and load prediction network and replay buffer, in the agent
    agent = DDPG( his_ar_size, ar_size, action_size, TAUt, is_batch_norm, write_sum, net_size_scale )
    exploration_noise = OUNoise( num_sbs )

    for i in range( num_his_ar, steps ):
        #print("i: "+str(i))
        his_ar  = np.reshape( arrival_rates[:,i-num_his_ar:i], (1, his_ar_size) , order='F' )
        pred_ar = agent.ar_pred_net.evaluate_ar_pred( his_ar )
        #real_ar = np.array( arrival_rates[:,i] )
        #print("his_ar: "+str(his_ar))
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
            sigmai = max(0, agent.decay(i, 0.01, 0.5, num_his_ar, 3000, 2) )
            #action = [ 1-a if random.uniform(0, 1)<sigmai else a for a in action ]
            noise  = exploration_noise.noisei( 0, sigmai )      #0.5/math.log(i+2)
            action = action + noise
        actions_o[i] = action

        # Refine the action, including rounding to 0 or 1, and greedy exploration
        if i<3000:
            action = agent.refine_action( state_ac, action, ac_ref1 )       # refine the action
        else:
            action = agent.refine_action( state_ac, action, ac_ref2 )

        # after taking the action and the env reacts
        #print("action_o: "+str(actions_o[i])+", action"+str(action))
        #print("pred_ar: "+str(pred_ar))
        pred_load = agent.load_map_net.evaluate_load_map( pred_ar, np.reshape( action, [1, action_size] ) )
        real_ar   = arrival_rates[:,i]
        real_load = agent.env.measure_load( real_ar, action )
        #print("pred_load: "+str(pred_load))
        #print("real_load: "+str(real_load))
        reward    = agent.env.find_reward( real_load, action, prev_action )   #
        #print("real reward: "+str(reward))
        next_his_ar   = np.reshape( arrival_rates[:, i-num_his_ar+1:i+1], (1, his_ar_size) , order='F' )
        next_pred_ar  = agent.ar_pred_net.evaluate_ar_pred( next_his_ar )
        next_state_ac = agent.construct_state( next_pred_ar, action )    #

        # Add s_t, s_t+1, action, reward to experience memory
        #print("real_ar: "+str(real_ar) + "action: "+str(action))
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
            cerrort, aerrort = agent.train_ac( learning_rate, soft_update )
            if cerrort !=1:
                cerror = cerrort
                aerror = aerrort

        if ( (i%tui == 0) and (soft_update==0) ):
            agent.update_target_net()
        
        # Train ar prediction network, after many steps, one minibatch is enough for each step
        arp_error = 1
        arp_train_times = min(10, max(1, int(i/ARP_BATCH_SIZE)) ) #if i<1000 else 5
        lr = max(ARP_LR_MIN, agent.decay(i, ARP_LR_MIN, ARP_LR_MAX, num_his_ar, 8000, 2) )
        for j in range( 0, arp_train_times ):
            arp_errort = agent.train_arp( lr )     #/math.log(i+2)
            if arp_errort !=1:
                arp_error = arp_errort
        
        # Train load mapping network, after many steps, one minibatch is enough for each step
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
        pred_ars[i] = pred_ar
        real_loads[i] = real_load
        pred_loads[i] = pred_load
        actions[i]  = action
        rewards[i]  = reward
        if i%(48) == 0:
            mean_reward[int(i/48)] = mean( rewards[i-48:i] )
            print("==== i: %5d, arp error: %1.5f, lm error: %1.5f, a error: %1.5f, c error: %1.5f, mean reward: %1.5f \n" % ( i, arp_errors[-1], lm_errors[-1], a_errors[-1], c_errors[-1], mean_reward[int(i/48)] ) )

    agent.close_all()      # this will write network parameters into .txt files


    writetext( rewards,  directory + 'ACDQN_rewards_net_size_scale_n_'+str(num_sbs)+'_scale_' + str(net_size_scale)  )

    return 1



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

if __name__ == '__main__':
    for m in range(2,13,1):
        scale = m/10
        runs(scale)
        print( "Finish scale = " + str(scale) + " at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )



