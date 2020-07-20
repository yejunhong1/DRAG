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
is_batch_norm = False            # key for batch normalization
eps_greedy    = 0
epsilon       = 0.3
write_sum     = 0
#net_scale     = 1

def arp_pred_net( AR, num_his_ar=4, funname='', net_scale=1 ):

    num_sbs, _ = AR.shape
    num_ts = 10000
    action_size = num_sbs
    ar_size     = num_sbs
    his_ar_size = ar_size * num_his_ar
    print( "Size of history ar: "  + str(his_ar_size) )

    arp_errors  = [1]            # average error in the predicted arrival rate

    # Randomly initialize critic, actor, target critic, target actor and load prediction network and replay buffer, in the agent
    agent = DDPG( his_ar_size, ar_size, action_size, TAU, is_batch_norm, write_sum, net_size_scale=net_scale )
 
    for i in range( num_his_ar, num_ts+num_his_ar ):
        his_ar  = np.reshape( AR[:,i-num_his_ar:i], (1, his_ar_size) , order='F' )

        real_ar = AR[:,i]
        agent.add_experience_arp( his_ar, real_ar )

        # Train ar prediction network, after many num_ts, one minibatch is enough for each step
        arp_error = 1
        arp_train_times = min(10, max(1, int(i/ARP_BATCH_SIZE)) ) #if i<1000 else 5
        lr = max(ARP_LR_MIN, agent.decay(i, ARP_LR_MIN, ARP_LR_MAX, num_his_ar, 8000, 2) )
        for j in range( 0, arp_train_times ):
            arp_errort = agent.train_arp( lr )     #/math.log(i+2)
            #print('arp_errort: ' + str(arp_errort))
            if arp_errort !=1:
                arp_error = arp_errort

        if arp_error !=1:
            arp_errors.append( math.sqrt( arp_error ) )

        if i%(100) == 0:
            print('    i: ' + str(i) + ', arp_error: ' + str(math.sqrt( arp_error )))
    return arp_errors

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