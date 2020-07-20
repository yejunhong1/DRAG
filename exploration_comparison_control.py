#-*-coding:utf-8-*-
# An actor-critic reinforcement learning framewrok to control the on/off mode of small base stations in heterogeneous network
# 
# Based on the implementation of "Deep Deterministic Gradient with Tensor Flow" 
# by Steven Spielberg Pon Kumar (github.com/stevenpjg)
# Author: YE Junhong @IE, CUHK, 17, May 2018
import os
import time
import socket

import scipy.io as sio
import numpy as np

from dqn_bsa import *

# Specify parameters here:
days      = 2000
write_sum = 0              # key/interval for writing summary data to file
directory = "./action_refine/"
host      = "_"+socket.gethostname()
num_ts    = 20000

def runs(ac_ref, m, shift=0, r=1, funname=''):
    data_file = directory + 'AR_sp_n_' + str(m) + '_d_2000_nm_2_0.05_0.03' # 'Simulated_arrival_rates_no_noise_' + str(days)   #

    AR0 = sio.loadmat( data_file )['AR']
    AR  = AR0[0:m, shift+0 : shift+num_ts]

    rewards = dqn_bsa( AR, ac_ref=ac_ref, write_sum=write_sum, net_scale=1, funname=funname )

    writetext( rewards, directory + 'ACDQN_rewards_' + funname + host  )

    return 1


if __name__ == '__main__':
    m=10
    for si in range(1): # si<=8
        shift = int(si*1000)
        for r in range(10): #[0]
            for ac_ref in range(5): #[0]
                funname = 'action_refine_m_' +str(m) +  '_ac_ref_' + str(ac_ref) + '_s_' +str(shift) + '_r_' + str(r)
                runs( ac_ref, m, shift, r, funname)
                print( "Finish " + funname + " at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )


