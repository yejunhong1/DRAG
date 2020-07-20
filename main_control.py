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
#noise_method = 2                # 0: no noise, 1: independent noise, 2: correlated noise, 3: scaling noise
#pattern_type = 1                # 1: stationary, 2: slowly varying, 3: suddently varying
directory = "./convergence/"
host      = "_"+socket.gethostname()
num_ts    = 12000


def runs(beta0, m, shift=0, r=1, funname=''):
    patterns   = ['sp','svp','fvp']
    noise_para = ['0','1_0.05','2_0.05_0.03','3_0.05']
    pattern_type = 1
    data_file  = './traffic_pattern/AR_'+patterns[pattern_type-1]+'_n_' + str(m) + '_d_2000_nm_2_0.05_0.03' # 'Simulated_arrival_rates_no_noise_' + str(days)   #
    
    AR0 = sio.loadmat( data_file )['AR']
    AR  = AR0[0:m, shift+0 : shift+num_ts]

    rewards, sum_powers, switch_powers, qos_costs, throughputs  = dqn_bsa( AR, ac_ref=4, write_sum=write_sum, net_scale=1, funname=funname, beta0=beta0 )

    writetext( rewards,       directory + 'ACDQN_rewards_'       + funname + host  )
    writetext( sum_powers,    directory + 'ACDQN_sum_powers_'    + funname + host  )
    writetext( switch_powers, directory + 'ACDQN_switch_powers_' + funname + host  )
    writetext( qos_costs,     directory + 'ACDQN_qos_costs_'     + funname + host  )
    writetext( throughputs,   directory + 'ACDQN_throughputs_'   + funname + host  )
    
    matdict = {}
    matdict['rewards']       = rewards
    matdict['sum_powers']    = sum_powers
    matdict['switch_powers'] = switch_powers
    matdict['qos_costs']     = qos_costs
    matdict['throughputs']   = throughputs
    sio.savemat( directory + 'ACDQN_' + funname + host + '.mat', matdict )
    
    return 1


if __name__ == '__main__':
    m = 10
    betas = range(400, 1100, 100)
    for si in range(1): # si<=8
        shift = int(si*1000)
        for r in range(10): #[0]
            for beta in betas: #[0]
                beta0 = -beta
                funname = 'convergence_m_' +str(m) +  '_beta_' + str(beta0) + '_s_' +str(shift) + '_r_' + str(r)
                runs( beta0, m, shift, r, funname)
                print( "Finish " + funname + " at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )


