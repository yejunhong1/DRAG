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

from arp_pred_net_test import *

# Specify parameters here:
days      = 2000
write_sum = 0              # key/interval for writing summary data to file
#noise_method = 2                # 0: no noise, 1: independent noise, 2: correlated noise, 3: scaling noise
#pattern_type = 1                # 1: stationary, 2: slowly varying, 3: suddently varying
directory = "./traffic_pattern/"
host      = "_"+socket.gethostname()


def runs(pattern_type, m, shift=0, num_his_ar=1, funname=''):
    noise_type = 2
    patterns   = ['sp','svp','fvp']
    noise_str  = ['0','1_0.08','2_0.05_0.03','3_0.05']
    data_file  = directory+'AR_'+patterns[pattern_type-1]+'_n_' + str(20) + '_d_2000_nm_'+noise_str[noise_type] # 'Simulated_arrival_rates_no_noise_' + str(days)   #
    
    AR0 = sio.loadmat( data_file )['AR']
    AR  = AR0[0:m, shift+0 : shift+20000]
    funname = funname + '_' + noise_str[noise_type]
    net_scale = num_his_ar * 0.1
    arp_errors = arp_pred_net( AR, num_his_ar, funname, net_scale )

    writetext( arp_errors, './arpn_test/ACDQN_arp_err_' + funname + host )

    return 1


if __name__ == '__main__':
    m=10
    for si in range(1): # si<=8
        for ni in range(1,11): #[0]
            num_his_ar = ni
            shift = int(si*1000)
            for pattern_type in [1]: #[0]
                funname = 'traffic_pattern_m_' +str(m) + '_num_his_ar_' + str(num_his_ar)
                print( "Begin " + funname + " at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
                runs( pattern_type, m, shift, num_his_ar, funname)
                print( "Finish " + funname + " at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )


