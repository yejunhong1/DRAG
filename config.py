#-*-coding:utf-8-*-
"""
Configuration Parameters of the system
"""
import scipy.io as sio
import tensorflow as tf
is_grad_inverter = True

TAU   = 0.0001    # update rate of target networks

GAMMA = 0.5      # discount factor in reinforcement learning

A_LR_MAX = 0.005
A_LR_MIN = 0.0008    # good around 0.0006 for greedy reward

C_LR_MAX = 0.002
C_LR_MIN = 0.0004    # good around 0.0006 for greedy reward

ARP_LR_MAX = 0.002
ARP_LR_MIN = 0.0001

LM_LR_MAX  = 0.002
LM_LR_MIN  = 0.0001

AC_BATCH_SIZE  = 128
ARP_BATCH_SIZE = 64
LM_BATCH_SIZE  = 64

AC_REPLAY_MEMORY_SIZE  = 6000
ARP_REPLAY_MEMORY_SIZE = 4000
LM_REPLAY_MEMORY_SIZE  = 4000


ntanh  = lambda x, name=[]: ( tf.nn.tanh( x+2 ) + 1 )/2
linear = lambda x, name=[]: x


AN_N_HIDDENS   = [200, 100]
AN_ACTS        = [tf.nn.softplus, tf.nn.relu, ntanh]

CN_N_HIDDENS   = [200, 100]
CN_ACTS        = [tf.nn.softplus, tf.nn.relu, linear]

ARPN_N_HIDDENS = [200, 100]
ARPN_ACTS      = [tf.nn.tanh, tf.nn.tanh, tf.nn.sigmoid]

LMN_N_HIDDENS  = [200, 100]
LMN_ACTS       = [tf.nn.tanh, tf.nn.tanh, tf.nn.sigmoid]
"""

AN_N_HIDDENS   = [200, 100, 100]
AN_ACTS        = [tf.nn.softplus, tf.nn.relu, tf.nn.relu, ntanh]

CN_N_HIDDENS   = [200, 100, 100]
CN_ACTS        = [tf.nn.softplus, tf.nn.relu, tf.nn.relu, linear]

ARPN_N_HIDDENS = [200, 100, 100]
ARPN_ACTS      = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.sigmoid]

LMN_N_HIDDENS  = [200, 100, 100]
LMN_ACTS       = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.sigmoid]
"""

# Hetnet-related settings
Pc  = 160#500
Pl  = 216#10
Pl0 = 1080#100
Ps  = 100
Cm  = 3.0       #Cm needs to be float in compatible with python 2.7
beta = 50
thr_coe = -400
MAX_LOAD = 0.99

tmfs = sio.loadmat( 'tmfs' )['tmfs']
tmf = tmfs[:,1]
# 10: 0.99, 20: 0.993, 30: 0.996