# Parameters of the system

AC_LR_MAX = [0.0008, 0.0006]
AC_LR     = [0.0003, 0.0002]                  # good around 0.0006 for greedy reward
LP_LR     = 0.0002

TAU       = 0.0006

GAMMA     = 0.5
is_grad_inverter = True

AC_BATCH_SIZE = 128
LP_BATCH_SIZE = 64

AC_REPLAY_MEMORY_SIZE = 6000
LP_REPLAY_MEMORY_SIZE = 4000


AN_N_HIDDEN_1 = 200
AN_N_HIDDEN_2 = 100

CN_N_HIDDEN_1 = 200
CN_N_HIDDEN_2 = 60

LPN_N_HIDDEN_1 = 200
LPN_N_HIDDEN_2 = 100

# Hetnet-related settings
Pc  = 500
Pl  = 10
Ps  = 20
Pl0 = 40
Cm  = 2
beta  = 5000


