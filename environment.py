

import numpy as np 
from config import *
import math

class ENV:
    """Environment part of the DRL algorithm (compute reward etc.)
    num_sbs: size of the action vector/tensor
    """
    def __init__(self, num_sbs, max_load=1, beta0=thr_coe):
        
        self.Pc  = Pc
        self.Pl  = Pl
        self.Ps  = Ps
        self.Pl0 = Pl0
        self.Cm  = Cm
        self.Cmc = Cm*1
        self.alpha = tmf[0:num_sbs]/Cm
        self.beta  = beta*np.ones((num_sbs+1))
        self.beta[-1] = beta*Cm
        self.factor = self.beta/beta
        self.thr_coe = beta0
        self.num_sbs = num_sbs
        self.Pci = np.multiply(self.Pc, np.ones((num_sbs)))
        self.Pli = np.multiply(self.Pl, np.ones((num_sbs)))
        self.MAX_LOAD = max_load
        self.ref_cost, _, _, _, _ = self.find_cost(0.5*np.ones((num_sbs+1)), np.ones((num_sbs)), np.zeros((num_sbs)) )
        if self.MAX_LOAD==1:
            self.MAX_LOAD = 1 - self.beta[-1]/(4*self.ref_cost)   # set the max load such that any load>1 would lead to a negative reward < -0.5
        print("beta0: " + str(beta0))
        print("thr_coe:  " + str(self.thr_coe))
        print("max load: " + str(self.MAX_LOAD))
        print("reference cost: " + str(self.ref_cost))

        #print("beta: " + str(self.beta))

    def find_cost(self, load0, action0, pre_action0, log=0 ):
        #print("load0: "+str(load0), "self.MAX_LOAD: "+str(self.MAX_LOAD))
        load0      = np.clip(load0, 0, self.MAX_LOAD)
        load       = load0[0:self.num_sbs]
        mbs_load   = load0[self.num_sbs]
        load       = np.array(load)
        action     = np.array(action0)
        pre_action = np.array(pre_action0)
        t_action   = action - pre_action
        pt_action  = [v if v>0 else 0 for v in t_action]
        
        #print("load: "+str(load), "action: "+str(action))
        power_sbs      = sum( np.multiply( self.Pci + np.multiply(self.Pli, load), action ) )
        penalty_switch = np.multiply( self.Ps , sum(pt_action) )
        power_mbs      = np.multiply( self.Pl0 , mbs_load )
        sum_power      = power_sbs + power_mbs
        penalty_delay  = sum( [a/(1-a)*b for a,b in zip(load0, self.beta)] )
        throughput     = sum( [a*b for a,b in zip(load0, self.factor)] )
        thr_cost       = np.multiply( self.thr_coe, throughput )
        cost           = sum_power + thr_cost + penalty_switch
        #print("load0: "+str(load0))
        #print("throughput: "+str(throughput))

        if log==1:
            
            print("switch: "+str(penalty_switch))
            print("mbs: "+str(power_mbs))
            print("mbs load: "+str(load0[-1]))
            print("sum power: "+str(sum_power))
            print("penalty: "+str(penalty_delay))
        return cost, sum_power, penalty_switch, penalty_delay, throughput

    def find_reward(self, load0, action0, pre_action0, log=0 ):
        cost, sum_power, penalty_switch, penalty_delay, throughput   = self.find_cost(load0, action0, pre_action0, log)
        #print("cost: "+str(cost))
        reward = (self.ref_cost - cost) / abs(self.ref_cost)
        return reward, sum_power, penalty_switch, penalty_delay, throughput


    def measure_load(self, ar, action):
        #print("ar: "+str(ar) + ", action: " + str(action))
        ar_size = max(ar.shape)
        #print("ar_size: " + str(ar_size))
        load  = np.zeros((ar_size+1))
        for i in range(ar_size):
            load[i] = self.map_load(ar[i]) * action[i] 
        mar   = sum( [a*b*(1-c) for a,b,c in zip(ar, self.alpha, action)] )
        mload = self.map_load(mar)
        load[ar_size] = mload
        #load = np.clip(load, 0, MAX_LOAD)
        return load

    def map_load(self, ar):
        #load = 2*( 1/(1+math.exp(-ar*6)) - 0.5 )
        load = ar
        return load

