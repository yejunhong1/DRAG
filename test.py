import scipy.io as sio 
import numpy as np
from agent import DDPG
from ou_noise import OUNoise
from environment import ENV
import matplotlib.pyplot as plt
from statistics import mean
import tensorflow as tf

x=tf.nn.tanh([-5.,5.])

pre_action = np.ones((10))
env        = ENV(10)
load=[0.578754399631581,0.671189190577800,0.813574914356523,0.795020116500634,0.681940967465423,0.688315468525857,0.870533944883698,0.702414579585091,0.684756134468678,0.865244808360227]
reward = env.find_reward( load, [ 0.,  1.,  1., -0.,  1.,  1.,  1.,  1.,  1.,  0.], pre_action )


print("reward is: "+str(reward))
print("x is: "+str(tf.InteractiveSession().run(x)))
