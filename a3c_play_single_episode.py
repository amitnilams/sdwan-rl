import gym
import gym_sdwan
import argparse
import numpy as np
import os

from tensorflow.python import keras
from tensorflow.python.keras import layers

import tensorflow as tf
tf.enable_eager_execution()


ENV_NAME = 'Sdwan-v0'

class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)
        

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
#         print ('Call - input shape', inputs.shape)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
#         print ('logits shape', logits.shape)
#         print ('values shape', values.shape)
        return logits, values

class A3CPlayer:
    def __init__(self, state_size, action_size, save_dir, model_file):               
        self.state_size = state_size
        self.action_size = action_size

        self.model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        
        model_path = os.path.join(save_dir, model_file)
        print('Loading model from: {}'.format(model_path))
        self.model.load_weights(model_path)
           
    def act(self, state):
        policy, value = self.model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        action = np.argmax(policy)
        print ("Taking predicted  action", action)
        return action


def main(args):
    env = gym.make(ENV_NAME)
    env.MAX_TICKS  = args.n_max_ticks
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    a3c_player = A3CPlayer(state_size, action_size, args.save_dir, args.model_file)
 
    step = 0
    score = 0
    error = False
      
    state = env.reset()
    state = np.reshape(state, [state_size,])
        
    
    print('Initial State:', state)
    while True:
        step += 1
        action = a3c_player.act(state)
        state, reward, done, _ = env.step(action)
        state = np.reshape(state, [state_size,])
        score += reward
        print('Ticks:', step, 'Action:', action, 'Ob:', state, 'R:', 
        reward, 'Total Reward:', score)

        if done:
            print("Episode Aborted  after {} timesteps".format(step))
            break
       
    env.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('A3C Agent')
    parser.add_argument(
        '--n-max-ticks',
        type=int,
        default=30)
    parser.add_argument(
        '--model-file',
        type=str,
        default="model.h5")
    parser.add_argument(
        '--save-dir',
        default='./', 
        type=str,
        help='Directory in which you desire to save the model.')
    
    args = parser.parse_args()

    main(args)


