#!/usr/bin/env python
# coding: utf-8

import os
import gym
import gym_sdwan_stat
import numpy as np
import argparse
import csv

from tensorflow.python import keras
from tensorflow.python.keras import layers

import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt 

LEARNING_RATE = 0.001
ENV_NAME = "Sdwan-stat-v0"


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
    run = 0
    MAX_RUN = args.n_episodes
    score_card = []
    
    while run < MAX_RUN:
        run += 1
        state = env.reset()
        state = np.reshape(state, [state_size,])
        step = 0
        score = 0
        while True:
            step += 1
            action = a3c_player.act(state)
            state, reward, done, _ = env.step(action)
            state = np.reshape(state, [state_size,])
            score += reward
            
            if done:
                print ("Run:", str(run), "score:", str(score))
                score_card.append((run, score, step))
                break
        
    with open('a3c_play_score_card_{0}.csv'.format(MAX_RUN), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(score_card)
    
    env.cleanup()
    
    plt.plot(score_card)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.show(block=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A3C Agent')
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100)
    parser.add_argument(
        '--n-max-ticks',
        type=int,
        default=300)
    parser.add_argument(
        '--model-file',
        type=str,
        default='model_{}.h5'.format(ENV_NAME))
    parser.add_argument(
        '--save-dir', 
        default='./', 
        type=str,
        help='Directory in which you desire to save the model.')
    
    
    args = parser.parse_args()

    main(args)

