#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import gym_sdwan_stat
import numpy as np
import argparse
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

LEARNING_RATE = 0.001

# In[2]:


class DQNPlayer:

    def __init__(self, observation_space, action_space, model_file):
        self.action_space = action_space
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        #self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        
        self.model.load_weights(model_file)

   
    def act(self, state):
        q_values = self.model.predict(state)
        action = np.argmax(q_values[0])
        print ("Taking predicted  action", action)
        return action


# In[ ]:

def main(args):
    env = gym.make('Sdwan-stat-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    env.MAX_TICKS  = args.n_max_ticks
    dqn_player = DQNPlayer(observation_space, action_space, args.model_file)
    run = 0
    MAX_RUN = args.n_episodes
    score_card = []
    while run < MAX_RUN:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        score = 0
        while True:
            step += 1
            #env.render()
            action = dqn_player.act(state)
            state_next, reward, terminal, info = env.step(action)
            #reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            score += reward
            
            state = state_next
            if terminal:
                    print ("Run:", str(run), "score:", str(score))
                    score_card.append((run, score, step))
                    break
        

    with open('dqn_stat_play_score_card_{0}.csv'.format(MAX_RUN), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(score_card)
    
    env.cleanup()

# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DQN Agent')
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
        default="model.h5")
    
    args = parser.parse_args()

    main(args)

