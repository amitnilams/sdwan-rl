#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import gym
import gym_sdwan
import numpy as np

import csv


ENV_NAME = "Sdwan-v0"



class RandomAgent:

    def __init__(self, observation_space, action_space):
        self.action_space = action_space
 
    def act(self, state):
        action = random.randrange(self.action_space)
        print ("Taking random action", action)
        return action 


def main(args):
    random.seed(100)
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    env.MAX_TICKS  = args.n_max_ticks
    agent = RandomAgent(observation_space, action_space)
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

            action = agent.act(state)
            state_next, reward, terminal, info = env.step(action)
            #reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            score += reward
            state = state_next
            if terminal:
                print ("Run: " + str(run)  + ", score: " + str(score))
                score_card.append((run, score, step))
                break
        

    with open('random_score_card_{0}.csv'.format(MAX_RUN), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(score_card)



    env.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DQN Agent')
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100)
    parser.add_argument(
        '--n-max-ticks',
        type=int,
        default=30)
    
    args = parser.parse_args()

    main(args)

