import gym
import gym_sdwan
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import Adam


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

    env = gym.make('Sdwan-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    env.MAX_TICKS  = args.n_max_ticks
    dqn_player = DQNPlayer(observation_space, action_space, args.model_file)
    total_reward = 0

    state = env.reset()
    print('Initial State:', state)
    state = np.reshape(state, [1, observation_space])
    step = 0
    score = 0
    error = False
    while True:
        step += 1
        action = dqn_player.act(state)
        next_state, reward, error, info = env.step(action)
        total_reward += reward
        print('Ticks:', step, 'Action:', action, 'Ob:', next_state, 'R:', 
            reward, 'Total Reward:', total_reward)

        state_next = np.reshape(next_state, [1, observation_space])
        state = state_next

        if error:
            print("Episode Aborted  after {} timesteps".format(step))
            break

    env.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DQN Agent')
    parser.add_argument(
        '--n-max-ticks',
        type=int,
        default=30)
    parser.add_argument(
        '--model-file',
        type=str,
        default="model.h5")
    
    args = parser.parse_args()

    main(args)


