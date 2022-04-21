import numpy as np
import gym
from agent import Agent
from matplotlib import pyplot as plt 
from environments import MujocoEnv

#This program is there to train the mountaincar


if __name__ == '__main__':

    with open('GROUP1/env_info.txt') as f:
        lines = f.readlines()
    env_type = lines[0].lower()

    env = MujocoEnv(gym.make('Hopper-v2'))
    env_specs = {'observation_space': env.observation_space, 'action_space': env.action_space}
    agent = Agent(env_specs)

    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make('Hopper-v2')
    score_history = []
    num_episodes = 10000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = np.array(agent.choose_action(observation)).reshape((3,))
            #env.render() #To see what is happening?
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode', i, "score =", score)

    with open("score_history.txt", 'w') as f:
        for s in score_history:
            f.write(str(s) + '\n')

    plt.plot(score_history)
    plt.show()
