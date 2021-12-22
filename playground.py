from DAGs_Generator import DAGs_generate
from DAGs_Generator import plot_DAG
import gym
 
env = gym.make('MyEnv-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action