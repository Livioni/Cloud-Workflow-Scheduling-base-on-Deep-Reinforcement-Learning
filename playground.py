from sre_parse import State
import gym
env = gym.make("graphEnv-v0")
state = env.reset()
print(state)
state,reward,done,info = env.step(0)
print(state)
state,reward,done,info = env.step(-1)
print(state)
env.close()