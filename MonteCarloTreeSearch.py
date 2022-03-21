import gym
import numpy as np
env = gym.make("clusterEnv-v0")

state = env.reset()
ready_list = getattr(env, 'ready_list')    
tasks = env.tasks.copy()
observation, reward, done, info = env.step(0)

print(state)
print(observation)

env.set_state(state)
env.set_ready_list(ready_list)

observation, reward, done, info = env.step(0)
print(observation)

class TreeNode(object):
    def __init__(self, parent,state,ready_list,done_job,tasks):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._state = state
        self._ready_list = ready_list
        self._done_job = done_job
        self._tasks = tasks

    def expand(self, action_list):
        """Expand tree by creating new children.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action in action_list:
            if action not in self._children:
                self._children[action] = TreeNode(self)


# root = TreeNode(None,state,ready_list,done_job,tasks)
# print(root._state)