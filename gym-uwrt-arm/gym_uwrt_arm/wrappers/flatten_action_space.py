import gym.spaces as spaces
from gym import ActionWrapper

class FlattenAction(ActionWrapper):
    r"""Action wrapper that flattens the action."""
    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

    def action(self, action):
        # return spaces.flatten(self.env.action_space, action)
        return action