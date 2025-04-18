"""
REINFORCE is a direct stochastic policy, meaning the action is directly learned instead of learning the expected return.
However, the expected return is optimized by estimating the gradient using the log-likelihood trick.
"""

class DiscretePolicy():
    """
    Caterogical policy: the NN returns the log probability of each action the
    agent can take, then sample the action according to a multinomial distribution.

    Log-likelihood is used to calculate the gradient
    """
    def __init__(self, env_spec):
        self._env_spec = env_spec

    # Should be implemented by all policies

    def get_action(self, observation):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space


    def terminate(self):
        """
        Clean up operation
        """
        pass

    def distribution(self):
        """
        :rtype Distribution
        """
        raise NotImplementedError

    def dist_info(self, obs, state_infos):
        """
        Return the distribution information about the actions.
        :param obs: observation values
        :param state_infos: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        raise NotImplementedError
    
class ContinuousPolicy():
    """
    Diagonal Gaussian policy: the NN returns the means and log stds of a multivariate Gaussian distribution 
    (# dimensions = # action spaces), then sample the vector-valued action from sampling the distribution.

    Log-likelihood is used to calculate the gradient.
    """
    def __init__(self, env_spec):
        self._env_spec = env_spec

    # Should be implemented by all policies

    def get_action(self, observation):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space


    def terminate(self):
        """
        Clean up operation
        """
        pass

    def distribution(self):
        """
        :rtype Distribution
        """
        raise NotImplementedError

    def dist_info(self, obs, state_infos):
        """
        Return the distribution information about the actions.
        :param obs: observation values
        :param state_infos: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        raise NotImplementedError