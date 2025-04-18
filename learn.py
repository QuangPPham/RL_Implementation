# import optimizer, or implement it myself
import numpy

class RLAlgorithm():
    def __init__(self, discount, learning_rate, ):
        self.lamb = discount
        self.alpha = learning_rate
        self.traj = 0 # (state, action, reward) from timestep 0 to T

    def calc_grad(self):
        pass

    def calc_R(self):
        pass

    def train(self):
        pass