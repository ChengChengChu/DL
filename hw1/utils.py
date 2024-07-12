import numpy as np

class NN:
    def __init__(self, layers, lr=10e-7):
        self.lr = lr
        self.layers = layers
        self.W = []