import numpy as np
import random
import time
np.random.seed(2022)

class WRMF():
    def __init__(self, data, train, test,k,alpha,learning_rate,cost_parameter):
        '''

        :param data:
        :param train:
        :param test:
        :param k:
        :param alpha:
        :param learning_rate:
        :param cost_parameter:
        '''

        self._gt_matrix = data #ground truth dataset
        self._train = np.array(self.binary(train), dtype = np.float64)
        self._test = np.array(self.binary(test), dtype = np.float64)
        self._num_users, self._num_items = self._train.shape
        self._k = k
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._cost_parameter = cost_parameter
        self._U = np.random.normal(0, scale = 1.0/self._k, size = (self._num_users, self._k))
        self._V = np.random.normal(0, scale = 1.0/self._k, size = (self._num_items, self._k))

    #binarization for implicit dataset
    def binary(self,array):
        idx = array.nonzero()
        for row, col in zip(*idx):
            array[row][col] = 1
        return array

    def bootstrap(self):
        data = self._train
        u = random.choice(data.nonzero()[0])
        i = random.choice(data[u].nonzero()[0])
        j = random.choice(np.argwhere(data[u] == 0).T[0])
        return u,i,j
