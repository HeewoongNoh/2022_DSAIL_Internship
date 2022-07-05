import numpy as np
import random
import time
np.random.seed(2022)
class BPR():
    def __init__(self, data, train, test, k, learning_rate, cost_parameter):
        '''

        :param data: Original data matrix
        :param train: train data
        :param test: test data
        :param k: number of factors
        :param learning_rate: learning_rate for SGD
        :param cost_paramter: cost parameter for regularization(lambda)
        '''
        self._gt_matrix = data #ground truth dataset
        self._train = np.array(self.binary(train), dtype = np.float64)
        self._test = np.array(self.binary(test), dtype = np.float64)
        self._num_users, self._num_items = self._train.shape
        self._k = k
        self._learning_rate = learning_rate
        self._cost_parameter = cost_parameter
        self._U = np.random.normal(0, scale = 1.0/self._k, size = (self._num_users, self._k))
        self._V = np.random.normal(0, scale = 1.0/self._k, size = (self._num_items, self._k))


    def train(self, epochs):
        '''
        training matrix factorization: update matrix(U,V)
        :param epochs: number of training iterations
        '''
        auc_list = []
        for epoch in range(1,epochs):
            start = time.time()
            #Equal probability with replacement
            u,i,j = self.bootstrap()
            self.gradient_descent(u,i,j)
            end = time.time() - start
            if epoch % 1000 ==0:
                print(f"Epoch:{epoch}--:{end}elapsed")
                auc_time = time.time()
                AUC = self.AUC()
                auc_list.append(AUC)
                auc_end = time.time() - auc_time
                print(f"AUC value: {AUC}----time:{auc_end}elapsed")

        return auc_list
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

    def gradient_descent(self,u,i,j):

        x_uij_hat = self._U[u].dot(self._V[i].T) - self._U[u].dot(self._V[j].T)
        sigmoid_value = 1/(1+ np.exp(x_uij_hat))
        #Derivatives  w.r.t (u,i,j)
        self._U[u,:] += self._learning_rate * (sigmoid_value * (self._V[i] - self._V[j]) - self._cost_parameter * self._U[u])
        self._V[i,:] += self._learning_rate * (sigmoid_value * (self._U[u]) - self._cost_parameter * self._V[i])
        self._V[j, :] += self._learning_rate * (-sigmoid_value * (-1*self._U[u]) - self._cost_parameter * self._V[j])

    def AUC(self):
        '''
        Area Under ROC Curve
        :return: auc
        '''
        self._train_pred = self._U.dot(self._V.T)
        hit = 0

        for user in self._test.nonzero()[0]:
            tmp = 0
            _i = self._test[user].nonzero()[0]
            _j = np.transpose(self._gt_matrix[user]==0).nonzero()[0].T

            for i in _i:
                for j in _j:
                    if self._train_pred[user,i] > self._train_pred[user,j]:
                        tmp+=1
            hit += (tmp/(len(_i)*len(_j)))
        auc = hit / len(self._test.nonzero()[0])

        return auc











