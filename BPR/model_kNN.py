# Learning models with BRP - Adaptive k-Nearest-Neighbor

import numpy as np
import random
import time
np.random.seed(2022)
class BPR_kNN():
    def __init__(self, data, train, test, learning_rate, reg_pos, reg_neg):
        '''

        :param data: Original data matrix
        :param train: train data
        :param test: test data
        :param learning_rate: learning_rate for SGD
        :param reg_pos,reg_neg: reg for updates C_il, reg for updates C_jl

        '''
        self._gt_matrix = data #ground truth dataset
        self._train = np.array(self.binary(train), dtype = np.float64)
        self._test = np.array(self.binary(test), dtype = np.float64)
        self._num_users, self._num_items = self._train.shape
        self._learning_rate = learning_rate
        # Changed code for using adaptive kNN for BPR learning.
        self._C =np.random.normal(0,scale = 1.0/self._num_items, size=(self._num_items,self._num_items) )
        self._reg_pos = reg_pos
        self._reg_neg = reg_neg



    def train(self, epochs):
        '''
        training matrix factorization: update matrix(U,V)
        :param epochs: number of training iterations
        '''
        #Asymetric matrix and matrix[i][i] = 0 remove self correlation
        np.fill_diagonal(self._C,0)
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
                print(self._C)
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

        x_ui_hat = np.sum(self._train[u]*self._C[i]) - self._C[i,i]
        x_uj_hat = np.sum(self._train[u]*self._C[j]) - self._C[j,j]
        x_uij_hat = x_ui_hat - x_uj_hat
        sigmoid_value = 1/(1+ np.exp(x_uij_hat))

        #user's item list
        item_list = self._train[u].nonzero()[0]
        l = np.delete(item_list,np.where(item_list == i))


        #Derivatives  w.r.t (u,i,j)
        #C_il, C_li update
        '''
        correlation function symmetric twice later?
        '''


        self._C[i,l] += self._learning_rate * (sigmoid_value * 1 + self._reg_pos * self._C[i,l])
        self._C[l,i] += self._learning_rate * (sigmoid_value * 1 + self._reg_pos *self._C[l,i])
        #C_ij, C_ji update
        self._C[j,l] += self._learning_rate * (sigmoid_value * -1 + self._reg_neg *self._C[j,l])
        self._C[l,j] += self._learning_rate * (sigmoid_value * -1 + self._reg_neg *self._C[l,j])

    def AUC(self):
        '''
        Area Under ROC Curve
        :return: auc
        '''
        #pred matrix
        self._train_pred = np.zeros(self._train.shape)
        for u in range(self._train.shape[0]):
            for i in range(self._train.shape[1]):
                self._train_pred[u,i] = np.sum(self._train[u] * self._C[i]) - self._C[i,i]

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











