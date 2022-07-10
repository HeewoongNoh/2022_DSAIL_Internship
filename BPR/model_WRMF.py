import numpy as np
import random
import scipy.sparse
import time
np.random.seed(2022)


class WRMF():
    def __init__(self, data, train, test,k,alpha,cost_parameter):

        '''

        :param data: Original data matrix(ground truth dataset)
        :param train: train data
        :param test: test data
        :param k: number of factors
        :param alpha:
        :param cost_parameter:
        '''

        self._gt_matrix = data
        self._train = np.array(self.binary(train), dtype = np.float64)
        self._test = np.array(self.binary(test), dtype = np.float64)
        self._num_users, self._num_items = self._train.shape
        self._k = k
        self._cost_parameter = cost_parameter
        self._X = np.random.normal(0, scale = 1.0/self._k, size = (self._num_users, self._k))
        self._Y = np.random.normal(0, scale = 1.0/self._k, size = (self._num_items, self._k))
        #changed codes for using Weighted regularized matrix factorization for BPR
        self._alpha = alpha
        self._C_ui = scipy.sparse.csr_matrix(self._train>0,dtype=np.float32)
        if self._alpha != 1.0:
            self._C_ui = self._alpha * self._C_ui

    #alternating least squares
    def train(self, epochs):
        auc_list = []
        C_iu = self._C_ui.T.tocsr()

        for epoch in range(epochs):
            start = time.time()
            self.least_squares(self._C_ui,self._X,self._Y)
            self.least_squares(C_iu,self._Y,self._X)
            end = time.time() - start
            if epoch:
                print(f"Epoch:{epoch}---{end}elapsed")
                auc_time = time.time()
                AUC = self.AUC()
                auc_list.append(AUC)
                auc_end = time.time() - auc_time
                print(f"AUC value: {AUC}----time:{auc_end}elapsed")

        return auc_list

    def least_squares(self,C_ui,X,Y):
        users, factors = X.shape
        YtY = Y.T.dot(Y)

        for u in range(users):
            A = YtY + self._cost_parameter * np.eye(factors)
            b = np.zeros(factors)
            for i, confidence in self.nonzeros(C_ui, u):
                factor = Y[i]
                A += (confidence-1)*np.outer(factor,factor)
                b += confidence * factor
            X[u] = np.linalg.solve(A,b)

    # returns the non zeroes of a row in csr_matrix
    def nonzeros(self,m,row):
        for index in range(m.indptr[row],m.indptr[row+1]):
            yield m.indices[index], m.data[index]

    #binarization for implicit dataset
    def binary(self,array):
        idx = array.nonzero()
        for row, col in zip(*idx):
            array[row][col] = 1
        return array

    # def bootstrap(self):
    #     data = self._train
    #     u = random.choice(data.nonzero()[0])
    #     i = random.choice(data[u].nonzero()[0])
    #     j = random.choice(np.argwhere(data[u] == 0).T[0])
    #     return u,i,j


    def AUC(self):
        '''
        Area Under ROC Curve
        :return: auc
        '''
        self._train_pred = self._X.dot(self._Y.T)
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


