# Referenced https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html/
#SVD based models: SVD, SVD++, Asymmetric SVD
import numpy as np
import time

class SVD():
    def __init__(self, trainset,testset, k, lr, cost, epochs):

        '''

        :param trainset: training set for SVD
        :param testset:  testset for Evaluation
        :param k:  Number of latent fatures
        :param lr: learning rate
        :param cost: cost parameter(regularization term's coefficient)
        '''
        self._trainset = trainset
        self._testset = testset
        self. _k = k
        self._lr = lr
        self._cost = cost
        self._epochs = epochs
        self._num_users, self._num_items = trainset.shape


    def train(self):

        #Set the matrix U, V / mu, biases
        self._U = np.random.normal(0, 0.1, size=(self._k, self._num_users))
        self._V = np.random.normal(0, 0.1, size=(self._k, self._num_items))
        self._mu = np.mean(self._trainset[np.where(self._trainset != 0)])
        self._bias_u = np.zeros(self._num_users)
        self._bias_i = np.zeros(self._num_items)

        self._training_process = []
        start = time.time()

        for epoch in range(self._epochs):

            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._trainset[i,j] > 0:
                        self.gradient_descent(i,j,self._trainset[i,j])

            train_loss, test_loss = self.RMSE()
            self._training_process.append((epoch,train_loss,test_loss))

            end = time.time()-start
            if (epoch+1) % 10 ==0:
                print(f"{epoch+1}-- Training loss[RMSE]: {train_loss}--Testing loss[RMSE]: {test_loss}--elapsed time--{end}")



    def RMSE(self):

        train_loss = 0
        test_loss = 0
        xs, ys = self._trainset.nonzero()
        predicted = self.full_matrix()
        x_test, y_test = self._testset.nonzero()

        for x, y in zip(xs, ys):
            train_loss += pow(self._trainset[x,y]-predicted[x,y],2)

        for i, j in zip(x_test,y_test):
            test_loss += pow(self._testset[i,j]-predicted[i,j],2)

        return np.sqrt(train_loss/len(xs)), np.sqrt(test_loss/len(x_test))

    def gradient(self, error, i,j):

        dp = (error * self._V[j,:])-(self._cost * self._U[i,:])
        dq = (error * self._U[i,:])-(self._cost * self._V[j,:])
    def gradient_descent(self,i,j,rating):
        prediction = self.get_ratings(i,j)
        error = rating - prediction
        self._bias_u[i] += self._lr * (error - self._cost * self._bias_u[i])
        self._bias_i[j] += self._lr * (error - self._cost * self._bias_i[j])


    def SGD (self,i,j):
        #calculate the error
        error = self._trainset[i,j] - self.get_ratings(i,j)

        self._U[:,i] +=self._lr * (error*self._V[:,j] - self._cost * self._U[:,i])
        self._V[:,j] +=self._lr * (error*self._U[:,i] - self._cost * self._V[:,j])

        self._bias_u[i] += self._lr * (error - self._cost * self._bias_u[i])
        self._bias_i[j] += self._lr * (error - self._cost * self._bias_i[j])

    def get_ratings(self,i,j):
        '''
        :return:
        '''
        ratings = self._mu + self._bias_u[i] + self._bias_i[j] +self._U[:,i].T.dot(self._V[:,j])
        return ratings

    def full_matrix(self):
        '''
        :return: computer full matrix using the bias , U, V
        '''
        full_matrix= self._U.T.dot(self._V) + self._mu + self._bias_u[:,np.newaxis] + self._bias_i[np.newaxis,:]
        return full_matrix
