# -*- coding: utf-8 -*-
import numpy as np

class Jean_logisticRegression(object):
    
    def __init__(self, features, target, params, lr=0.05, tolerance=1e-4, Max_iter=400, verbose=True):
        self.features=features
        self.target=target
        self.lr=lr
        self.tolerance=tolerance
        self.Max_iter=Max_iter
        self.params=params
        self.verbose=verbose
        
    def add_intercept(self, intercept):
        self.intercept=intercept
        self.features=np.hstack([self.intercept, self.features])
        #print("Intercept added successfully")
        
    def sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def get_gradient(self, theta, X, y):
        #ipdb.set_trace
        return np.dot(y-self.sigmoid(np.dot(theta, X)), X)
    
    def loss(self):
        return (-1.0/len(self.target))*sum([self.target[i]*np.log(self.sigmoid(np.dot(self.params, \
                                            self.features[i,:])))+\
                                            (1-self.target[i])*np.log(1-self.sigmoid(np.dot(\
                                            self.params, self.features[i,:])))\
                                            for i in range(len(self.target))])
    
    def fit(self):
        Norm=np.linalg.norm
        Iter=0
        old_params=self.params+1
        while(Norm(self.params-old_params)>self.tolerance and Iter<=self.Max_iter):
            old_params=self.params
            indices=list(range(self.features.shape[0]))
            np.random.shuffle(indices)
            for i in indices:
    #             ipdb.set_trace()
                gradient=self.get_gradient(self.params, self.features[i,:], self.target[i])
                self.params=self.params+self.lr*gradient
            if self.verbose:
                print("Loss: %.6f"%self.loss())
            Iter += 1
        print('Number of iterations:', Iter-1)
        return self.params
    
    def predict(self, Features):
        Predictions=[]
        Intercept=np.ones(Features.shape[0]).reshape(-1,1)
        Features=np.hstack([Intercept, Features])
        for i in range(Features.shape[0]):
            if self.sigmoid(np.dot(Features[i,:], self.params))>0.5:
                Predictions.append(1.0)
            else:
                Predictions.append(0.0)


        return np.array(Predictions)
    
    
    
    
class Perceptron(object):
    
    def __init__(self, data, target, w, tolerance, alpha=0.01, Max_iter=1000):
        self.data=data
        self.target=target
        self.w=w
        self.tolerance=tolerance
        self.alpha=alpha
        self.Max_iter=Max_iter
        
    def get_gradient(self, vectors, coeffs):
        '''This function computes the gradient of the loss function in the predictions'''
        return -sum([coeffs[i]*vectors[i] for i in range(len(coeffs))]) #This is the formula for the gredient of 
#the loss function

    def fit(self):
        '''This function uses the gradient given by the above get_gredient function
        to update the parameter w until we reach the desired level of accuracy in our
        predictions'''
        Norm=np.linalg.norm
        old_w=self.w+1
        num_iter=0
        while(Norm(self.w-old_w)>self.tolerance and num_iter<self.Max_iter):
            old_w=self.w
            y_hat=[np.sign(self.w.dot(np.array(self.data.iloc[i,:]))) for i in range(self.data.shape[0])]
            Misclassified_indices=[i for i in range(len(y_hat)) if y_hat[i]!=self.target[i]] # finding the indices of the misclassified
            if len(Misclassified_indices)>0:#If there are misclassified points, then we compute the gradient and update w
                gradient=self.get_gradient(np.array([self.data.iloc[j,:] for j in Misclassified_indices]), np.array([self.target[j] for j in Misclassified_indices]))
            else:
                gradient=np.zeros_like(self.data.iloc[0,:])
            self.w=self.w-self.alpha*gradient 
            num_iter += 1
            print("gradient: {} and number of misclassified: {}".format(gradient,len(Misclassified_indices))) #We print this 
        # to track the progression of the classifications. The desired stopping point is when there are no misclassified
        #points but it can come that we stop the process when the maximal number of iterations we fixed is exceeded
        return self.w
    
    