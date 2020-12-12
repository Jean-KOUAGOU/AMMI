import numpy as np, pandas as pd, progress_bar as pb

class MulticlassLogReg(object):
    def __init__(self, lr=0.2, Max_iter=30, verbose=False):
        self.verbose = verbose
        self.lr=lr
        self.Max_iter=Max_iter
       
    #Definition of the softmax function
    def softmax(self, k, X):
        return np.exp(X[k]-np.max(X))/np.sum([np.exp(X[j]-np.max(X)) for j in range(len(X))])
    
    #Function to compute the probability of (y=k/X_i; weights)
    def prob_yequal_k_given_Xi_weights(self, X_i, k, Weights):
        return self.softmax(k, [np.dot(Weights[l], X_i) for l in range(len(Weights))])
    
    # Function to compute the gradient associated to the target k
    def get_gradient(self, k, Weights, X, y):
        #K=len(Weights)
        grad_k=-np.array([(float(y.iloc[i]==k)-self.prob_yequal_k_given_Xi_weights(X.iloc[i, :], k, Weights))\
                 *X.iloc[i, :] for i in range(X.shape[0])]).mean(axis=0)
        return grad_k
    
    def fit(self, features, target):
        self.weights=np.ones((len(np.unique(target)), 1+features.shape[1]))
        self.target=target
        Intercept=np.ones_like(self.target).reshape(-1,1)
        features=pd.DataFrame(np.concatenate([Intercept, features], axis=1))
        
        num_iter=0
        
        while (num_iter<self.Max_iter):
            #previous_weights=self.weights.copy()
            for j in range(self.weights.shape[0]):
                #We update each weight
                self.weights[j]=self.weights[j]-self.lr*self.get_gradient(j, self.weights,\
                                    features, target)
            num_iter += 1
            
    def predict(self, X_):
         #We first add the intercept column
        Intercept=np.ones(X_.shape[0]).reshape(-1,1)
        X=pd.DataFrame(np.concatenate([Intercept, X_], axis=1))
        
        predictions=[]
    
        for i in range(X.shape[0]):
            
            probs=[]
            # Here we compute the probabilties for y to be a given target
            for k in np.unique(self.target):
                probs.append(self.prob_yequal_k_given_Xi_weights(X.iloc[i, :], k, self.weights))
            # Now we choose the target with the highest probability
            predictions.append(np.argmax(probs))
        return np.array(predictions)  