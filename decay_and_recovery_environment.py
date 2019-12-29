''' 
This file contains the classes and utilities defining a secondary exp environment where rewards decay 
each time the arm is played and will start recovering if is not played.

The jig is how to balance decay, recovery and how to encapsulate this in the context...
Right now... the context will be the number of rounds since each arm was last pulled

For now, this environment doesn't incorporate any sense of user preferences nor does it include any
notion of user specific decay rates. -- Future work!!


 Dec. 2019 by Taylor Killian, University of Toronto
----------------------------------------------------------------------------------------
'''

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import numpy as np
import numpy.random as npr

import itertools

# Initialize the environment, take as input the number and values of choices

# Provide methods to:
# 
#    - Select arms based on context (user preferences and number of items remaining) + Thompson Sampling
#        - It seems that it might be best to use a BNN for the reward prediction since we can comfortably adjust
#          for context as well as the various actions. If I use a GP, it might get a little unwieldly since we'll 
#          need to train a GPLVM for each action and might need significant numbers of users to get the regressions to work
#        - We can't use BLR since the expected reward should vanish if an item is 'gone' and such an item is chosen. This
#          is inherently a non-linear relationship in the context.
#        - The true reward will be something of the form:
#                  r = \sum_j (val_j*Pr(usrpref_j))*I(amtRemaining_j>0)
#           where Pr(usrpref_j) is a bernoulli random variable \in {0,1} based on the probabilty of usrpref_j
#    - Sample from user preferences whether they "accept" the item
#    - Update arms/resources available
#    - Update BLR or other context regression algorithm
#    - Evaluate per episode regret (Here we're going to focus on instantaneous regret)

# defining a class for our online bayesian logistic regression
class OnlineLogisticRegression:
    '''
    Class to run Bayesian Logistic Regression...
    Adapted from https://gdmarmerola.github.io/ts-for-contextual-bandits/
    '''
    
    # initializing
    def __init__(self, lambda_, alpha, n_dim):
        
        # the only hyperparameter is the deviation on the prior (L2 regularizer)
        self.lambda_ = lambda_; self.alpha = alpha
                
        # initializing parameters of the model
        self.n_dim = n_dim, 
        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_
        
        # initializing weights
        self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
        
    # the loss function
    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])
        
    # the gradient
    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)
    
    # method for sampling weights
    def get_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
    
    # fitting method
    def fit(self, X, y):
                
        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':True}).x
        self.m = self.w
        
        # step 2, update q
        P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)
                
    # probability output method, using weights sample
    def predict_proba(self, X, mode='sample'):
        
        # adding intercept to X
        #X = add_constant(X)
        
        # sampling weights after update
        self.w = self.get_weights()
        
        # using weight depending on mode
        if mode == 'sample':
            w = self.w # weights are samples of posteriors
        elif mode == 'expected':
            w = self.m # weights are expected values of posteriors
        else:
            raise Exception('mode not recognized!')
        
        # calculating probabilities
        proba = 1 / (1 + np.exp(-1 * X.dot(w)))
        return np.array([1-proba , proba]).T


class DandR_Environment:

    '''
    Decay and Recovery Environment
    ------------------------------
    class description incoming
    '''

    def __init__(self, random_state=0, num_arms=3, init_values=[1.0,1.0,1.0], decay_rate=0.05, recovery_rate=0.05):
        self.num_arms = num_arms 
        self.init_probs = init_values
        self.arm_probs = init_values # The initial arm probabilities
        self.context = np.zeros(num_arms)

        # Set initialization for BNN parameters
        self.blr_dx = num_arms # Input dimensions
        self.blr_dy = 1 # Size of output dimensions
        self.num_samples = 1000 # Number of BLR Samples

        self.decay_rate = 1-decay_rate
        self.recovery_rate = 1+recovery_rate

    
    def _update_context(self,arm=0):
        ''' Update the internal context of the bandit (set arm played to zero, increment all other arms) '''
        self.context[arm] = 0
        self.context[np.arange(self.num_arms) != arm] += 1

    def _update_arm_probs(self,arm=0):
        ''' Update the arm probabilities according to whether they were pulled or not '''
        for ii in range(self.num_arms):
            if ii == arm:
                self.arm_probs[ii] *= self.decay_rate
                continue
            self.arm_probs[ii] = min(self.arm_probs[ii]*self.recovery_rate, self.init_probs[ii])

    def _reset(self):
        ''' Reset the arms reward probabilities and the internal context'''
        self.arm_probs = np.copy(self.init_probs)
        self.context = np.zeros(self.num_arms)
    
    def pull_arm(self,arm=0):
        ''' Method for pulling, returning reward and updating arms + context '''
        # Evaluate if pull is successful with user's specific bernoulli prob for the chosen item
        # If successful reward = item's value, deduct item from bin
        true_prob = self.arm_probs[arm]

        if npr.random()<=true_prob: # Pull was successful
            reward = 1
        else: # Pull was unsuccessful
            reward = 0
        
        # Update arm's context and 
        self._update_context(arm)
        self._update_arm_probs(arm)

        return reward




