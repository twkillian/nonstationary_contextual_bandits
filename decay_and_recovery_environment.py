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

from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import numpy as onp
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
        self.m = onp.zeros(self.n_dim)
        self.q = onp.ones(self.n_dim) * self.lambda_
        
        # initializing weights
        self.w = onp.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
        
    # the loss function
    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + onp.sum([onp.log(1 + onp.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])
        
    # the gradient
    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * onp.array([y[j] *  X[j] / (1. + onp.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)
    
    # method for sampling weights
    def get_weights(self):
        return onp.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
    
    # fitting method
    def fit(self, X, y):
                
        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 50, 'disp':False}).x
        self.m = self.w
        
        # step 2, update q
        P = (1 + onp.exp(1 - X.dot(self.m))) ** (-1)
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
        proba = 1 / (1 + onp.exp(-1 * X.dot(w)))
        return onp.array([1-proba , proba]).T


class DandR_Environment:

    '''
    Decay and Recovery Environment
    ------------------------------
    class description incoming
    '''

    def __init__(self, num_arms=3, init_values=[1.0,1.0,1.0], decay_rate=0.05, recovery_rate=0.05, num_init_draws=350, experiment_iters=5000, agg_window=50, random_state = 1234, num_chains = 1, num_samples=1000, num_warmup = 500):
        
        self.num_chains = num_chains
        self.random_state = random_state
        self.num_warmup = num_warmup
        self.num_samples = num_samples # Number of BLR Samples
        self.eps = 0.15 # Random exploration within TS agent *bugged that I have to do this*
        
        self.num_arms = num_arms 
        self.exp_iters = experiment_iters
        self.init_probs = init_values
        self.arm_probs = init_values # The initial arm probabilities
        self.context = onp.zeros(num_arms)

        # Set initialization for BNN parameters
        self.blr_dx = 2*num_arms+1 # Input dimensions
        self.blr_alpha = 0.9
        self.blr_lambda = 0.01 

        self.decay_rate = 1-decay_rate
        self.recovery_rate = 1+recovery_rate

        self.batch = []
        self.history = onp.zeros((self.exp_iters,self.num_arms))
        self.hist_context = onp.zeros(self.num_arms)
        self.round = 0
        self.agg_window = 50

        self._reset()
        self._initialize_batch_of_data(num_iters=num_init_draws)
        self._reset()

    
    def _update_context(self,arm=0):
        ''' Update the internal context of the bandit (set arm played to zero, increment all other arms) '''
        self.context[arm] = 0
        self.context[onp.arange(self.num_arms) != arm] += 1

        self.history[self.round,arm] = 1
        self.round += 1
        self.hist_context = self._agg_history()
        

    def _update_arm_probs(self,arm=0):
        ''' Update the arm probabilities according to whether they were pulled or not '''
        for ii in range(self.num_arms):
            if ii == arm:
                self.arm_probs[ii] *= self.decay_rate
                continue
            self.arm_probs[ii] = min(self.arm_probs[ii]*self.recovery_rate, self.init_probs[ii])

    def _reset(self):
        ''' Reset the arms reward probabilities and the internal context'''
        self.arm_probs = onp.copy(self.init_probs)
        self.context = onp.zeros(self.num_arms)
        self.history = onp.zeros((self.exp_iters,self.num_arms))
        self.hist_context = onp.zeros(self.num_arms)
        self.round = 0
    
    def _initialize_batch_of_data(self,num_iters=1000):
        '''
        This method pulls arms randomly for 'num_iters' rounds 
        to fit a BLR to predict the reward for each arm. 

        Batch of data is a list of tuples (context, action, reward)
        '''
        
        for __ in range(num_iters):
            # Randomly sample an action/arm to pull
            curr_action = npr.choice(range(self.num_arms))
            # Pull arm
            __ = self.pull_arm(arm=curr_action)

    
    def _agg_history(self):
        ''' Aggregating history over self.agg_window to provide a bit more context into the arms '''
        return onp.mean(self.history[max(self.round-self.agg_window,0):self.round,:],axis=0)

    def get_context(self):
        curr_context = onp.copy(self.context)
        curr_hist_context = onp.copy(self.hist_context)
        return onp.hstack([curr_context,curr_hist_context])

    
    def pull_arm(self,arm=0,execute=True):
        ''' Method for pulling, returning reward and updating arms + context '''
        # Evaluate if pull is successful with user's specific bernoulli prob for the chosen item
        # If successful reward = item's value, deduct item from bin
        true_prob = self.arm_probs[arm]

        if npr.random()<=true_prob: # Pull was successful
            reward = 1
        else: # Pull was unsuccessful
            reward = 0

        if execute: # Account for Oracle draws that we don't want to affect history
            # Aggregate history
            curr_hist_context = onp.copy(self.hist_context)

            # Record round
            curr_context = onp.copy(self.context)
            self.batch.append([*curr_context,*curr_hist_context,arm,reward])
            
            # Update all arm contexts and probabilities
            self._update_context(arm)
            self._update_arm_probs(arm)

        return reward

    # def fit_BLR(self, X=None, y=None, init=False):
        # ''' Method to fit reward estimates for each arm '''

        # if init:
        #     # Initialize the separate Bayesian Linear Models for each arm and extract training data
        #     self.model = []
        #     batch = onp.vstack(self.batch)
        #     X, y = [], []
        #     for ii in range(self.num_arms):
        #         self.model.append(OnlineLogisticRegression(self.blr_lambda, self.blr_alpha, self.blr_dx))

        #         X.append( onp.hstack([ onp.ones( (sum(batch[:,-2]==ii),1) ) , batch[batch[:,-2]==ii,:self.blr_dx-1] ]) )
        #         y.append(batch[batch[:,-2]==ii,-1])

        # else:
        #     # Create training data (if not provided)
        #     if X is None:
        #         batch = onp.vstack(self.batch)
        #         X, y = [], []
        #         for ii in range(self.num_arms):
        #             X.append( onp.hstack([ onp.ones( (sum(batch[:,-2]==ii),1) ), batch[batch[:,-2]==ii,:self.blr_dx-1] ]) )
        #             y.append(batch[batch[:,-2]==ii,-1])

        # # Fit the separate Bayesian Linear Models for each arm
        # for ii in range(self.num_arms):
            
        #     self.model[ii].fit(X[ii],y[ii])

    # def predict_arms(self,X_test, mode='sample'):
        ''' Predicts arm probabilies '''
        # probs = []
        # X_test = onp.insert(X_test,0,1)[onp.newaxis,:]
        # for ii in range(self.num_arms):
        #     temp = self.model[ii].predict_proba(X_test, mode=mode)
        #     probs.append(temp[0,1])

        # return probs / onp.sum(probs)

    def _model(self, X=None, Y=None,predict=False):
        if predict:
            pass
        betas = numpyro.sample('betas', dist.MultivariateNormal(loc=np.zeros(X.shape[1]), covariance_matrix = 5*np.eye(X.shape[1])) )
        sigma = numpyro.sample('sigma', dist.Exponential(1.))
        
        mu = np.matmul(X,betas)

        numpyro.sample('obs', dist.Normal(mu,sigma), obs=Y)

    def _run_inference(self,rng_key=None,X=None,Y=None):
        ''' Run inference on the model specified above with the supplied data '''
        
        if rng_key is None:
            rng_key = random.PRNGKey(self.random_state)
        
        if self.num_chains > 1:
            rng_key_ = random.split(rng_key,self.num_chains)
        else:
            rng_key, rng_key_ = random.split(rng_key)
        
        start = time.time()
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel,self.num_warmup,self.num_samples)
        mcmc.run(rng_key_,X=X, Y=Y)
        print('/n MCMC elapsed time:', time.time()-start)

        return mcmc.get_samples()

    def _blr_predict(self, rng_key, samples, X,predict=False):
        ''' This module takes the samples of a "trained" BLR and produces predictions based on the provided X '''
        model = handlers.substitute( handlers.seed(self._model,rng_key), samples ) # Pass post. sampled parameters to the model
        # Gather a trace over possible Y values given the model parameters and input value X
        model_trace = handlers.trace(model).get_trace(X=X, Y=None, predict=predict)

        return model_trace['obs']['value']

    def fit_predictor(self, rng_key):

        # Extract training data
        batch = onp.vstack(self.batch)
        X, y = [], []
        for ii in range(self.num_arms):
            X.append( onp.hstack([ onp.ones( (sum(batch[:,-2]==ii),1) ) , batch[batch[:,-2]==ii,:self.blr_dx-1] ]) )
            y.append(batch[batch[:,-2]==ii,-1])
        
        post_samples = []
        for ii in range(self.num_arms):
            post_samples.append(self._run_inference(rng_key, X[ii],y[ii]))


        self.blr_samples = post_samples

    def predict_values(self, predict_rng_key, X):

        value_predictions = []

        for ii in range(self.num_arms):
            vmap_args = (self.blr_samples[ii], random.split(predict_rng_key, self.num_samples*self.num_chains))
            prediction = vmap(lambda samples, rng_key: self._blr_predict(rng_key,samples,X,predict=True))(*vmap_args)
            # Take mean of prediction samples to provide prediction for this model
            pred = np.mean(prediction,axis=0)
            # Append prediction to output list
            proba = 1 / (1 + onp.exp(-1 * pred))
            value_predictions.append(proba)

        return value_predictions / onp.sum(value_predictions)
    
    def random_agent(self,predict_rng_key,X_test):
        '''
        Agent that returns a random selection among the arms
        '''
        return npr.choice(onp.arange(self.num_arms))
    
    def ts_bayes_agent(self,predict_rng_key, X_test, exploration = True):
        '''
        Agent that utilizes BLR prediction of expected rewards to choose from a la Thompson Sampling
        '''
        # Predict expected values of reward (per bin) via BNN
        value_probs = self.predict_values(predict_rng_key,X_test)
        # Return action that has the highest expected return
        if npr.random() < self.eps:
            return npr.choice(onp.arange(self.num_arms))
        else:
            return onp.argmax(value_probs)

    def oracle(self, X_test):
        '''
        Oracle that returns the current best arm
        '''
        return onp.argmax(self.arm_probs)