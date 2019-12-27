''' 
This file contains the classes and utilities defining the CB+Resource Management environment


 Dec. 2019 by Taylor Killian, University of Toronto
----------------------------------------------------------------------------------------
'''

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp

from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import itertools

# from pyro_bnn import *

# X Initialize the environment, take as input the number and values of choices

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

class Bandit_Resource_Environment:

    def __init__(self, random_state=0, num_bins=3, tot_init_items=1000, bin_values=[1.0,1.0,1.0], num_users=50):
        self.num_bins = num_bins # The number of bins is roughly equivalent to the number of arms.
        self.bin_values = bin_values # The bin values correspond to the potential reward if the user is provided the item and they purchase it.

        rng_key, rng_key_predict = random.split(random.PRNGKey(random_state))

        self.rng_key = rng_key
        self.rng_key_predict = rng_key_predict

        # Create initial allotments of bins where the lowest value items will be plentiful while the high value items may be scarce
        bin_allotments = sorted(random.randint(self.rng_key,(self.num_bins,),25,75))[::-1]
        bin_allotments /= sum(bin_allotments) # Normalize to get the percentage of items that are assigned to each bin initially
        self.init_bins = onp.round(tot_init_items*bin_allotments) # Place most items in first, low value bin and decrease from there.
        self.num_users = num_users

        # Set initialization for BNN parameters
        self.bnn_dx = 4 # Input dimensions
        self.bnn_dh = 5 # Size of hidden layer
        self.bnn_dy = num_bins # Size of output dimensions
        self.bnn_warm_up = 250 # Number of warmup runs for MCMC
        self.bnn_num_samples = 2000 # Number of BNN Samples
        self.bnn_num_chains = 1 # Number of MCMC chains
        self.bnn_device = 'cpu'

        numpyro.set_platform(self.bnn_device)
        numpyro.set_host_device_count(self.bnn_num_chains)

        self.restock() # Fill bins
        self._generate_user_preferences()
    
    def restock(self,fill_type='capacity'):
        '''
        Fills the bins to capacity or some fraction of the space that remains.
        '''
        if fill_type == 'capacity':
            self.resources_avail = onp.copy(self.init_bins)
        elif (type(fill_type) is float) and (fill_type < 1):
            self.resources_avail = onp.round(self.resources_avail + fill_type*(self.init_bins - self.resources_avail))
    
    def _generate_user_preferences(self):
        ''' 
        generate_user_preferences creates a NxK list of bernoulli probabilities for 
        how likely a user will purchase an item from each bin (each row sums to 1)
        '''
        # Create one column that is largely preferred and peter out over the rest of the bins.
        user_prefs = random.multivariate_normal(self.rng_key,np.array([0.9]+[0.1/(self.num_bins-1)]*(self.num_bins-1)) , 0.005*np.eye(self.num_bins), shape=(self.num_users,))
        all_perm = onp.array((list(itertools.permutations(list(range(self.num_bins)))))) # Note all possible permutations of the bins
        temp = all_perm[random.randint(self.rng_key,(self.num_users,),0,len(all_perm))] # Randomly select a permutation for each row of 'user_prefs'
        # Randomly permute the columns of each row in 'user_prefs'
        user_prefs = (user_prefs.flatten()[(temp+self.num_bins*np.arange(self.num_users)[...,np.newaxis]).flatten()]).reshape(user_prefs.shape) 
        # Normalize user preferences
        user_prefs = abs(user_prefs) * (1/np.sum(abs(user_prefs),axis=1))[:,np.newaxis]

        self.user_prefs = user_prefs # Set class values
        self._generate_user_context() # Generate the various context variables that will be provided to the algorithm

    def _generate_user_context(self):
        '''
        Create latent contextual identifier based on the underlying bernoulli probabilities for each user
        Right now, based on the user's bin preference (the highest probability in each row) this will dictate 
        which region of the number line the context will be uniformly sampled from.
        The user contexts will then be stratified along the number line based on bin preference.
        '''
        user_cntxt = onp.zeros(self.num_users) # Initialize Context array
        strata_vals = onp.append(np.linspace(0,1,self.num_bins,endpoint=False),1.0) # Create strata to sample contexts from based on user preferences
        for ii in range(self.num_users):
            user_pref_bin = onp.argmax(self.user_prefs[ii]) # Get which bin user prefers
            user_cntxt[ii] = onp.random.normal(0.5*(strata_vals[user_pref_bin]+strata_vals[user_pref_bin+1]),0.045) # Sample context from stratum of number line corresponding to desired bin
        
        self.user_context = user_cntxt
        

    def sample_new_user(self):
        '''
        This method draws a new user from the batch and returns the user index and context
        '''
        # Sample the user from the batch of all users
        curr_user = onp.random.choice(range(self.num_users))
        # Extract the context for this user
        curr_context = self.user_context[curr_user]

        return curr_user, curr_context

    
    def pull_arm(self,user_id,arm=0):
        '''
        Pulls arm if bernoulli prob. (as keyed by user id) is satisfied, receives reward
        '''

        # Check to see if desired arm has any resources
        if self.resources_avail[arm] <= 0:
            return 0
        
        # Evaluate if pull is successful with user's specific bernoulli prob for the chosen item
        # If successful reward = item's value, deduct item from bin
        true_prob = self.user_prefs[user_id,arm]

        if onp.random.random()<=true_prob: # "Sale" was successful
            self.resources_avail[arm] = self.resources_avail[arm]-1
            return self.bin_values[arm]
        else: # "Sale" did not go through
            return 0


    def initialize_batch_of_data(self,num_iters=100,num_pulls_per_user=10):
        '''
        This method randomly pulls arms for 100 randomly sampled users to generate a batch of data to be used 
        to fit a BNN (or GP or BLR) to predict the reward for each arm. 

        Batch of data is a list of tuples (user_idx, user_context, resources_available, action, reward)
        '''
        batch = []
        for __ in range(num_iters):
            user_idx, user_cntxt = self.sample_new_user()
            for __ in range(num_pulls_per_user):

                # Note the resources available before each pull
                resources_avail = onp.copy(self.resources_avail) + 1e-6 * onp.random.random()
                # Randomly sample an action/arm to pull
                curr_action = onp.random.choice(range(self.num_bins))
                # Pull arm
                curr_reward = self.pull_arm(user_idx,arm=curr_action)
                # Append data to batch.
                batch.append(onp.array([user_idx,user_cntxt,*resources_avail,curr_action,curr_reward]))

        self.batch = onp.vstack(batch)
        return batch


    # the non-linearity we use in our neural network
    def _nonlin(self,x):
        return np.tanh(x)


    def _model(self, X, Y, D_H, train=True):

        D_X, D_Y = X.shape[1], 1#self.num_bins
        # if train:
        #     targets = Y[:,1]

        # Sample first layer (we put unit normal priors on all weights)
        w1 = numpyro.sample("w1", dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))  # D_X D_H
        z1 = self._nonlin(np.matmul(X, w1))   # N D_H  <= first layer of activations

        # # sample second layer
        # w2 = numpyro.sample("w2", dist.Normal(np.zeros((D_H, D_H)), np.ones((D_H, D_H))))  # D_H D_H
        # z2 = self._nonlin(np.matmul(z1, w2))  # N D_H  <= second layer of activations

        # sample final layer of weights and neural network output
        w3 = numpyro.sample("w3", dist.Normal(np.zeros((D_H, D_Y)), np.ones((D_H, D_Y))))  # D_H D_Y
        z3 = np.matmul(z1, w3)  # N D_Y  <= output of the neural network

        # if train: # Isolate z3 to only the relevant predictions (which action was selected) from the neural network
        #     filter_z3 = np.array(Y[:,0])
        #     red_z3 = np.take(z3,filter_z3)

        # we put a prior on the observation noise
        prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
        sigma_obs = 1.0 / np.sqrt(prec_obs)

        # # observe data
        # if train:
        #     numpyro.sample("Y", dist.Normal(red_z3, sigma_obs), obs=1.0*targets)
        # else:
        numpyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)
    
    
    def _run_inference(self, X, Y, D_H,initial_run=True):
        
        if self.bnn_num_chains > 1:
            self.rng_key = random.split(self.rng_key,self.bnn_num_chains)
        start = time.time()
        # if initial_run:
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel, self.bnn_warm_up, self.bnn_num_samples, num_chains = self.bnn_num_chains)
        mcmc.run(self.rng_key, X, Y, D_H)
        print('\nMCMC elapsed time:', time.time() - start)
        
        return mcmc.get_samples()


    
    def bnn_predict(self, rng_key, samples, X):
        '''
        This module takes the samples of a "trained" BNN and produces predictions
        based on the X values passed in
        '''
        
        value_predictions = []
        
        for ii in range(self.num_bins):
            model = handlers.substitute( handlers.seed(self._model,rng_key), samples[ii] )
            # Note: Y will be sampled in the model because we pass Y=None here
            model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=self.bnn_dh,train=False)
            value_predictions.append(model_trace['Y']['value'])
        
        return value_predictions
    
    
    def fit_bnn_predictor(self,X=None,Y=None,initial_run=True):
        N, D_X, D_H = len(self.batch), self.bnn_dx, self.bnn_dh
        
        # Extract training data
        # if initial_run:
        X, y = [],[]
        for ii in range(self.num_bins):
            X.append(self.batch[self.batch[:,5]==ii,1:5])
            y.append(self.batch[self.batch[:,5]==ii,-1])

        post_samples = []
        for ii in range(self.num_bins):
            post_samples.append(self._run_inference(X[ii],y[ii],D_H))
        
        # samples = self._run_inference(X, Y, D_H,initial_run)

        self.bnn_samples = post_samples