''' 
This file contains the classes and utilities defining the CB+Resource Management environment


 Dec. 2019 by Taylor Killian, University of Toronto
----------------------------------------------------------------------------------------
'''

import numpy as np
import numpy.random as npr


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

    def __init__(self, num_bins=3, tot_init_items=1000, bin_values=[1,10,100], num_users=50):
        self.num_bins = num_bins # The number of bins is roughly equivalent to the number of arms.
        self.bin_values = bin_values # The bin values correspond to the potential reward if the user is provided the item and they purchase it.

        # Create initial allotments of bins where the lowest value items will be plentiful while the high value items may be scarce
        bin_allotments = sorted(npr.randint(25,2000,self.num_bins))[::-1]
        bin_allotments /= sum(bin_allotments) # Normalize to get the percentage of items that are assigned to each bin initially
        self.init_bins = np.round(tot_init_items*bin_allotments) # Place most items in first, low value bin and decrease from there.
        self.num_users = num_users

        self.restock() # Fill bins
        self._generate_user_preferences()

    
    
    def restock(self,fill_type='capacity'):
        '''
        Fills the bins to capacity or some fraction of the space that remains.
        '''
        if fill_type == 'capacity':
            self.resources_avail = np.copy(self.init_bins)
        elif (type(fill_type) is float) and (fill_type < 1):
            self.resources_avail = np.round(self.resources_avail + fill_type*(self.init_bins - self.resources_avail))
    
    def _generate_user_preferences(self):
        ''' 
        generate_user_preferences creates a NxK list of bernoulli probabilities for 
        how likely a user will purchase an item from each bin (each row sums to 1)
        '''

        user_prefs = npr.random((self.num_users,self.num_bins))
        # Randomly magnify certain rows and columns (This is used to make one of the items more desired by the randomly selected user)
        rands = npr.choice(range(self.num_users),size=(round(self.num_users*.45),2))
        rands[:,1] = rands[:,1]%self.num_bins # Isolate to only the relevant columns of the experiment.
        user_prefs[rands[:,0],rands[:,1]] = 5*user_prefs[rands[:,0],rands[:,1]] # Increase the randomly chosen item preferences
        # Normalize user preferences
        user_prefs = user_prefs * (1/np.sum(user_prefs,axis=1))[:,np.newaxis]

        self.user_prefs = user_prefs # Set class values
        self._generate_user_context() # Generate the various context variables that will be provided to the algorithm

    def _generate_user_context(self):
        '''
        Create latent contextual identifier based on the underlying bernoulli probabilities for each user
        '''
        pass

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

        if npr.random()<=true_prob: # "Sale" was successful
            self.resources_avail[arm] -= 1
            return self.bin_values[arm]
        else: # "Sale" did not go through
            return 0


        
    
