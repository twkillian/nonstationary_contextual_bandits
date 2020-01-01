''' 
This file be the primary driver for running a variant of rotting/non-stationary contextual bandits. 
The notion is that rewards may decay over time as the "best" arm may be over used. This creates a change
in what would be considered the best arm. The idea behind this notional environmental set-up is that
patients bodies become resistant or build a tolerance toward certain medications, ads lose their effect
the more frequently they are served... In a way, you could consider this non-stationarity to be a form of 
inverse novelty. This notion is covered for MAB problems (poss. non-contextual) in part in [2] below.

There are a couple of different ways that I've considered implementing the nonstationarity... 
    I: The distributions of reward have a temporal component (decaying or improving) that influence their magnitude
    II: The reward can be considered a resource. High potential reward with fewer draws vs. medium potential reward with many draws vs. low/no reward with infinite draws
    III: Similar to II but the resources can be replenished if the different arms are left alone for a period of time.

The real challenge is determining how to best organize an algorithm to stay somewhat flexible as many Contextual Bandit 
approaches converge to a fixed/static policy after some time. There have been a couple of papers that have dealt with
this. I'm needing to read them to get a better idea of what I need to do to differentiate my work from theirs as well 
as build upon already "proven" methods.
    [x1] "Stochastic MAB Problem with Non-stationary Rewards": https://papers.nips.cc/paper/5378-stochastic-multi-armed-bandit-problem-with-non-stationary-rewards.pdf
    [x2] "Rotting Bandits": https://arxiv.org/pdf/1702.07274.pdf
    [x3] "Rotting bandits are no harder than stochastic ones": https://arxiv.org/pdf/1811.11043.pdf
    [x4] "Efficient Contextual Bandits in Non-Stationary Worlds": https://arxiv.org/pdf/1708.01799.pdf
    [x5] "Contextual GP Bandit Optimization": http://www.ong-home.my/papers/krause11cgp-ucb.pdf
    [x6] "A Contextual Bandit Alg for Ad Creative under Ad Fatigue": https://arxiv.org/pdf/1908.08936.pdf
    [7] "Mortal Multi-Armed Bandits": http://papers.nips.cc/paper/3580-mortal-multi-armed-bandits.pdf
    [x8] "Recovering Bandits": https://papers.nips.cc/paper/9561-recovering-bandits.pdf
    [9] "Optimistic Planning for the Stochastic Knapsack Problem": http://proceedings.mlr.press/v54/pike-burke17a.html
    [10] "Blocking Bandits": https://papers.nips.cc/paper/8725-blocking-bandits.pdf
    [11] "Categorized Bandits": https://papers.nips.cc/paper/9586-categorized-bandits.pdf
    [12] "Weighted Linear Bandits for Non-Stationary Environments": https://papers.nips.cc/paper/9372-weighted-linear-bandits-for-non-stationary-environments.pdf

 Nov. 2019 by Taylor Killian, University of Toronto
----------------------------------------------------------------------------------------
Notes:

4 Nov 2019 -- Algorithm ideas:
            One potential idea is EWS: https://twitter.com/eigenikos/status/1191279528875741185?s=20
            - Another idea is Rexp3 from [1] above -- There may need to be some adjustments made to make this "contextual"
            - SWA for the Non-parametric case presented in [2]
            - FEWA from [3]
            X - Consider adding in fatigue terms as done in [6] -- (Dec 29--Added by including rolling window of how often each arm has been pulled, included in the context...)

11 Nov 2019 -- Further notes:
            It's pretty clear that [1-3] are non-contextual settings. One potential contribution that I can make is in applying 
            these ideas to contextual settings. I'm not entirely sure how to appropriately set up the context parameterization and how
            to update the inference for a change in context... I'm also unsure how the context may confuse the above algs.

1 Dec 2019 -- Even further notes...
            After chatting with Audrey Durand, it's unclear how to build a contextual environment that has the right interchange between
            global decaying reward functions and user specific guidance in how to select arms. What we came together to figure out is to 
            think of this as a full on resource allocation problem where there is a global allocation of items (low, medium and high value)
            that gets integrated into the context of each user (some randomly generated latent variable vector) that belies some user preference
            over those items. This way, the bandit selection algoritm needs to learn to associate user preferences with the resources available 
            when choose which arm to pull.

            In essence this will become a set of K bins with varying levels of fill and value. [xxxxx] [xx   ] [xxxx ]
            Then each user will have a set of preferences over each item (eg. binomal probabilities that sum to 1): 
                            User 1 (id: 245064): 0.25  0.45  0.30
                            User 2 (id: 243345): 0.05  0.90  0.05
                            User 3 (id: 323443): 0.125 0.225 0.65
                            etc...
            
            What I need to do is figure out the right way to create the user context that belies similar preferences... Can use an equation of the
            three preferences, or maybe even the preferences themselves at first...

            The way that this will work is that User j is introduced along with their context including the resources available
            --Using BLR or GPs / BayesOpt with Thompson Sampling... We determine which arm to "pull". 
            --The user purchases the suggested item based on their personal preference for that item. 
            --Update the priors and resources based on whether user chooses to purchase the suggested item or not.
                                                                                                            
'''
# Imports
import sys, os, time

import numpy as onp
import itertools

import jax.random as random
import jax.numpy as np

import numpy.random as npr

# from resource_environment import Bandit_Resource_Environment
from decay_and_recovery_environment import DandR_Environment


def main(env,agent,agent_type,oracle,num_iters,refit_fq,rng_trainers,rng_testers):
    # Main wrapper function for running bandit algorithms
    # Loop over the specified number of iterations with the provided agent (random or TS-BNN or ---)
    #   Record regret (oracle - received reward) over the experiment, report and save

    times_fit = 1 # Track the number of times the BNN has been fit (only for dealing with RNG Keys for BNN training)

    # Train a BNN for each action if needed
    if agent_type != 'random':
        env.fit_predictor(rng_trainers[0])

    regret = onp.zeros(num_iters)
    rewards = onp.zeros((num_iters,2))
    for itr in onp.arange(num_iters):
        # Sample new context
        # usr_idx, usr_cntxt = env.sample_new_user()
        cntxt = onp.insert(env.get_context(),0,1)[np.newaxis,:]
        # Get current resources avail
        # resources_avail = onp.copy(env.resources_avail) + 1e-4 * onp.random.random()
        # Create new query point
        # x_test = onp.insert(resources_avail,0,usr_cntxt)[np.newaxis,:]

        # Get action from agent
        # action = agent(rng_testers[itr],x_test)
        action = agent(rng_testers[0],cntxt)
        # Get oracle action
        # o_act = oracle(usr_idx,x_test)
        o_act = oracle(cntxt)

        # Take step in env and get reward
        # a_rew = env.pull_arm(usr_idx, execute=True, arm=action)
        a_rew = env.pull_arm(execute=True, arm=action)
        # Get oracle reward
        # o_rew = env.pull_arm(usr_idx, execute=False, arm=o_act)
        o_rew = env.pull_arm(execute=False, arm=o_act)
        # print(f"{agent_type} Action: {action}, Oracle Action: {o_act}, {agent_type} Reward: {a_rew}, Oracle Reward: {o_rew}, Regret: {o_rew-a_rew}")

        # Record the regret
        regret[itr] = regret[itr-1] + (o_rew-a_rew)

        # Add experience to the batch
        # env.batch.append(onp.array([usr_idx,usr_cntxt,*resources_avail,action,a_rew]))

        # if itr % rstk_fq == 0:
        #     env.restock(fill_type=rstk_per)

        if (agent_type == 'ts_bayes') and ((itr+1) % refit_fq == 0):
            print(f"========Fitting BLR after iteration {itr+1}========")
            env.fit_predictor(rng_trainers[0])
            times_fit += 1


    return regret, rewards



if __name__ == '__main__':
    
    # Run initializations
    #    -- Environment (Resource env, pot. shared rewards across users)
    #    -- Agents (Random, TS, "Greedy", Pot. algs from above)
    #    -- Oracle needs to take into account non-zero resources
    #    -- Define number of iterations, frequency and rate of restocking as well as refitting of BNNs
    
    # = Meta parameters for experiments = "Resource Env"
    # n_bins = 3
    # tot_items=300
    # init_batch_size = 200
    # n_train_pulls = 3
    # num_iterations = 5000
    # restock_percent = 0.5
    # restock_freq = 300

    # bnn_refit_freq = 100

    # env_type = 'resource'

    # = Meta parameters for experiments = "Decay and Recovery Env"
    n_arms = 10
    init_batch_size = 1000
    num_iterations = 2500
    agg_window = 50

    refit_freq = 100

    env_type = 'decay'

    dandr_rates = {'slow':0.05,'moderate':0.10,'fast':0.25}
    decay_type = 'fast'
    recovery_type = 'fast'

    reward_center = 0.65
    random_seed = 2
    npr.seed(random_seed)
    init_values = npr.normal(loc=reward_center,scale=0.1,size=n_arms)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    

    # agent_type = 'ts_bayes'
    agent_type = 'random'

    # == Create RNG Keys == 
    rng_trainers = random.split(random.PRNGKey(1234),50)
    rng_testers = random.split(random.PRNGKey(2334),5000)

    # === Initialize the environment ===
    # env = Bandit_Resource_Environment(
    #                                     random_state=1234,
    #                                     num_users=1000,
    #                                     num_bins=n_bins,
    #                                     bin_values=[1.0]*n_bins,
    #                                     tot_init_items=tot_items)
    # Initialize a batch of data from the environment to train BNNs with
    # train_batch = env.initialize_batch_of_data(num_iters=init_batch_size, num_pulls_per_user=n_train_pulls)
    # Restock env to get started
    # env.restock()
    
    env = DandR_Environment(num_arms=n_arms,
                            init_values=init_values,
                            num_init_draws=init_batch_size,
                            decay_rate=dandr_rates[decay_type],
                            recovery_rate=dandr_rates[recovery_type],
                            experiment_iters=num_iterations,
                            agg_window=agg_window)
    
    # ==== Initialize Agents + Oracle ====
    if agent_type == 'random':
        agent = env.random_agent
    elif agent_type == 'ts_bayes':
        agent = env.ts_bayes_agent
    else:
        print("Specified agent type hasn't been implemented")
    
    oracle = env.oracle
    
    # ===== Run Bandit Algs, compare with oracle =====
    regret, rewards = main(
                    env,
                    agent,
                    agent_type,
                    oracle,
                    num_iterations,
                    refit_freq,
                    rng_trainers,
                    rng_testers)
    # Save regret
    onp.save(f'regrets/regret_{env_type}_{agent_type}_{n_arms}arms_{num_iterations}iters_{refit_freq}fitfreq_{decay_type}decay_{recovery_type}recovery_{timestamp}.npy', onp.array(regret))
    onp.save(f'regrets/reward_{env_type}_{agent_type}_{n_arms}arms_{num_iterations}iters_{refit_freq}fitfreq_{decay_type}decay_{recovery_type}recovery_{timestamp}.npy', onp.array(rewards))


