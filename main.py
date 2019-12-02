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
    [1] "Stochastic MAB Problem with Non-stationary Rewards": https://papers.nips.cc/paper/5378-stochastic-multi-armed-bandit-problem-with-non-stationary-rewards.pdf
    [2] "Rotting Bandits": https://arxiv.org/pdf/1702.07274.pdf
    [3] "Rotting bandits are no harder than stochastic ones": https://arxiv.org/pdf/1811.11043.pdf
    [4] "Efficient Contextual Bandits in Non-Stationary Worlds": https://arxiv.org/pdf/1708.01799.pdf
    [5] "Contextual GP Bandit Optimization": http://www.ong-home.my/papers/krause11cgp-ucb.pdf
    [6] "A Contextual Bandit Alg for Ad Creative under Ad Fatigue": https://arxiv.org/pdf/1908.08936.pdf

 Nov. 2019 by Taylor Killian, University of Toronto
----------------------------------------------------------------------------------------
Notes:

4 Nov 2019 -- Algorithm ideas:
            One potential idea is EWS: https://twitter.com/eigenikos/status/1191279528875741185?s=20
            Another idea is Rexp3 from [1] above -- There may need to be some adjustments made to make this "contextual"
            SWA for the Non-parametric case presented in [2]
            FEWA from [3]
            Consider adding in fatigue terms as done in [6]

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

