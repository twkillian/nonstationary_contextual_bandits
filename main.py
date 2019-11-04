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
    [3] "Efficient Contextual Bandits in Non-Stationary Worlds": https://arxiv.org/pdf/1708.01799.pdf
    [4] "Contextual GP Bandit Optimization": http://www.ong-home.my/papers/krause11cgp-ucb.pdf

 Nov. 2019 by Taylor Killian, University of Toronto
----------------------------------------------------------------------------------------
Notes:

4 Nov 2019 -- Algorithm ideas:
            One potential idea is EWS: https://twitter.com/eigenikos/status/1191279528875741185?s=20
            Another idea is Rexp3 from [1] above -- There may need to be some adjustments made to make this "contextual"
            SWA for the Non-parametric case presented in [2]
      
'''