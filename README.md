# Welcome

This is my personal practice of implementing various algorithms of RL from scratch.

Most of them will be in jupyter notebook and some of them involving multiprocess
would be in normal python files.

The framework will always be PyTorch, as a personal practice too.

Normally I use cartpole for easy algorithms in this project and I skip the 
visual input part. (which is quite trivial if you add few conv layers). 

And for
harder and visual-related algorithms I will pick various atari game as my environment.

Due to time limit, I will not provide systematic analysis to any particular algorithm. 
And be aware these are personal usage so bugs do appear frequently.

If the project is mature, I will accept open issues.
For now, however, let me dive in. (I guess no one even read this repo though)

Project file structure will be changed continuously to match my needs.

# PLAN: 
## Model-Free RL
### Policy Gradient
- [x] REINFORCE 
- [x] Off-Policy REINFORCE
- [x] Basic Actor Critic
- [x] Advantage Actor Critic using Huber loss and Entropy 
- [x] A3C async using time interval
- [x] A3C async using episode
- [x] A2C sync using time interval
- [x] A2C sync using episode
- [x] Experimental trial, off-policy Actor Critic using naive bootstrap (failed)
- [x] Fail log of above experiment
- [ ] DPG (I guess I will skip this and jump to DDPG) 
- [ ] DDPG
- [ ] D4PG
- [ ] MADDPG
- [ ] TRPO
- [ ] PPO
- [ ] ACER
- [ ] ACTKR
- [ ] SAC
- [ ] SAC with AAT(Automatically Adjusted Temperature
- [ ] TD3
- [ ] SVPG
- [ ] IMPALA
### Deep Q Learning 
- [ ] DQN
- [ ] DRQN for POMDP
- [ ] Dueling DQN
- [ ] Double DQN
- [ ] PER
- [ ] Rainbow DQN
### Distributed RL 
- [ ] C51
- [ ] QR-DQN
- [ ] IQN
- [ ] Dopamine (DQN + C51 + IQN + Rainbow)
### Policy Gradient with Action-Dependent Baselines:
- [ ] Q-prop
- [ ] Stein Control Variates
### Path-Consistency Learning
- [ ] PCL
- [ ] Trust-PCL
### Q-learning + Policy Gradient:
- [ ] PGQL
- [ ] Reactor
- [ ] IPG
### Evolutionary Algorithm
### Monte Carlo Tree (Alpha Zero)
## Exploration RL
### Intrinsic Motivation
- [ ] VIME
- [ ] CTS-based Pseudocounts
- [ ] PixelCNN-based Pseudocounts
- [ ] Hash-based Counts
- [ ] EX2
- [ ] ICM
- [ ] RND
### Unsupervised RL
- [ ] VIC
- [ ] DIAYN
- [ ] VALOR
## Hierachy RL
## Memory RL
## Model-Based RL
## Meta-RL
## Scaling-RL