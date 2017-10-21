# dqn_multiprocessing
multi-processing version of dqn on cartpole and atari game
it only takes 1 minute to solve CartPole-v1 of openai gym

# Dependencies
* Python 3.5
* Tensorflow
* Keras
* h5py
* numpy
* gym

# Basic Idea
Idea is simple. Just seperate agent into two process. One is to interact with
environment and the other is to train agent's model.
* process 1(actor) : interact with environment and collect sample
* process 2(learner) : with sample, do training

So agent do not need to wait the interaction process to train. It is simple
method but improves learning speed a lot.


# Multi-processing
I have used multi-processing module which is included in python as default. This
code uses three queue to share information between actor(interacting process)
* queue 1 : collected sample by actor
* queue 2 : model learned by learner
* queue 3 : indicator whether learning is over or not

# How to train
'''
python cartpole_mp.py
'''

# Future do
I will implement multi-processing on Atari game!


