# Atari Q Learner


### Introduction

One of my professors introduced me to an extremely interesting article written by DeepMind shortly before their acquisition by Google titled [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) that presents a deep learning model to play atari games with the [Arcade Learning Environment](http://www.arcadelearningenvironment.org/).

### Q-Network

The model that DeepMind created was a convolutional neural network with a Q-learner built in on top of it. A Q-learner is a combination reinforcement learner and class based DNN that learns an action value function (a $Q$ function). This action value function is a version of Markov decision process. According to wikipedia:

> A Markov decision process provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.

This is perfect for playing a video game.


### The Convolutional Net

Before the images get read in as observations by the Q-learner, the image is fed through a convolutional neural net (aka a CNN). A CNN is a sequential neural net inspired by animal sight and it is commonly used for image processing. It uses layers of convolutions and pooling to create feature maps that are used in a fully connected layer. Commonly this fully connected layer is a classifier, but in DeepMinds atari player it is a Q-learner.

![alt text](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png "typical cnn")



## PSD

### Sparse Coding

Sparse coding is a core class of unsupervised learning methods whose goals are finding the most optimal way to sparsely represent data. This translates to how to represent data with only the important features in a way to be able to recreate it.

### Predictive Sparse Decomposition

Predictive sparse decomposition or PSD is an adaptation of the sparse coding method that jointly trains a decoder and an encoder. It is outlined in [a paper from NYU](http://yann.lecun.com/exdb/publis/pdf/koray-psd-08.pdf).


### ConvPSD

Instead of using a CNN for learning the visual features of the atari screen, I propose using a convolutional PSD. This a PSD that learns convolutional filters, much like the convolutional layers of a CNN. This is outlined in [yet another paper from NYU](http://cs.nyu.edu/~ylan/files/publi/koray-nips-10.pdf).

# Running the Code

\# First train the PSD

cd dqn; th psd_conv.lua

\# Then train the network

./run_gpu breakout

