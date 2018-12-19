# Deep-Q-Network-AtariBreakoutGame
Playing Atari Breakout Game with Reinforcement Learning ( Deep Q Learning )




# Overview 

This project follows the description of the [Deep Q Learning algorithm](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) described in this paper.

# Installation Dependencies:

* Python 3.x
* Numpy
* OpenCV-Python
* PyGame
* PyTorch

# How To Run

* `git clone https://github.com/SnnGnc/Deep-Q-Network-AtariBreakoutGame.git`
* `cd brekout`
* To train the game;
* `python dqn.py train`,
* To test the pre-trained version,
* `python dqn.py test`

# What is Deep Q Learning and How does is work ?

I highly recommend to read this [Demystifying Deep Reinforcement Learning](https://ai.intel.com/demystifying-deep-reinforcement-learning/) who are curious about reinforcement learning.

# DQN Algorithm

![dqn](https://user-images.githubusercontent.com/23141486/50232917-50c36700-03c3-11e9-8fd7-2af40b5c16ce.png)




# Network Architecture

"Working directly with raw Atari frames, which are 210 × 160 (our case it depends on pygame screen) pixel images with a 128 color palette, can be computationally demanding, so we apply a basic preprocessing step aimed at reducing the input dimensionality. The raw frames are preprocessed by first converting their RGB representation to gray-scale and down-sampling it to a 84×84 image.As input Q-Network is preprocessing to the last 4 frames of a history and stacks them to produce the input to the Q-function.This process can be visualized as the following figure:

![a0](https://user-images.githubusercontent.com/23141486/50234733-63d83600-03c7-11e9-9ecb-67617efefb64.jpeg)![a1](https://user-images.githubusercontent.com/23141486/50234736-63d83600-03c7-11e9-9a20-da116e518b31.jpeg)![a2](https://user-images.githubusercontent.com/23141486/50234737-6470cc80-03c7-11e9-8136-60523fb67ed7.jpeg)![a3](https://user-images.githubusercontent.com/23141486/50234740-663a9000-03c7-11e9-938a-82e82fdaac4c.jpeg)

And convert these images to gray scale...

![stack0](https://user-images.githubusercontent.com/23141486/50235392-dac1fe80-03c8-11e9-814a-8f4daceea4eb.jpeg)![stack1](https://user-images.githubusercontent.com/23141486/50235394-dac1fe80-03c8-11e9-999d-e4c1f8966c4c.jpeg)![stack2](https://user-images.githubusercontent.com/23141486/50235395-dac1fe80-03c8-11e9-95d8-300addb446c3.jpeg)![stack3](https://user-images.githubusercontent.com/23141486/50235397-db5a9500-03c8-11e9-896e-169a499909a3.jpeg)

And send these into the Q-Network.


