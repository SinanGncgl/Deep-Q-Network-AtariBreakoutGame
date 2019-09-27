# Deep-Q-Network-AtariBreakoutGame
Playing Atari Breakout Game with Reinforcement Learning ( Deep Q Learning )

<img src="https://user-images.githubusercontent.com/23141486/50246167-84af8400-03e5-11e9-87b1-99813981482e.gif" width="390" height="410">

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

# What is Deep Q Learning and How does it works ?

I highly recommend to read this [Demystifying Deep Reinforcement Learning](https://ai.intel.com/demystifying-deep-reinforcement-learning/) who are curious about reinforcement learning.

# DQN Algorithm

![dqn](https://user-images.githubusercontent.com/23141486/50232917-50c36700-03c3-11e9-8fd7-2af40b5c16ce.png)

# Future Reward Function = Q

![q_learning_equation](https://user-images.githubusercontent.com/23141486/51302629-fd851880-1a43-11e9-8453-6610d40b059f.png)

# Loss Function

![q-learning-equation](https://user-images.githubusercontent.com/23141486/51302649-0d9cf800-1a44-11e9-8c36-12c36776628d.png)

# Network Architecture

"Working directly with raw Atari frames, which are 210 × 160 (in our case it depends on pygame screen) pixel images with a 128 color palette, can be computationally demanding, so we apply a basic preprocessing step aimed at reducing the input dimensionality. The raw frames are preprocessed by first converting their RGB representation to gray-scale and down-sampling it to a 84×84 image.As input Q-Network is preprocessing to the last 4 frames of a history and stacks them to produce the input to the Q-function.This process can be visualized as the following figure:

![a0](https://user-images.githubusercontent.com/23141486/50234733-63d83600-03c7-11e9-9ecb-67617efefb64.jpeg)![a1](https://user-images.githubusercontent.com/23141486/50234736-63d83600-03c7-11e9-9a20-da116e518b31.jpeg)![a2](https://user-images.githubusercontent.com/23141486/50234737-6470cc80-03c7-11e9-8136-60523fb67ed7.jpeg)![a3](https://user-images.githubusercontent.com/23141486/50234740-663a9000-03c7-11e9-938a-82e82fdaac4c.jpeg)

And convert these images to gray scale...

![stack0](https://user-images.githubusercontent.com/23141486/50235392-dac1fe80-03c8-11e9-814a-8f4daceea4eb.jpeg)![stack1](https://user-images.githubusercontent.com/23141486/50235394-dac1fe80-03c8-11e9-999d-e4c1f8966c4c.jpeg)![stack2](https://user-images.githubusercontent.com/23141486/50235395-dac1fe80-03c8-11e9-95d8-300addb446c3.jpeg)![stack3](https://user-images.githubusercontent.com/23141486/50235397-db5a9500-03c8-11e9-896e-169a499909a3.jpeg)

And send these into the Q-Network.

So what we have done;

* Take last 4 frames
* Resize images to 84x84
* Convert frames to gray-scale
* Stack them 84x84x4 input array and send them into the Q-Network.

The input to the neural network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 32 8 × 8 filters with stride 4 with the input image and applies a rectifier nonlinearity. The second hidden layer convolves 64 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity.The third hidden layer is fully-connected and consists of 7x7x64 input with 512 output,followed by a rectifier nonlinearity(input tensor is flattened). The final hidden layer is fully-connected and consists of 512 rectifier units. The output layer is a fully-connected linear layer with a single output for each valid action. The number of valid actions are 1 for left and 0 for right action.The architecture of the network is shown in the figure below:(Coming...)

Any contribution is welcome.
