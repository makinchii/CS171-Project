# Fruit Ninja Bot
### By An Ngo

## Description of Question and Research Topic:
This project aims to develop an agent that is able to competently play Fruit Ninja. I will be using OpenCV and PyTorch to develop a model that can scan and process a screen playing Fruit Ninja.
The goal of the agent will be to achieve the highest possible score, avoid cutting bombs, and to play optimally relative to a human. The bot's performance will be evaluated on its score, error rate in slicing fruit versus bombs, and how well it can combo fruit together. 
This project aims to solidfy the concepts of computer vision and machine learning by applying it to a classic game.

## Project Outline
### Data Collection Plan

Data will be collected for training by taking screenshots of fruit and bombs during a play session of Fruit Ninja. The screenshots will be of a standard size and have a cleaned background. There will be several photos of each type of fruit and bomb at different angles and rotations to provide the most data to train a model upon. Additional transformations will be applied to the training images to produce more training data. This data will be used in our classifier model to determine the locations and boundings of fruit and bombs. Then, I will collect data on optimal play and create a data set with states, actions, and rewards to train the model that plays the game (i.e. makes swipes).
    
### Model Plans

My plan to create an excellent Fruit Ninja agent will be to use a CNN to classify fruit vs bombs. The type of fruit is irrelevant to our goal, so it is only necessary to determine if something on the screen is a bomb or a fruit. After we determine what is on the screen, I will also train another model, a multilayer perceptron, to determine the optimal slicing line to maximize score and combos.
    
### Project Timeline

First, I will create the data set for the fruit and bombs, since it is trivial to take screenshots of gameplay footage and clean the images to a common standard. I will train a classifier model to be able to distinguish between fruits and bombs. After that, I will create a data set for the game state and scoring, so that I can train a model on the best slice to make given any particular game state. Using this in conjunction with the classifier model I trained earlier, I will have an agent capable of playing Fruit Ninja optimally.
