# Regression-Using-a-Neural-Network
## Overview
This is a simple implementation of a feedforward neural network in Python using object-oriented programming. The neural network consists of an input layer, one hidden layer, and an output layer. The network is trained using backpropagation and uses a sigmoid activation function for hidden layers and a rectified linear unit (ReLU) activation function for the input and output layer.

## Classes
### Node
Represents a single node (neuron) in the neural network.
Includes methods for activation functions, weight initialization, and backpropagation.
### Layer
Represents a layer in the neural network, including input, hidden, and output layers.
Handles the connection between nodes in different layers and provides methods for processing inputs and backpropagation.
### NeuralNetwork
Main class that brings together layers to form a complete neural network.
Includes methods for training the network (fitting) and making predictions.

## Usage
https://github.com/MoayadR/Regression-Using-a-Neural-Network/blob/2d04270dcf2733a906bc57e567672175cc36b38b/__init__.py#L1C1-L41

## Parameters
* numOfNodesHidden: Number of nodes in the hidden layer.
* epochs: Number of training epochs.
* lr: Learning rate for weight updates.

## Training
The fit method is used to train the neural network. It takes input data (x_train) and target data (y_train) and iteratively updates the weights using backpropagation.

## Predictions
The predict method is used to make predictions on new data after the neural network has been trained. It takes input data (x_test) and returns the predicted outputs.

## Evaluation
The mean squared error (MSE) is used as the evaluation metric during training. The MSE is printed after each epoch, providing insights into the training progress.

## Dependencies
* numpy
* pandas

## Note
This implementation is a basic example and may need further customization for specific use cases. Feel free to explore and modify the code to suit your needs.