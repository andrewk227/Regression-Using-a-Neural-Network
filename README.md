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
```python
import pandas as ps
from sklearn.preprocessing import MinMaxScaler , StandardScaler , RobustScaler
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
from sklearn.metrics import r2_score

# read data
data = ps.read_excel('concrete_data.xlsx' , sheet_name='concrete_data')

# drop missing values
data = data.dropna()

#normalize the data
scaler = MinMaxScaler()

features = data.drop(columns=["concrete_compressive_strength"] , axis=1)
targets = data["concrete_compressive_strength"]

xtrain , xtest , ytrain , ytest = train_test_split(features , targets , test_size=0.25 , random_state=42)

xtrain = ps.DataFrame(scaler.fit_transform(xtrain))
xtest = ps.DataFrame(scaler.transform(xtest))

NN = NeuralNetwork(epochs=100 , lr = 0.01 , numOfNodesHidden=128)
print("Learning Phase:...")
NN.fit(xtrain , ytrain)

ypredict = NN.predict(xtest)
print(ypredict)
print()
print(f"MSE: {NN.MSE(ypredict , ytest)}")
print(f"Error: {NN.calculateError(ypredict , ytest)}")

r2 = r2_score(ytest, ypredict)
print(f'R2 Score: {r2}')

```

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