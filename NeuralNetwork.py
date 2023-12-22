import random
import numpy as np
import pandas as ps


class Node():
    def __init__(self , lr = 0.5)->None:
        self.connections = []
        self.inputs = []
        self.weights = []
        self.output = 0
        self.numberOfInputs = 0
        self.error = 0
        self.bias = 0
        self.lr = lr
    
    def activationFunction(self , x):
        return 1 / (1 + np.exp(-x))

    def summationFunction(self) -> float:
        Sum = 0
        for i in range(len(self.inputs)):
            Sum += (self.inputs[i] * self.weights[i])
        return Sum

    def processInput(self) -> None:
        x = self.summationFunction() + self.bias
        self.output = self.activationFunction(x)

    def sendOutput(self):
        for node in self.connections: # appends the output of the current node to the inputs to the nodes in connections
            node.inputs.append(self.output)

    def setWeightsOne(self):
        for i in range(self.numberOfInputs):
            self.weights.append(1)

    def initializeWeights(self , totalNumberOfWeights): # numberOfInputs needed
        for i in range(self.numberOfInputs):
            self.weights.append(random.uniform(-1/totalNumberOfWeights , 1/totalNumberOfWeights))
    
    def setNumberOfInputs(self , numberOfInputs)->None: # set number of inputs for current node
        self.numberOfInputs = numberOfInputs

    def clearInputs(self):
        self.inputs.clear()

    def adjustTheBias(self)->None:
        self.bias += (self.error * self.lr) # adjust the bias

    def adjustTheWeights(self)->None:
        for iterator in range(len(self.weights)):
            self.weights[iterator] += ((self.error * self.inputs[iterator]) * self.lr)

    def backPropagate(self, type:str , target:float, index:int)->None:
        self.error = self.output*(1-self.output)

        if type == 'output':
            self.error *= (target -self.output)
        else:
            Sum = 0
            for node in self.connections:
                Sum +=  (node.error * node.weights[index])

            self.error *= Sum

        self.adjustTheWeights()
        self.adjustTheBias()

class Layer():
    def __init__(self , type:str , numOfNodes:int , lr:float)->None:
        self.type = type 
        self.numOfNodes = numOfNodes
        self.nodes = self.initializeNodes(lr)

    def initializeNodes(self , lr:float = 0.5) ->list[Node]:
        nodes = []
        for n in range(self.numOfNodes):
            nodes.append(Node(lr))
        return nodes
    
    def connectNodes(self,otherLayer)->None: # connect every node in layer1 with every node in layer2
        otherLayerNodes = otherLayer.nodes
        list(map(lambda x : x.connections.extend(otherLayerNodes) , self.nodes)) # connect every node

    
    def setNumberOfInputs(self,nextLayer) ->None: # set number of inputs for next layer with number of nodes in the current layer
        nextLayerNodes = nextLayer.nodes
        list(map(lambda x : x.setNumberOfInputs(len(self.nodes)), nextLayerNodes))

    def initializeLayerWeights(self , totalNumberOfWeights)->None:
        if self.type == 'input':
            for node in self.nodes:
                node.setWeightsOne()
            return
        
        for node in self.nodes:
            node.initializeWeights(totalNumberOfWeights)

    def processInputs(self)->None:
        for node in self.nodes:
            node.processInput()

    def sendOutputs(self)->list[float]:
        if self.type == 'output':
            outputs = []
            for node in self.nodes:
                outputs.append(node.output)
            return outputs
        else: # input or hidden               
            for node in self.nodes:
                node.sendOutput()
            return []
        
    def clearLayerInputs(self)->None:
        for node in self.nodes:
            node.clearInputs()

    def backPropagateNodes(self , target:float)->None:
        for index in range(len(self.nodes)):
            self.nodes[index].backPropagate( self.type , target , index) # index for knowing hidden layer weight index

class NeuralNetwork():
    def __init__(self , numOfNodesHidden:int = 4 , epochs:int = 1000 , lr:float = 0.5 , layersNum:int = 3)->None:
        self.layersNum = layersNum
        self.inputLayersNodes = 4
        self.outputLayersNodes = 1
        self.numOfNodesHidden = numOfNodesHidden
        self.epochs = epochs
        self.totalNumberOfWeights = 0
        self.layers = []

        # Setup the NN
        self.generateLayers(lr)
        self.connectLayers()
        self.setNumberOfInputs()
        self.getTotalWeight()
        self.intializeWeights()

    def generateLayers(self , lr:float)->None:
        for n in range(self.layersNum):
            if n == 0:
                self.layers.append(Layer('input' , self.inputLayersNodes , lr) ) # input layer
            elif n == self.layersNum -1:
                self.layers.append(Layer('output' , self.outputLayersNodes ,lr)) # output layer
            else:
                self.layers.append(Layer('hidden', self.numOfNodesHidden , lr)) # hidden Layer

    def connectLayers(self)->None:
        for i in range(len(self.layers)-1):
            self.layers[i].connectNodes(self.layers[i+1])

    def setNumberOfInputs(self)->None:
        for node in self.layers[0].nodes: # set input layer number of inputs
            node.setNumberOfInputs(1)

        for i in range(len(self.layers)-1):
            self.layers[i].setNumberOfInputs(self.layers[i+1])

    def getTotalWeight(self)->None:
        for i in range(len(self.layers)-1):
            self.totalNumberOfWeights += (len(self.layers[i].nodes) * len(self.layers[i+1].nodes))
        
    def intializeWeights(self)->None:
        for layer in self.layers:
            layer.initializeLayerWeights(self.totalNumberOfWeights)
    
    def clearNNInputs(self)->None:
        for layer in self.layers:
            layer.clearLayerInputs()

    def MSE(self ,yPredict ,  ytrain:ps.DataFrame)->float:
        return (self.calculateError(yPredict , ytrain) *2) / len(ytrain)

    def calculateError(self ,yPredict , ytrain:ps.DataFrame)->float:
        result = 0
        iterator = 0
        for value in ytrain.values:
            result += ((value - yPredict[iterator]) ** 2)
            iterator+=1
        return result / 2
    
    def inititateBackPropagation(self,target:float)->None:
        for iterator in range(len(self.layers)-1 , 0 , -1): # input layer excluded
            self.layers[iterator].backPropagateNodes(target)

    def predict(self, X:ps.DataFrame)->None:
        outputs = []
        iterator = 0
        for row in X.values:
                for value in row:
                    self.layers[0].nodes[iterator].inputs.append(value)
                    iterator+=1 
                    iterator %= self.inputLayersNodes

                for layer in self.layers:
                        layer.processInputs()
                        output = layer.sendOutputs()
                        if output:
                            outputs.extend(output)
                self.clearNNInputs()
        return outputs


    def fit(self , xtrain:ps.DataFrame , ytrain:ps.DataFrame)->None:
        iterator = 0
        for epoch in range(self.epochs):
            iterator = 0
            secondIterator = 0
            targets = ytrain.values

            for row in xtrain.values:
                for value in row:
                    self.layers[0].nodes[iterator].inputs.append(value)
                    iterator+=1 
                    iterator %= self.inputLayersNodes

                for layer in self.layers:
                        layer.processInputs()
                        output = layer.sendOutputs()

                self.inititateBackPropagation(targets[secondIterator])
                secondIterator += 1
                self.clearNNInputs()
            



                

    
        


