import random
import numpy as np
import pandas as ps


class Node():
    def __init__(self)->None:
        self.connections = []
        self.inputs = []
        self.weights = []
        self.output = 0
        self.numberOfInputs = 0
    
    def activationFunction(self , x):
        return 1 / (1 + np.exp(-x))

    def summationFunction(self) -> float:
        Sum = 0
        for i in range(len(self.inputs)):
            Sum += (self.inputs[i] * self.weights[i])
        return Sum

    def processInput(self) -> None:
        x = self.summationFunction()
        self.output = self.activationFunction(x)
        self.inputs.clear() # clear inputs

    def sendOutput(self):
        for node in self.connections: # appends the output of the current node to the inputs to the nodes in connections
            node.inputs.append(self.output)

    def initializeWeights(self , totalNumberOfWeights): # numberOfInputs needed
        for i in range(self.numberOfInputs):
            self.weights.append(random.uniform(-1/totalNumberOfWeights , 1/totalNumberOfWeights))
    
    def setNumberOfInputs(self , numberOfInputs)->None: # set number of inputs for current node
        self.numberOfInputs = numberOfInputs

class Layer():
    def __init__(self , type:str , numOfNodes:int)->None:
        self.type = type 
        self.numOfNodes = numOfNodes
        self.nodes = self.initializeNodes()

    def initializeNodes(self) ->list[Node]:
        nodes = []
        for n in range(self.numOfNodes):
            nodes.append(Node())
        return nodes
    
    def connectNodes(self,otherLayer)->None: # connect every node in layer1 with every node in layer2
        otherLayerNodes = otherLayer.nodes
        list(map(lambda x : x.connections.extend(otherLayerNodes) , self.nodes)) # connect every node
    
    def setNumberOfInputs(self,nextLayer) ->None: # set number of inputs for next layer with number of nodes in the current layer
        nextLayerNodes = nextLayer.nodes
        list(map(lambda x : x.setNumberOfInputs(len(self.nodes)), nextLayerNodes))

    def initializeLayerWeights(self , totalNumberOfWeights)->None:
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

class NeuralNetwork():
    def __init__(self , numOfNodesHidden:int = 4 , epochs:int = 1000)->None:
        self.layersNum = 3
        self.inputLayersNodes = 4
        self.outputLayersNodes = 1
        self.numOfNodesHidden = numOfNodesHidden
        self.epochs = epochs
        self.totalNumberOfWeights = 0
        self.layers = []
        self.outputs = []

        # Setup the NN
        self.generateLayers()
        self.connectLayers()
        self.setNumberOfInputs()
        self.getTotalWeight()
        self.intializeWeights()

    
    def generateLayers(self)->None:
        for n in range(self.layersNum):
            if n == 0:
                self.layers.append(Layer('input' , self.inputLayersNodes) ) # input layer
            elif n == self.layersNum -1:
                self.layers.append(Layer('output' , self.outputLayersNodes)) # output layer
            else:
                self.layers.append(Layer('hidden', self.numOfNodesHidden)) # hidden Layer

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

    def fit(self , xtrain:ps.DataFrame , ytrain:ps.DataFrame)->None:
        i = 0
        for row in xtrain.values: # epoches missing
            for value in row:
                self.layers[0].nodes[i].inputs.append(value)
                i+=1 
                i %= self.inputLayersNodes

            for layer in self.layers:
                    layer.processInputs()
                    output = layer.sendOutputs()
                    if output:
                        self.outputs.extend(output)
        
        
        


