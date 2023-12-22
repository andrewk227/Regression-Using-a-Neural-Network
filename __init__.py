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

# print(data)

features = data.drop(columns=["concrete_compressive_strength"] , axis=1)
targets = data["concrete_compressive_strength"]

# print(features)
# print(targets)
xtrain , xtest , ytrain , ytest = train_test_split(features , targets , test_size=0.25 , random_state=42)

xtrain = ps.DataFrame(scaler.fit_transform(xtrain))
xtest = ps.DataFrame(scaler.transform(xtest))


NN = NeuralNetwork(epochs=100 , lr = 0.01 , numOfNodesHidden=128)
print("Learning Phase:...")
NN.fit(xtrain , ytrain)

ypredict = NN.predict(xtest)
print(ypredict)
print(NN.MSE(ypredict , ytest))

r2 = r2_score(ytest, ypredict)
print(f'R2 Score: {r2}')

# Predict user defined data
print('-----------------------------------------------------------')
print("User defined data:")

userData = [
    [10 , 20 , 30 ,40]
                     ]

newX = ps.DataFrame( userData, columns=['cement', 'water', 'superplasticizer' , 'age'])

ypredict = NN.predict(newX)
print(ypredict)
