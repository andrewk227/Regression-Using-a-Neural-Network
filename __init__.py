import pandas as ps
from sklearn.preprocessing import MinMaxScaler , StandardScaler , RobustScaler
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork

# read data
data = ps.read_excel('concrete_data.xlsx' , sheet_name='concrete_data')

# drop missing values
data = data.dropna()

#normalize the data
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['int', 'float']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# print(data)

features = data.drop(columns=["concrete_compressive_strength"] , axis=1)
targets = data["concrete_compressive_strength"]

# print(features)
# print(targets)
xtrain , xtest , ytrain , ytest = train_test_split(features , targets , test_size=0.25 , random_state=42)

NN = NeuralNetwork(epochs=100 , lr = 0.5)
NN.fit(xtrain , ytrain)

ypredict = NN.predict(xtest)
print(ypredict)
print(NN.MSE(ypredict , ytest))