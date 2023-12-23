import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from perceptron import Neural_Network as nn

def convert(x):
    result = []
    for i in range(10):
        if(i == x): result.append(1)
        else: result.append(0)
    return result

def hot_one(x):
    result = []
    max_ind = 0
    for i in range(len(x)):
        if(x[i] > x[max_ind]): max_ind = i
    for i in range(len(x)):
        if(i == max_ind): result.append(1)
        else: result.append(0)
    return result

def softmax(Z):
    return np.exp(Z)/ np.sum(np.exp(Z))


data = pd.read_csv('mnist_train.csv')

data = np.array(data)
m, n = data.shape

network = nn(784, 15, 10, 0.35)

print("####Training####")
for row in data:
    input = row.copy()[1:].T.tolist()
    output = convert(row[0])
    network.train(input, output)

test_data = np.array(pd.read_csv('mnist_test.csv'))
corr = 0
print("####Testing####")
for row in test_data:
    input = row.copy()[1:].T.tolist()
    req_output = convert(row[0])
    output = network.query(input)
    print(softmax(output), hot_one(output))
    print(req_output)
    if(hot_one(output) == req_output): corr += 1

print("The ratio of correct values is : ", corr/test_data.shape[0])






