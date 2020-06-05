import numpy as np

np.random.seed(42)

# dataset generator
def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        '''forward pass which computes the output of a neuron (a la: mx + b)'''
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def __init__(self):
        pass
    '''rectified linear activation function'''
    def forward(self, inputs):
        '''designed to adjust the output from a neuron to make all values positive or 0'''
        self.output = np.maximum(0, inputs)

# create the data set, 3 sets, 100 values per set
X, y = create_data(100, 3)

# the n_neurons can be anything, we chose 5
# the n_inputs is the length of one row of the X input data (like different sensor readings)
# 2 is specified in the create_data function
layer1 = Layer_Dense(2, 5)

activation1 = Activation_ReLU()

layer1.forward(X)

# note that the input to the forward function here is the output of the neuron
activation1.forward(layer1.output)
# print(layer1.output)
print(activation1.output)
