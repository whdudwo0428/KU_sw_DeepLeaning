# 내적 dot product
a = [1,2,3]
b = [2,3,4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#a Single Neuron_NP
import numpy as np
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
outputs = np.dot(weights,inputs) + bias

print(outputs)

#A Layer of Neurons_NP
#import numpy as np
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5]
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
outputs = np.dot(weights,inputs) + bias

print(outputs)