import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt  # 그래프 보려고 사용

nnfs.init()  # 랜덤 시드 고정

### 1. Define the Dense Layer!!!
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''
        :param n_inputs: 입력의 개수
        :param n_neurons: Layer 내의 뉴런의 개수
        :param initialize_method: 가중치 초기화 방법
        '''

        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01  # Gaussian 초기화
        self.biases = np.zeros((1, n_neurons))  # 기본적으로 bias는 0으로 설정

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        # Softmax 계산
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        # enumerate는 index랑 내부의 값을 순차적으로 내뱉는 역할)) enumerate(['a', 'b', 'c']) 0,'a' 1,'b' 2,'c'
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) = np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        '''
        :param predictions: Softmax를 통과한 출력
        :param targets: 정답지, one-hot encoding 또는 정수 레이블
        :return: categorical cross entropy loss 연산값
        '''
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        # If targets are sparse
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(sample),y_true
            ]
        elif len(y_true.shape) == 2:
            # if targets are one-hot encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Calculate negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)

        # Calculate average loss
        return negative_log_likelihoods

    def backward(self,dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self,y_pred, y_true):
        self.activation.forward(y_pred)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

dense1 = Layer_Dense(2,3)
activation1 = Activation_Relu()
dense2 = Layer_Dense(3,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

X,y = spiral_data(samples=100, classes=3)

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

print('loss', loss)

prediction = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(prediction == y)

print('accuracy', accuracy)

loss_activation.backward(loss_activation.output,y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)