import numpy as np  # 행렬을 쉽게 쓰기 위해
import nnfs  # 데이터 호출 및 random seed 고정
from nnfs.datasets import vertical_data


# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''
        :param n_inputs: 입력의 개수
        :param n_neurons: 출력의 개수
        '''
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        '''
        :param inputs: 입력
        :return: 뉴런 연산의 결과 y = ax + b
        '''
        return np.dot(inputs, self.weights) + self.biases


# Relu Activation
class Activation_ReLU:
    def forward(self, inputs):
        '''
        :param inputs:뉴런의 출력
        :return: activation 결과
        '''
        return np.maximum(0, inputs)


# Softmax Activation
class Activation_Softmax:
    def forward(self, inputs):
        '''
        :param inputs: 뉴런의 출력
        :return: activation 결과
        '''
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        return prob


class Loss:  # Loss 가 class의 이름
    def calculate(self, output, y):
        '''
        :param output: Dense + activation 한 결과
        :param y: 실제 정답지
        :return: 결과와 정답지의 차이(단, 우리가 정의한 식으로 계산됨)
        '''
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:  ##
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:  ##
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
#return np.mean(negative_log_likelihood)

X, y = vertical_data(samples=100, classes=10)
# 입력의 형태가 2
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 10)
# output_ n_neuron이 3인 이유는 class의 개수가 3개이기 때문에
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense1_bias = dense1.biases.copy()
best_dense2_bias = dense2.biases.copy()

for iteration in range(200000):
    dense1.weights += 0.02 * np.random.randn(2, 3)
    dense2.weights += 0.02 * np.random.randn(3, 10)
    dense1.biases += 0.02 * np.random.randn(1, 3)
    dense2.biases += 0.03 * np.random.randn(1, 10)
    ## MLP 연산
    out = activation1.forward(dense1.forward(X))
    out = activation2.forward(dense2.forward(out))
    ## loss계산
    loss = loss_function.calculate(out, y)
    ## 정확도 계산
    predictions = np.argmax(out, axis=1)
    accuracy = np.mean(predictions == y)

    ##최소 로스일때 값 저장
    if loss < lowest_loss:
        print("New set pf weights found, iteration:",
              iteration, 'loss:', loss, 'acc :', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense1_bias = dense1.biases.copy()
        best_dense2_bias = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense2.weights = best_dense2_weights.copy()
        dense1.biases = best_dense1_bias.copy()
        dense2.biases = best_dense2_bias.copy()
