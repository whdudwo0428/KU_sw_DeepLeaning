import numpy as np
import nnfs  # 실험에 쓰는 데이터 호출 및 random seed 고정
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt

nnfs.init()  # 랜덤 시드 고정


### 1. Define the Dense Layer!!!
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='random'):
        '''
        :param n_inputs: 입력의 개수
        :param n_neurons: Layer 내의 뉴런의 개수
        :param initialize_method: 가중치 초기화 방법
        '''
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01  # Gaussian 초기화
        self.biases = np.zeros((1, n_neurons))  # 기본적으로 bias는 0으로 설정

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


### 2. Activation Class
# 활성화 함수 클래스들 (추상 클래스 상속)
class ReLUActivation:
    def forward(self, inputs):
        return np.maximum(0, inputs)


### 3. Activation Softmax Class
# Softmax : 주어진 입력값을 0과 1 사이의 값으로 변환하여 각 클래스에 속할 확률을 계산
class SoftmaxActivation:
    def forward(self, inputs):
        '''
        :param inputs: 뉴런의 출력
        :return: activation 결과
        '''
        # Softmax 계산
        ############################################################
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


### 4. Loss_Categorical_Cross_entropy Class
class Loss:
    def calculate(self, output, y):
        '''
        :param output: Dense + activation 한 결과
        :param y: 실제 정답지
        :return:  결과와 정답지의 차이
        '''
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_ture):
        '''
        :param predictions: Softmax를 통과한 출력
        :param targets: 정답지, one-hot encoding 또는 정수 레이블
        :return: categorical cross entropy loss 연산값
        '''
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_ture.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_ture
            ]
        elif len(y_ture.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_ture,
                axis=1
            )
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

        # Clip predictions to prevent log(0)
        # clip : 자르다! 식을 보면 로그가 있는데 로그0은 없잖, 정의x
        # 컴터가 알아먹을 수 있게 "범위를 정해줌"
        ''' e는 10을 의미 e-7 = 10^7
        if predictions == 0:
            predictions = 1e-7
        '''


### 5. Create dataset
inputs, y = vertical_data(samples=100, classes=10)
# plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap='brg')
# plt.show()

Dense1 = Layer_Dense(2, 3)  # 위 샘플은 2차원공간에서 정의되기 때문에 인풋을 2로 설정해야함
Dense2 = Layer_Dense(3, 10)  # 인자에 위 가중치 초기화 방법 입력 추가 가능

# 활성화 함수 인스턴스화
relu_activation = ReLUActivation()
softmax_activation = SoftmaxActivation()

### 7. Loss calculation
loss_function = Loss_CategoricalCrossEntropy()

lowest_loss = 9999999
best_dense1_weights = Dense1.weights.copy()
best_dense2_weights = Dense2.weights.copy()
best_dense1_bias = Dense1.weights.copy()
best_dense2_bias = Dense2.weights.copy()

for iteration in range(200000):  # iteration 반복을 의미
    Dense1.weights += 0.05 * np.random.randn(2, 3)
    Dense2.weights += 0.05 * np.random.randn(3, 10)
    Dense1.biases += 0.05 * np.random.randn(1, 3)
    Dense2.biases += 0.05 * np.random.randn(1, 10)

    ### 6. Forward pass MLP연산
    output1 = relu_activation.forward(Dense1.forward(inputs))
    output2 = softmax_activation.forward(Dense2.forward(output1))

    ### 7. Loss calculation
    loss = loss_function.calculate(output2, y)
    ## 정확도 계산
    predictions = np.argmax(output2, axis=1)
    accuracy = np.mean(predictions == y)

    ## 최소 로스일 때 값 저장
    if loss < lowest_loss:
        print("New set pf weights found, iterration",
              iteration, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = Dense1.weights.copy()
        best_dense2_weights = Dense2.weights.copy()
        best_dense1_biases = Dense1.biases.copy()
        best_dense2_biases = Dense2.biases.copy()
        lowest_loss = loss
    else:
        Dense1_weights = best_dense1_weights.copy()
        Dense2_weights = best_dense2_weights.copy()
        Dense1_bias = best_dense1_biases.copy()
        Dense2_bias = best_dense2_biases.copy()

### 8. Plotting results
plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap='brg', label="True")
plt.scatter(inputs[:, 0], inputs[:, 1], c=np.argmax(output2, axis=1), cmap='coolwarm', marker='x', label="Predicted")
plt.legend()  # 축의 각 색깔이 무엇을 의미하는지
plt.title("Spiral Data Classification using Neural Network")
plt.show()
