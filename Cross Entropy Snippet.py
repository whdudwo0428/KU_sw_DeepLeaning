import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt  # 그래프 보려고 사용

nnfs.init()  # 랜덤 시드 고정

### 1. Define the Dense Layer!!!
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='random'):
        '''   
        :param n_inputs: 입력의 개수
        :param n_neurons: Layer 내의 뉴런의 개수
        :param initialize_method: 가중치 초기화 방법 
        '''
        match initialize_method:  # 가중치 초기화 방법 선택
            case 'uniform':
                self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
            case 'xavier':
                self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)  # Xavier 초기화
            case 'he':
                self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)  # He 초기화
            case 'gaussian':
                self.weights = np.random.randn(n_inputs, n_neurons) * 0.01  # Gaussian 초기화
            case _:
                self.weights = np.random.randn(n_inputs, n_neurons)  # 기본 랜덤 초기화

        self.biases = np.zeros((1, n_neurons))  # 기본적으로 bias는 0으로 설정

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases
        '''     
        match activation_Func:  # 활성화 함수 선택
            case 'sigmoid':     # 확률이 필요할때 시그모이드 주로 사용
                return 1 / (1 + np.exp(-layers_outputs))
            case 'relu':        # 일반적으로 렐루 사용
                return np.maximum(0, layers_outputs)
            case 'tanh':
                return np.tanh(layers_outputs)
        '''
        # type1 : return을 사용하기 싫은경우
        # self.output = np.dot(inputs, np.array(self.weights)) + self.biases
        # type2 : return을 사용하고 싶은경우
        # return np.dot(inputs, np.array(self.weights)) + self.biases

### 2. Activation Class
# 추상 클래스 정의 (활성화 함수들의 공통 구조)
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

# 활성화 함수 클래스들 (추상 클래스 상속)
class ReLUActivation(ActivationFunction):
    def forward(self, inputs):
        return np.maximum(0, inputs)

class SigmoidActivation(ActivationFunction):
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class TanhActivation(ActivationFunction):
    def forward(self, inputs):
        return np.tanh(inputs)

### 3. Activation Softmax Class
# Softmax : 주어진 입력값을 0과 1 사이의 값으로 변환하여 각 클래스에 속할 확률을 계산
class SoftmaxActivation(ActivationFunction):
    def forward(self, inputs):
        # Softmax 계산
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

### 4. Loss_Categorical_Cross_entropy Class
class LossCategoricalCrossEntropy:
    def forward(self, predictions, targets):
        '''
        :param predictions: Softmax를 통과한 출력
        :param targets: 정답지, one-hot encoding 또는 정수 레이블
        :return: categorical cross entropy loss 연산값
        '''

        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        # clip : 자르다! 식을 보면 로그가 있는데 로그0은 없잖, 정의x
        # 컴터가 알아먹을 수 있게 "범위를 정해줌"
        ''' e는 10을 의미 e-7 = 10^7
        if predictions == 0:
            predictions = 1e-7
        '''

        # If targets are sparse
        if targets.ndim == 1:
            correct_confidences = predictions[range(len(predictions)), targets]
        else:
            # if targets are one-hot encoded
            correct_confidences = np.sum(predictions * targets, axis=1)

        # Calculate negative log likelihood
        negative_log_likelihood = -np.log(correct_confidences)

        # Calculate average loss
        return np.mean(negative_log_likelihood)

### 5. Create dataset
inputs, y = spiral_data(samples=100, classes=3)
# plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap='brg')
# plt.show()

Dense1 = Layer_Dense(2, 8, initialize_method='xavier')  # 위 샘플은 2차원공간에서 정의되기 때문에 인풋을 2로 설정해야함
Dense2 = Layer_Dense(8, 8, initialize_method='xavier')  # 인자에 위 가중치 초기화 방법 입력 추가 가능
Dense3 = Layer_Dense(8, 3, initialize_method='xavier')  # 출력이 3 classes

# 활성화 함수 인스턴스화
relu_activation = ReLUActivation()
softmax_activation = SoftmaxActivation()

### 6. Forward pass
output1 = relu_activation.forward(Dense1.forward(inputs))
output2 = relu_activation.forward(Dense2.forward(output1))
output3 = softmax_activation.forward(Dense3.forward(output2))  # 마지막 레이어에서 softmax 적용

### 7. Loss calculation
loss_function = LossCategoricalCrossEntropy()
loss = loss_function.forward(output3, y)
print("Loss:", loss)

### 8. Plotting results
plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap='brg', label="True")
plt.scatter(inputs[:, 0], inputs[:, 1], c=np.argmax(output3, axis=1), cmap='coolwarm', marker='x', label="Predicted")
plt.legend()     # 축의 각 색깔이 무엇을 의미하는지
plt.title("Spiral Data Classification using Neural Network")
plt.show()
