import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init() #랜덤시드들이 고정됨 /이 기준으로만 랜덤값이 설정됨

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method = 'random'): #입력갯수 출력갯수
        match initialize_method:    # 가중치 초기화 방법 선택       #인자에서  'random'을 미리 대입시켜두면 함수를 부를 때 기본값으로 들어감
            case 'uniform'  :
                self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
            case 'xavier'   :
                self.weights =  np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)    # Xavier 초기화
            case 'he'       :
                self.weights =  np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)    # He 초기화
            case 'gaussian' :
                self.weights = np.random.randn(n_inputs, n_neurons) * 0.01                      # Gaussian 초기화
            case _: #파이썬에서 _는 아무것도 입력되지 않은 상태를 의미함
                self.weights = np.random.randn(n_inputs, n_neurons)  # 기본 랜덤 초기화
                
        self.biases = np.random.uniform(0, 1, (1,n_neurons))

    def forward(self, inputs, activation_Func= 'random'):
        layers_outputs = np.dot(inputs, np.array(self.weights)) + self.biases

        match activation_Func:    # 활성화 함수 선택
            case 'sigmoid' :
                return 1 / ( 1 + np.exp(-layers_outputs) )
            case 'relu'    :
                return np.maximum(0, layers_outputs)
            case 'tanh'    :
                return np.tanh(layers_outputs)

        return layers_outputs

# 샘플 데이터 생성
inputs,y = spiral_data(samples=2, classes=3)
plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap = 'brg')
plt.show()

# Dense 레이어 생성
DNN = Layer_Dense(2,5) #위 샘플은 2차원공간에서 정의되기 때문에 인풋을2로 설정해야함
DNN2 = Layer_Dense(5,3) #인자에 위 가중치 초기화 방법 입력추가가능

outputs = DNN.forward(inputs)  # forward 메서드의 결과를 저장 //activation_Func = 'sigmoid', 'relu', 'tanh' 입력
print(outputs)  # 실제로 forward 메서드의 출력값을 출력
#행렬을 설명하자면  [첫 데이터에 대한 출력값5개]
#                [두번째 데이터에 대한 출력값5개] -------
