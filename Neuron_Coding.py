"""
입력의 개수를 탐지하여 random하게 weight를 생성하는 함수
random범위 (-1~1)
입력 inputs
출력 weights
함수명 init_weight
"""
import random

def init_weight(inputs):  # 먼저 함수 명 선언  함수명(인풋)
    weights = [random.uniform(-1, 1) for i in range(inputs)]
    return weights

'''
  입력의 값과 weight 그리고 bias를 입력 받아 결과 출력하는 함수 만들기
• 입력 inputs, weights, bias
• 출력 output
• 함수명 cal
'''
def cal(inputs, weights, bias):
    output = [inputs[i] * weights[i] + bias for i in range(len(inputs))]
    return output

'''
 위 두 함수를 활용하여 neuron의 개수를 입력받기
 입력 num_neuron, inputs
 출력 outpus
 함수면 cal_neuron
'''
def cal_neuron(num_neuron, inputs):
    outputs = []    # outputs 리스트 초기화
    for i in range(num_neuron):    # num_neuron만큼 반복
        weights = init_weight(len(inputs)) # 랜덤 가중치
        bias = 1
        outputs.append(cal(inputs, weights, bias))
    return outputs

num_neuron = 3
inputs = [0.5, -0.2, 0.1]
outputs = cal_neuron(num_neuron, inputs)
print(outputs)