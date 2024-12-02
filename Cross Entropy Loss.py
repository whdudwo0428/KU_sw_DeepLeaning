import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

def categorical_cross_entropy(predictions, targets):
    '''
    :param predictions: Dense Later output -> Softmax 취한 출력
    :param targets: 정답지, one-hot encoding
    :return: categorical cross entropy loss 연산값
    '''

    # Clip predictions to prevent log(0)
    predictions = np.clip(predictions,1e-7, 1 - 1e-7)
    # clip : 자르다! 식을 보면 로그가 있는데 로그0은 없잖, 정의x
    # 컴터가 알아먹을 수 있게 "범위를 정해줌"
    ''' e는 10을 의미 e-7 = 10^7
    if predictions == 0:
        predictions = 1e-7
    '''

    # If targets are sparse
    if targets.ndim == 1:
        correct_confidences = predictions[np.arange(len(predictions)), targets]
    else:
        # if targets are one-hot encoded
        correct_confidences = np.sum(predictions * targets, axis=1)

    # Calculate negative log likehood
    negative_log_likehood = - np.log(correct_confidences)

    # Calulate avergae loss
    return np.mean(negative_log_likehood)

# Example usage
softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.2, 0.2, 0.6]
])
targets = np.array([0, 1, 2])

loss = categorical_cross_entropy(softmax_outputs, targets)
print("Categorical Cross-Entropy Loss:", loss)
