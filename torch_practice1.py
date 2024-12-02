# 셸로우 뉴럴 넷
import numpy as np
import torch
import torch.nn as nn
import torchvision

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b

y.backward()

print(x.grad, w.grad, y.grad)

x = torch.randn(10, 3)
y = torch.randn(10, 2)

linear = nn.Linear(3, 2)

print('w: ', linear.weight)
print('b: ', linear.bias)

criterion = nn.MSELoss()  # mean Square Error Loss
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)  # lr decay를 줄여줘야 수렴한다.

for i in range(100):
    pred = linear(x)  # forward pass

    loss = criterion(pred, y)  # loss 정해줌
    print('loss:', loss.item())

    loss.backward()  # 역전파로 grad 생성

    print('dL/dw : ', linear.weight.grad)
    print('dL/db : ', linear.bias.grad)

    optimizer.step()  # 데이터 전체에 대해 한 epoch 돌았다.

    pred = linear(x)
    loss = criterion(pred, y)
    print(loss.item())

x= np.array([[1,2], [3,4]])
y= torch.from_numpy(x)  # 넘파이가 텐서로 바뀜
z = y.numpy()   #텐서가 쉽에 넘파이 어레이로 바뀜

print("A")

# 실무에서는 pandas로 만들어두고 속도가 빠른 numpy로 저장해두고 불러올때 넘파이로 불러온 후 토치로 바꾼 후 사용
