import torch
import torch.nn as nn # deep learning 구성하는 모듈
import numpy as np
import matplotlib.pyplot as plt

'''
ex) X:몸무게 , Y:키
'''
input_size = 1
output_size = 1
num_epochs = 300
learning_rate = 0.001

'''
Toy dataset
'''
x_train = np.array([ [60.0],[87.5],[70.0],[77.0],[101.2],[48.9],[51.2]], dtype=np.float32)
y_train = np.array([ [165.0],[180.0],[168.0],[175.2],[177.2],[150.2],[151.3] ], dtype=np.float32)
# 정규화가 안됨 위에 보면 크게 적어서 가우시안형태가 안됨 그래서   평균/분산
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
y_train = (y_train - np.mean(y_train)) / np.std(y_train)

'''
Linear Regression
'''
model= nn.Linear(input_size,output_size)

'''
Loss and optimizer
'''
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
Train the model
'''
for epoch in range(num_epochs):
    # Convert np to Torch
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    # Forward Pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # train log
    if (epoch +1) % 5 == 0:
        print('Epoch {}/{}, Loss: {:.4f}'
              .format(epoch+1, num_epochs, loss.item()))

predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train,y_train, 'ro', label='Original Data')
plt.plot(x_train, predicted, label='Original Data')
plt.legend()
plt.show()