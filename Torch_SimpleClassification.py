import torch
import torch.nn as nn # deep learning 구성하는 모듈
import torchvision
import torchvision.transforms as transforms

'''
ex) 손글씨 데이터셋
input : 28*28 img
class : 10개 (0-9)
'''
input_size = 28*28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

'''
MNIST dataset
'''
train_dataset = torchvision.datasets.MNIST(root='.1/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='.1/data',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

train_loader = torch.utils.data.DataLoader(dataset= train_dataset,
                                           batch_size= batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset,
                                           batch_size= batch_size,
                                           shuffle=True)
# 셔플 : 데이터를 섞어서 넣는거

'''
Linear Regression
'''
model= nn.Linear(input_size, num_classes)

'''
Loss and optimizer
'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
train the model
'''
total_step = len(train_loader) #전체 중 몇번째 배치를 돌고있는지 로그용
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)
        # (28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train log
        if (i +1) % 100 == 0:
            print('Epoch {}/{}, Step{}/{}, Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad(): # 추론속도를 안 늦어지도록
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) #_는 max의 리턴을 보면 값,인뎃스가 들어옴 값을 저장을 안하는거/ 받기는하는데 저장은 안한다!
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print("acc {}%".format(100*correct/total))

