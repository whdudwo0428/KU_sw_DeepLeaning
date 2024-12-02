# 셸로우 뉴럴 넷
import numpy as np
import torch
import torch.nn as nn
import torchvision

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

images, lables = train_dataset[0]
print(images.size())  # Tensor: (3,32,32)  각각이 rgb마냥 C H W 값
print(lables)

train_loader = torch.utils.data.DataLoader(  # batch size대로 자르는걸 도와주는 친구가 로더임
    dataset=train_dataset,
    batch_size=64,
    shuffle=True  # dataset의 순서를 뒤섞어줌
)

data_iter = iter(train_loader)

images, lables = data_iter.next()

for images, lables in train_loader:
    pass


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    # x = 입력데이터 뭉치 # np.array(100000,4)
    # y = 정답데이터 뭉치 # np.array(100000,2)

    def __getitem__(self, index):
        pass
    # trainload에서 batch단위로 데이터를 불러오는데 각각 데이터의 번호가 부여되어있는데 안에 index를 이용해 뭐 아래처럼 network에 들어감
    # return x[index], y[index]
    def __len__(self):
        pass

# return len(self.x)  #dataload에 batch size를 보고 짤라서 넣을텐데  만약 100개를 만개단위로 자르면 실제 없는 데이터를 자를 수 없으니 에러남 얼마나 접근 가능한지 제한


# pretrained_model
resnet = torchvision.models.resnet18(pretrained=True)  # 미리 학습된 데이터를 불러옴

for param in resnet.parameters():
    param.requires_grad = False
    #분류 검출할때 데이터들이 다 비슷비슷함 특징들이 비슷할거아님 resnet이라는 모델이 구지 다른 학습데이터를 뭐 어쩌구 할 필요가없음

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt') #ckpt는 체크포인트의 약자

