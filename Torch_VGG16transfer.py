import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = Compose([
    Resize(224),  # img를 224크기로 바꾸고
    RandomCrop((224, 224), padding=4),
    RandomHorizontalFlip(p=0.5),  # 50%확률로 가로로 뒤집음
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465),
              std=(0.247, 0.243, 0.261))  # RGP세 채널에 대해 각 값을 주어야하기때문에 세개씩 있는거
])

model = vgg16(pretrained=True)  # ① vgg16 모델 객체 생성
fc = nn.Sequential(  # ② 분류층 정의
    nn.Linear(512 * 7 * 7, 4096),
    nn.BatchNorm1d(32), # 32:Batch size
    nn.ReLU(),
    nn.Dropout(),  # ③ 드롭아웃층 정의
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 10),
)

model.classifier = fc  # ④ VGG의 classifier를 덮어씀
model.to(device)

training_data = CIFAR10(root='/n', train=True,
                        download=True, transform=transforms)
test_data = CIFAR10(root='/n', train=False,
                    download=False, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(training_data, batch_size=32, shuffle=False)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader)
    # 반복문으로 찍히는 값을 이쁘게 보여줌
    for data, label in iterator:
        optim.zero_grad()
        preds = model(data.to(device))
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")