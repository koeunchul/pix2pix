import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

class Train_Dataset(Dataset):
    def __init__(self, ng, ok, mask, transforms_=None):
        self.transform = transforms_
        self.ng     = ng
        self.ok     = ok      
        self.mask   = mask        

    def __getitem__(self, index):
        img_ng      = Image.open(self.ng[index % len(self.ng)])
        img_ok      = Image.open(self.ok[index % len(self.ok)])
        img_mask    = Image.open(self.mask[index % len(self.mask)])

        img_ng      = self.transform(img_ng)
        img_ok      = self.transform(img_ok)
        img_mask    = self.transform(img_mask)

        return {"ng": img_ng, "ok": img_ok, "mask": img_mask}

    def __len__(self):
        return len(self.ok)    

class Test_Dataset(Dataset):
    def __init__(self, ok, mask, transforms_=None):
        self.transform = transforms_
        self.ok     = ok      
        self.mask   = mask      

    def __getitem__(self, index):
        img_ok      = Image.open(self.ok[index % len(self.ok)])
        img_mask    = Image.open(self.mask[index % len(self.mask)])
        
        img_ok      = self.transform(img_ok)
        img_mask    = self.transform(img_mask)

        return {"ok": img_ok, "mask": img_mask}

    def __len__(self):
        return len(self.ok)   


transforms_ = transforms.Compose([
    # transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_ok_dir    = r'./datasets/capsule/train_ok'
train_mask_dir  = r'./datasets/capsule/train_mask'
train_ng_dir    = r'./datasets/capsule/train_ng'

test_ok_dir     = r'./datasets/capsule/test_ok'
test_mask_dir   = r'./datasets/capsule/test_mask'
save_dir        = r'./result_koko'

# save_dir 디렉토리가 없으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_ok_path   = [os.path.join(train_ok_dir, f) for f in os.listdir(train_ok_dir) if f.endswith('.png')]
train_ng_path   = [os.path.join(train_ng_dir, f) for f in os.listdir(train_ng_dir) if f.endswith('.png')]
train_mask_path = [os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith('.png')]
test_ok_path    = [os.path.join(test_ok_dir, f) for f in os.listdir(test_ok_dir) if f.endswith('.png')]
test_mask_path  = [os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir) if f.endswith('.png')]
train_batch_size    = 8
test_batch_size     = 8
ch = 3

train_dataset       = Train_Dataset(ng=train_ng_path, ok=train_ok_path, mask=train_mask_path, transforms_=transforms_)
test_dataset        = Test_Dataset(ok=test_ok_path, mask=test_mask_path, transforms_=transforms_)

train_dataloader    = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
test_dataloader     = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=4)

# U-Net 아키텍처의 다운 샘플링(Down Sampling) 모듈
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        # 너비와 높이가 2배씩 감소
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# U-Net 아키텍처의 업 샘플링(Up Sampling) 모듈: Skip Connection 사용
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        # 너비와 높이가 2배씩 증가
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1) # 채널 레벨에서 합치기(concatenation)

        return x


# U-Net 생성자(Generator) 아키텍처
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False) # 출력: [64 X 128 X 128]
        self.down2 = UNetDown(64, 128) # 출력: [128 X 64 X 64]
        self.down3 = UNetDown(128, 256) # 출력: [256 X 32 X 32]
        self.down4 = UNetDown(256, 512, dropout=0.5) # 출력: [512 X 16 X 16]
        self.down5 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 8 X 8]
        self.down6 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 4 X 4]
        self.down7 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 2 X 2]
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) # 출력: [512 X 1 X 1]

        # Skip Connection 사용(출력 채널의 크기 X 2 == 다음 입력 채널의 크기)
        self.up1 = UNetUp(512, 512, dropout=0.5) # 출력: [1024 X 2 X 2]
        self.up2 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 4 X 4]
        self.up3 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 8 X 8]
        self.up4 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 16 X 16]
        self.up5 = UNetUp(1024, 256) # 출력: [512 X 32 X 32]
        self.up6 = UNetUp(512, 128) # 출력: [256 X 64 X 64]
        self.up7 = UNetUp(256, 64) # 출력: [128 X 128 X 128]

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), # 출력: [128 X 256 X 256]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1), # 출력: [3 X 256 X 256]
            nn.Tanh(),
        )

    def forward(self, x):
        # 인코더부터 디코더까지 순전파하는 U-Net 생성자(Generator)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# U-Net 판별자(Discriminator) 아키텍처
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            # 너비와 높이가 2배씩 감소
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 두 개의 이미지(실제/변환된 이미지, 조건 이미지)를 입력 받으므로 입력 채널의 크기는 2배
            *discriminator_block(in_channels * 2, 64, normalization=False), # 출력: [64 X 128 X 128]
            *discriminator_block(64, 128), # 출력: [128 X 64 X 64]
            *discriminator_block(128, 256), # 출력: [256 X 32 X 32]
            *discriminator_block(256, 512), # 출력: [512 X 16 X 16]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False) # 출력: [1 X 16 X 16]
        )

    # img_A: 실제/변환된 이미지, img_B: 조건(condition)
    def forward(self, img_A, img_B):
        # 이미지 두 개를 채널 레벨에서 연결하여(concatenate) 입력 데이터 생성
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 생성자(generator)와 판별자(discriminator) 초기화
generator       = GeneratorUNet(in_channels=ch, out_channels=ch)
discriminator   = Discriminator(in_channels=ch)

generator.cuda()
discriminator.cuda()

# 가중치(weights) 초기화
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 손실 함수(loss function)
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

criterion_GAN.cuda()
criterion_pixelwise.cuda()

# 학습률(learning rate) 설정
lr = 0.0002

# 생성자와 판별자를 위한 최적화 함수
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


n_epochs        = 30000 # 학습의 횟수(epoch) 설정
sample_interval = 100 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정

# 변환된 이미지와 정답 이미지 사이의 L1 픽셀 단위(pixel-wise) 손실 가중치(weight) 파라미터
lambda_pixel = 100
start_time = time.time()

for epoch in range(n_epochs):
    for i, batch in enumerate(train_dataloader):
        # 모델의 입력(input) 데이터 불러오기
        train_ng    = batch["ng"].cuda()
        train_ok    = batch["ok"].cuda()
        train_mask  = batch["mask"].cuda()

        # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성 (너바와 높이를 16씩 나눈 크기)
        real = torch.cuda.FloatTensor(train_ng.size(0), 1, 16, 16).fill_(1.0) # 진짜(real): 1
        fake = torch.cuda.FloatTensor(train_ng.size(0), 1, 16, 16).fill_(0.0) # 가짜(fake): 0

        """ 생성자(generator)를 학습합니다. """
        optimizer_G.zero_grad()

        # 이미지 생성
        train_gen = generator(train_ok)
        

        # 생성자(generator)의 손실(loss) 값 계산
        loss_GAN = criterion_GAN(discriminator(train_gen, train_ok), real)

        # 픽셀 단위(pixel-wise) L1 손실 값 계산
        loss_pixel = criterion_pixelwise(train_gen, train_ng) 

        # 최종적인 손실(loss)
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        # 생성자(generator) 업데이트
        loss_G.backward()
        optimizer_G.step()

        """ 판별자(discriminator)를 학습합니다. """
        optimizer_D.zero_grad()

        # 판별자(discriminator)의 손실(loss) 값 계산
        loss_real = criterion_GAN(discriminator(train_ng, train_ok), real) # 조건(condition): real_A
        loss_fake = criterion_GAN(discriminator(train_gen.detach(), train_ok), fake)
        loss_D = (loss_real + loss_fake) / 2

        # 판별자(discriminator) 업데이트
        loss_D.backward()
        optimizer_D.step()

        done = epoch * len(train_dataloader) + i
        if done % sample_interval == 0:
            imgs = next(iter(test_dataloader))
            test_ok = imgs["ok"].cuda()
            test_gen = generator(test_ok)
            
            train_sample = torch.cat((train_ok.data, train_ng.data, train_gen.data), -2)
            save_image(train_sample, os.path.join(save_dir, f"train_{done}.png"))
            
            test_sample = torch.cat((test_ok.data, test_gen.data), -2)
            save_image(test_sample, os.path.join(save_dir, f"test_{done}.png"))

    # 하나의 epoch이 끝날 때마다 로그(log) 출력
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}, adv loss: {loss_GAN.item()}] [Elapsed time: {time.time() - start_time:.2f}s]")
     

    
# 모델 파라미터 저장
torch.save(generator.state_dict(), "Pix2Pix_Generator_for_Facades.pt")
torch.save(discriminator.state_dict(), "Pix2Pix_Discriminator_for_Facades.pt")
print("Model saved!")