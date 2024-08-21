# 랜덤 시드 설정
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#1 데이터 준비

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CellImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset = CellImageDataset(image_dir="class_a3+", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


#2
from diffusers import DDPMPipeline, DDPMScheduler
import torch.optim as optim

# 모델 ID 설정
model_id = "google/ddpm-celebahq-256"

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 및 디바이스로 이동
ddpm = DDPMPipeline.from_pretrained(model_id, use_safetensors=False)
ddpm.to(device)

# Optimizer 및 Loss Function
optimizer = optim.Adam(ddpm.unet.parameters(), lr=1e-5)  # Learning rate 조정 가능


# 3
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 학습 파라미터 설정
num_epochs = 10  # 학습할 epoch 수
log_interval = 10  # 로그를 출력할 interval

# 모델을 학습 모드로 전환
ddpm.unet.train()

# 모델 저장 경로 설정 및 폴더 생성
model_save_path = "path_to_save_model"
os.makedirs(model_save_path, exist_ok=True)

# 학습 루프
for epoch in range(num_epochs):
    for batch_idx, images in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        images = images.to(device)
        
        # Forward pass
        noise = torch.randn_like(images).to(device)
        timesteps = torch.randint(0, ddpm.scheduler.num_train_timesteps, (images.size(0),), device=device).long()
        noisy_images = ddpm.scheduler.add_noise(images, noise, timesteps)
        
        # 모델 출력 계산
        outputs = ddpm.unet(noisy_images, timesteps)["sample"]
        
        # 손실 계산
        loss = F.mse_loss(outputs, noise)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.6f}")

    # 에포크 끝날 때마다 모델 저장
    torch.save(ddpm.unet.state_dict(), os.path.join(model_save_path, f"model4_epoch_{epoch+1}.pth")) #저장될 .pth 이름 설정(업데이트 안하면 덮어쓰임)

#4 모델 저장 
torch.save(ddpm.unet.state_dict(), os.path.join(model_save_path, "model4.pth"))

# #5 학습 완료 후 샘플 데이터 생성
# ddpm.unet.eval()
# sample_images = ddpm(num_inference_steps=50)["sample"]
# sample_images = sample_images.permute(0, 2, 3, 1)  # NCHW to NHWC
# sample_images = (sample_images + 1) / 2.0  # [-1, 1] -> [0, 1]
# sample_images = sample_images.clamp(0, 1)

# # 샘플 이미지 저장
# os.makedirs("generated_images", exist_ok=True)
# for i, img in enumerate(sample_images):
#     plt.imsave(f"generated_images/sample_{i}.png", img.cpu().numpy())
