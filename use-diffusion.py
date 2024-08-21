#infrence와 만들 장수를 선택해야함.

import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline
import os
import random

random.seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device 설정 (GPU가 사용 가능한 경우 GPU를 사용하고, 그렇지 않으면 CPU를 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 저장 경로 설정
model_save_path = "path_to_save_model\model4.pth"

# DDPMPipeline 모델 로드(모델 불러오기)
ddpm = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256", use_safetensors=False)  # 원본 모델 ID 사용
ddpm.unet.load_state_dict(torch.load(model_save_path))  # 파인튜닝된 가중치 로드
ddpm.to(device)

# 모델을 평가 모드로 전환
ddpm.unet.eval()

# 총 이미지 생성 수 및 배치당 생성 수
total_images = 100
batch_size = 4  # 기존 코드에서 사용한 배치 크기

# 생성 및 저장할 폴더 설정
os.makedirs("generated_images/1000-1-100", exist_ok=True)

for batch_start in range(0, total_images, batch_size):
    # 현재 배치에서 생성할 이미지 수 결정
    current_batch_size = min(batch_size, total_images - batch_start)
    
    # 이미지 생성
    output = ddpm(num_inference_steps=1000, batch_size=current_batch_size)
    
    # 생성된 이미지 저장
    for i, img in enumerate(output.images):
        image_index = batch_start + i + 1
        image_filename = f"a{image_index:04}.jpg"  # a0001.jpg, a0002.jpg, ... 형식으로 저장
        img.save(f"generated_images/1000-1-100/{image_filename}")
    
    print(f"Generated and saved images {batch_start+1} to {batch_start+current_batch_size}")

print("All images generated and saved.")
