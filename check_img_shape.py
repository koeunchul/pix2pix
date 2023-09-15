import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

dir_path = './datasets/capsule/test_mask'
files = [f for f in os.listdir(dir_path)]
for file in files:
    # 이미지를 로드
    img_path = os.path.join(dir_path, file)
    img = Image.open(img_path)
    
    # 이미지를 numpy 배열로 변환
    img_np = np.array(img)
    
    print(img_np.shape)  # 이미지의 (height, width, channels)를 출력
    print(dir_path)
