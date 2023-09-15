import torchvision.transforms as transforms
from PIL import Image
import os

dir_path = './datasets/capsule/test_mask'
files = [f for f in os.listdir(dir_path)]
for file in files:
    # 이미지를 로드
    img_path = os.path.join(dir_path, file)
    img = Image.open(img_path)

    # 4채널 이미지를 3채널 (RGB)로 변환
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # 리사이즈된 이미지 저장
    base_name, extension = os.path.splitext(file)  # Remove the extension
    # if extension == '.jpg':
    #     os.remove(img_path)

    save_path = os.path.join(dir_path, base_name + '.png')
    img.save(save_path, 'PNG')  # Specify the format as PNG
print("All images have been resized to 256x256!")
