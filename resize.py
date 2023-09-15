import torchvision.transforms as transforms
from PIL import Image
import os

resize_size = 256
dir_path = './datasets/capsule/test_mask'

# 이미지 전처리를 위한 transform 정의
resize_transform = transforms.Compose([transforms.Resize((resize_size, resize_size))])
files = [f for f in os.listdir(dir_path)]
for file in files:
    # 이미지를 로드
    img_path = os.path.join(dir_path, file)
    img = Image.open(img_path)

    # 이미지를 256x256 크기로 resize
    img_resized = resize_transform(img)
    
    # 리사이즈된 이미지 저장
    base_name, extension = os.path.splitext(file)  # Remove the extension
    # if extension == '.jpg':
    #     os.remove(img_path)

    save_path = os.path.join(dir_path, base_name + '.png')
    img_resized.save(save_path, 'PNG')  # Specify the format as PNG
print("All images have been resized to 256x256!")