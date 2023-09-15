import os
from PIL import Image

# 이미지 경로 설정
mask_dir = r'./mask/good'
mask_dir_1ch = r'./mask_1ch'

# mask_dir_1ch 디렉토리가 없으면 생성
if not os.path.exists(mask_dir_1ch):
    os.makedirs(mask_dir_1ch)

# mask_dir에서 모든 jpg 파일을 가져옴
image_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

# 각 이미지에 대해 1 채널로 변환 후 저장
for image_file in image_files:
    img_path = os.path.join(mask_dir, image_file)
    # 이미지 로드
    img = Image.open(img_path)
    # 1 채널로 변환
    img_1ch = img.convert('L')
    # 변환된 이미지 저장
    save_path = os.path.join(mask_dir_1ch, image_file)
    img_1ch.save(save_path)

print("모든 이미지 변환 및 저장이 완료되었습니다.")
