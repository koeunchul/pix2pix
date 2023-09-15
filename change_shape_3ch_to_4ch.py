import os
import cv2
import numpy as np

img_dir = r'./datasets/capsule/good_ori'
mask_dir = r'./datasets/capsule/good_mask'
merge_dir = r'./datasets/capsule/good_4ch'

# merge_dir 디렉토리가 없으면 생성
if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)

file_names = os.listdir(mask_dir)

for file in file_names:
    base_name, ext = os.path.splitext(file)
    # .jpg 확장자로 변경
    new_file_name = base_name + '.png'
    
    img_path = os.path.join(img_dir, new_file_name)
    mask_path = os.path.join(mask_dir, file)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지와 마스크의 차원을 확장하여 4차원으로 만들기
    img_4d = np.expand_dims(img.transpose(2, 0, 1), axis=0)  
    mask_4d = np.expand_dims(np.expand_dims(mask, axis=0), axis=1) 

    # RGB 이미지와 마스크를 4차원으로 합침
    merged_4d = np.concatenate((img_4d, mask_4d), axis=1)

    # 4차원 배열을 다시 3차원으로 줄임 (이미지 저장을 위해)
    merged_3d = merged_4d.squeeze(0).transpose(1, 2, 0)

    merge_path = os.path.join(merge_dir, file)
    cv2.imwrite(merge_path, merged_3d)

print("모든 이미지 변환 및 저장이 완료되었습니다.")
