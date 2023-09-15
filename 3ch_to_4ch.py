import os
import cv2

img_dir = r'./good'
mask_dir = r'./mask_1ch'
merge_dir = r'./good_4ch'

# merge_dir 디렉토리가 없으면 생성
if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)

file_names = os.listdir(mask_dir)

for file in file_names:
    base_name, ext = os.path.splitext(file)
    # .jpg 확장자로 변경
    new_file_name = base_name + '.jpg'
    
    img_path = os.path.join(img_dir, new_file_name)

    # img_path = os.path.join(img_dir, file)
    mask_path = os.path.join(mask_dir, file)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 1채널 이미지로 읽기

    # 3채널 이미지와 1채널 마스크를 병합하기 위해 마스크의 차원을 확장
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 각 채널별로 이미지와 마스크를 합침
    merged = cv2.addWeighted(img, 1, mask_colored, 0.5, 0)

    merge_path = os.path.join(merge_dir, file)
    cv2.imwrite(merge_path, merged)

print("모든 이미지 변환 및 저장이 완료되었습니다.")



