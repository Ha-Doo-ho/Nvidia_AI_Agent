import random
import numpy as np
from torch.cuda import is_available
import torch
from torch import tensor, cat
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn import (
    Module, Sequential, Conv2d, ReLU, BatchNorm2d,
    MaxPool2d, Upsample, Softplus, MSELoss, ConvTranspose2d, Sigmoid, 
)
from torch.optim import Adam
from torch.nn.functional import interpolate
from pathlib import Path
from collections import Counter
import os
from PIL import Image
import glob 
import cv2

# 프로젝터 폴더 기준으로 원본 폴더와 결과 폴더를 지정한다.
# IMGS_DIR = "./_data/keggle_dataset1/images"
# MASKS_DIR = "./_data/keggle_dataset1/masks"
# OUTPUT_DIR = "./_data/keggle_dataset1/output"
# TARGET_SIZE = 512

# # 두 폴더에서 TIFF 경로를 찾고, 같은 파일명 순으로 정렬한다. 
# img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
# mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

# img_paths.sort()
# mask_paths.sort()

# 자른 타일을 저장할 output 폴더를 처음 한 번 생성한다.
# os.makedirs(OUTPUT_DIR)
# for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
#     # 경로에서 폴더와 확장자를 떼고 원본 파일명만 가져온다.
#     img_filename = os.path.splitext(os.path.basename(img_path))[0] # 베이스이름 기준으로 나눈다. 베이스이름은 가장 마지막 / 뒤에 붙어있는 것이다. .뒤는 확장자이므로 
#     mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    
#     # 항공사진을 읽고, 마스크는 단일 채널과 정답 번호를 유지해 읽습니다.
#     img = cv2.imread(img_path)
#     mask = cv2.imread(mask_path)

#     assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

#     k = 0
#     # 위에서 아래로, 각 줄에서는 왼쪽에서 오른쪽으로 512픽셀씩 이동합니다.
#     for y in range(0, img.shape[0], TARGET_SIZE):
#         for x in range(0, img.shape[1], TARGET_SIZE):
#             img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
#             mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

#             # 가장자리에서도 가로·세로가 모두 512픽셀인 타일만 저장합니다.
#             if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
#                 out_img_path = os.path.join(OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k))
#                 cv2.imwrite(out_img_path, img_tile)

#                 out_mask_path = os.path.join(OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k))
#                 cv2.imwrite(out_mask_path, mask_tile)

#             k += 1

#     print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))

# 1. 데이터 
output_dir = Path("./_data/keggle_dataset1/output")
image_paths = sorted(output_dir.glob("*.jpg"))[:8]

if not image_paths:
    raise FileNotFoundError(f"영상 타일이 없습니다: {output_dir.resolve()}")

# 영상 경로와 그에 대응하는 마스크 경로를 한 쌍으로 묶습니다. -> TensorDataset
practice_pairs = [
    (path, path.with_name(f"{path.stem}_m.png"))
    for path in image_paths
]


# 영상 한 장(문제)과 마스크 한 장(답)을 텐서로 변환
def read_pair(image_path, mask_path):
    # 파일 한 쌍을 열어 x와 y를 만듭니다.
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB")) # RGBA --> RGB
    with Image.open(mask_path) as mask: 
        labels = np.array(mask.convert("L")) # RGB --> L

    x = tensor(data=rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0 
    y = tensor(data=(labels == 3), dtype=torch.float32).reshape(1, 512, 512) # 수역이면 1 아니면 0을 반환
    return x, y


class WaterDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    # 같은 인덱스의 영상(문제)과 마스크(답)를 읽고 return x, y 로 한쌍을 반환
    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        return read_pair(image_path, mask_path)


practice_dataset = WaterDataset(practice_pairs)

# 먼저 한 쌍을 직접 꺼내 확인합니다.
x, y = practice_dataset[0]
print("한 쌍:", x.shape, y.shape)

# 그다음 DataLoader가 여러 쌍을 배치로 묶습니다.
practice_loader = DataLoader(practice_dataset, batch_size=4, shuffle=True)
x_batch, y_batch = next(iter(practice_loader))
print("첫 배치:", x_batch.shape, y_batch.shape)

# 2 모델구성
class CustomCNN(Module):
    def __init__(self):
        super(CustomCNN).__init__()
        
        self.features = Sequential(
            Sequential(
                Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding="same"),
                ReLU(),
                BatchNorm2d(num_features=8),
                MaxPool2d(kernel_size=2)
            ),
            
            Sequential(
                Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding="same"),
                ReLU(),
                BatchNorm2d(num_features=16),
                MaxPool2d(kernel_size=2)
            ),
            
            Sequential(
                Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
                ReLU(),
                BatchNorm2d(num_features=32),
                MaxPool2d(kernel_size=2)
            )
        )
        
        self.fusion2 = Sequential(
            Conv2d(in_channels=32 + 16, out_channels=16, kernel_size=3, padding="same"),
            ReLU()
        )
        
        self.fusion1 = Sequential(
            Conv2d(in_channels=16 + 8, out_channels=8, kernel_size=3, padding="same"),
            ReLU() 
        )
        
        self.final_upsample = Sequential(
            Upsample(scale_factor=2, str="bilinear", align_corners=False),
            Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding="same"),
            ReLU()
        )
        
        # 각 픽셀의 특징 8개를 수역 점수 하나로 바꿉니다.
        self.water_head = Conv2d(8, 1, kernel_size=1) # 수역이냐 아니냐, 이진분류
    
    def extract_features(self, x):
        f1 = self.features[0](x)   # [B,  8, 256, 256]
        f2 = self.features[1](f1)  # [B, 16, 128, 128]
        f3 = self.features[2](f2)  # [B, 32,  64,  64]    