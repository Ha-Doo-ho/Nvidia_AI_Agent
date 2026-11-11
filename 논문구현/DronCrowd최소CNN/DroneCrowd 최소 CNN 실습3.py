"""DroneCrowd (2105.02440v1.pdf), Figure 3 학습용 최소 구현.

목표:
  - 하나의 CNN을 연속된 두 프레임에 재사용: 공유 가중치(Siamese)
  - 3단계 feature map을 skip connection으로 결합: U-Net 스타일 fusion
  - density map과 위치 heatmap 출력
  - 1 epoch train() / evaluate() / 함수 밖 epoch 반복

주의:
  - 논문의 실제 STNNet을 재현한 코드가 아님.
  - VGG-16, 다중스케일 density heads, localization 좌표회귀/NMS,
    correlation, PointConv, association, 논문의 정식 multi-task loss는 생략.
  - 생성한 가짜 RGB 점 이미지로만 동작을 검증한다.

실행: python dronecrowd_stnnet_minimal_step1.py
필요 패키지: torch
"""

import random
import torch
from torch import tensor, cat
from torch.nn import (
    Module, Sequential, Conv2d, ReLU, BatchNorm2d,
    MaxPool2d, Upsample, Softplus, MSELoss, ConvTranspose2d, Sigmoid
)
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda import is_available
from torch.nn.functional import interpolate

# 0 디바이스 결정
if is_available():
    device = "cuda"
else:
    device = "cpu"

# 1. 데이터 (교육용 합성 이미지 & 정답 지도)
# head_points: [(x,y), . . .] 
# Gaussian 하나의 sum이 1명이 되도록 정규화한다.
def make_density_map(head_points, img_size=32, sigma=1.3):
    ys, xs = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32),
        torch.arange(img_size, dtype=torch.float32),
        indexing="ij"
    )
    density_map = torch.zeros(1, img_size, img_size)
    location_map = torch.zeros(1, img_size, img_size)

    for cx, cy in head_points:
        gaussian = torch.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2))
        density_map[0] += gaussian / gaussian.sum()
        # 위치 heatmap은 중심이 1에 가까운 형태 (밀집도 합계와는 별개의 의미)
        location_map[0] = torch.maximum(location_map[0], gaussian)

    return density_map, location_map


def make_toy_data(num_samples=80, img_size=32):
    """두 장의 RGB 이미지를 생성. 실제 드론 영상이 아니며 머리를 밝은 원으로 표현."""
    x_data, density_data, location_data = [], [], []
    ys, xs = torch.meshgrid(
        torch.arange(img_size), torch.arange(img_size), indexing="ij"
    )

    for _ in range(num_samples):
        num_people = random.randint(1, 32)
        dx, dy = random.randint(-2, 2), random.randint(-2, 2)
        points_prev = [
            (random.randint(6, img_size - 7), random.randint(6, img_size - 7))
            for _ in range(num_people)
        ]
        points_curr = [(cx + dx, cy + dy) for cx, cy in points_prev]

        density_prev, location_prev = make_density_map(points_prev, img_size)
        density_curr, location_curr = make_density_map(points_curr, img_size)

        # 지도와 별도로 RGB 합성 이미지 만들기: 픽셀 하나가 아닌 작은 원판을 머리로 표현
        frames = []
        for points in (points_prev, points_curr):
            frame = (torch.rand(3, img_size, img_size) * 0.03) + 0.02
            for cx, cy in points:
                disc = ((xs - cx)**2 + (ys - cy)**2) <= 2**2
                frame[0, disc] = 0.85
                frame[1, disc] = 0.65
                frame[2, disc] = 0.35
            frames.append(frame.clamp(0, 1))

        x_data.append(torch.stack(frames, dim=0))       # (2,3,32,32)
        density_data.append(torch.stack([density_prev, density_curr], dim=0))
        location_data.append(torch.stack([location_prev, location_curr], dim=0))

    return (torch.stack(x_data), torch.stack(density_data),
            torch.stack(location_data))


def makeDataSet(x_train, y_density_train, y_loc_train,
                x_val, y_density_val, y_loc_val, batch_size=8):
    """CIFAR-10과 달리 입력은 이미 (N,2,3,H,W)이고 [0,1]이므로 permute/255 불필요."""
    train_tensor = TensorDataset(
        x_train.float(), y_density_train.float(), y_loc_train.float()
    )
    train_DataLoader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    x_val_tensor = x_val.float()
    y_density_val_tensor = y_density_val.float()
    y_loc_val_tensor = y_loc_val.float()
    return train_DataLoader, x_val_tensor, y_density_val_tensor, y_loc_val_tensor


# 2. 모델 구성 
class CustomCNN(Module):
    def __init__(self):
        super(CustomCNN,self).__init__()
        """__init__ 에서 계층 정의, forward에서 실제 사용."""
        
        # 2-1. 특징 추출: 동일한 self.features를 두 프레임(t-1, t)에 적용 → 가중치 공유
        # ※ 논문 구성의 공유 VGG-16 두 갈래이다. 
        # U-Net그림에서는 파란색 박스 하나가 Conv(3x3), BatchNorm, Relu를 하나로 묶은 것이다. 
        # U-Net그림 보면 그 파란 박스가 두개씩 있다. 그래서  Conv(3x3), BatchNorm, Relu 를 한번 더 만들면 된다.
        self.features = Sequential(
            Sequential(Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding="same"),
                        ReLU(),
                        BatchNorm2d(num_features=8),
                        MaxPool2d(kernel_size=2)), # features[0]
            
            Sequential(Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding="same"),
                        ReLU(),
                        BatchNorm2d(num_features=16),
                        MaxPool2d(kernel_size=2)), # features[1]

            Sequential(Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
                        ReLU(),
                        BatchNorm2d(num_features=32),
                        MaxPool2d(kernel_size=2)), # features[2]
        )
        
        # 2-2 U-Net 형태 특징 융합: 특징 채널을 변환(48 --> 16 Deconv아님.), 깊은 특징을 확대해 얕은 특징과 cat
        self.fusion2 = Sequential(
            Conv2d(in_channels=32 + 16, out_channels=16, kernel_size=3, padding="same"),
            ReLU(), # 채널의 가로 새로 크기를 2배 늘리는 Upsample(32)과 그걸 더하는(32+16) Concat이 합쳐져 있다.  
        )
        
        self.fusion1 = Sequential(
            Conv2d(in_channels=16 + 8, out_channels=8, kernel_size=3, padding="same"),
            ReLU()
        )
        
        self.final_upsample = Sequential(
            Upsample(scale_factor=2, mode="bilinear", align_corners=False), #Upsample 부분. interpolate와 함께 Upsample(높이/너비 늘이기)구현한 방법이다. 즉, 구현 방법이 2가지인 것.
            Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding="same"),
            ReLU()
        )   # 16 x 16 --> 32 x 32. 마지막의 final_upsample에는 torch.cat이 없으므로 in_channel과 out_channel이 같음.
        
        
        # 2-3. 픽셀별 Density Map 출력 (B,1,32,32)
        self.density_head = Sequential(
            Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding="same"),
            ReLU(),
            Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            Softplus(beta=1)                   # 음수가 아닌 밀집도 값
        )
        torch.nn.init.constant_(self.density_head[-2].bias, -6.0)  # 합성실습 초기화
        
        # 2-4. 사람 머리의 위치를 나타내는 heatmap (단순화된 localization)
        # 원 논문의 classification + x/y regression + NMS와는 다름.
        self.localization_head = Sequential(
            Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding="same"),
            ReLU(),
            Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            Sigmoid()                         # 위치 heatmap: 값 범위 [0,1]
        )
        torch.nn.init.constant_(self.localization_head[-2].bias, -3.0)
        
    def extract_features(self, x):
        """한 프레임의 다중 스케일 CNN 특징 추출 후 U-Net 스타일로 결합."""
        f1 = self.features[0](x)   # (B, 8, 16, 16)
        f2 = self.features[1](f1)  # (B, 16, 8, 8)
        f3 = self.features[2](f2)  # (B, 32, 4, 4)
        
        f3_up = interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False) # 논문과는 약간 다르다. 논문의 Deconv 부분, Upsample부분이 여기이다. 
        
        fused2 = cat(tensors=[f3_up, f2], dim=1)
        fused2 = self.fusion2(fused2)
        
        fused2_up = interpolate(fused2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        
        fused1 = cat([fused2_up, f1], dim=1)   # (B,24,16,16)
        fused1 = self.fusion1(fused1)          # (B,8,16,16)
        
        return self.final_upsample(fused1)   # (B,8,32,32)
    
    def forward(self, x):
            """x shape: (B,2,3,32,32) -> density / localization (B,2,1,32,32)."""
            frame_prev = x[:, 0]             # (B,3,32,32)
            frame_curr = x[:, 1]             # (B,3,32,32)
    
            # 동일한 CNN 사용 = Siamese, 서로 독립된 CNN 두 개를 생성하지 않음.
            features_prev = self.extract_features(frame_prev)
            features_curr = self.extract_features(frame_curr)
    
            density_prev = self.density_head(features_prev)
            density_curr = self.density_head(features_curr)
            loc_prev = self.localization_head(features_prev)
            loc_curr = self.localization_head(features_curr)
    
            pred_density = torch.stack([density_prev, density_curr], dim=1)
            pred_location = torch.stack([loc_prev, loc_curr], dim=1)
            return pred_density, pred_location
    
# 3. 컴파일/훈련: train()은 딱 1 epoch만 실행, epoch 반복은 외부에서 수행
# Loss는 교육용 임시 수식: 논문의 L_den + L_loc + L_ass 재현 아님.
def train(device, model, train_DataLoader, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    num_samples = 0

    for batch_idx, (x, y_density, y_loc) in enumerate(train_DataLoader):
        x = x.to(device)
        y_density = y_density.to(device)
        y_loc = y_loc.to(device)

        # FeedForward
        pred_density, pred_loc = model(x)

        # 단순 학습 Loss: 일단 동작을 확인하고, 다음 차례에 논문 정식 수식을 분석한다.
        map_loss = loss_fn(pred_density, y_density)
        location_loss = loss_fn(pred_loc, y_loc)
        pred_count = pred_density.sum(dim=(2, 3, 4))
        true_count = y_density.sum(dim=(2, 3, 4))
        count_loss = (pred_count - true_count).abs().mean()
        loss = 100.0 * map_loss + 0.5 * location_loss + 0.1 * count_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        num_samples += x.size(0)

    return train_loss / num_samples


# 4. 평가: 각각의 프레임에서 Density Map 전체 합 -> 예측 사람 수
def evaluate(device, model, x_val_tensor, y_density_val_tensor, y_loc_val_tensor, loss_fn):
    model.eval()
    with torch.no_grad():
        x = x_val_tensor.to(device)
        y_density = y_density_val_tensor.to(device)
        y_loc = y_loc_val_tensor.to(device)

        pred_density, pred_loc = model(x)
        val_map_loss = loss_fn(pred_density, y_density).item()
        val_loc_loss = loss_fn(pred_loc, y_loc).item()
        pred_count = pred_density.sum(dim=(2, 3, 4))
        true_count = y_density.sum(dim=(2, 3, 4))
        count_mae = (pred_count - true_count).abs().mean().item()

    return val_map_loss, val_loc_loss, count_mae, pred_density, pred_loc


if __name__ == "__main__":
    # 적은 합성 데이터에서는 CPU 스레드 수를 제한하면 일정하고 빠른 테스트가 가능
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)

    # 1-1. 데이터 생성 / 분리 (64 train / 16 validation)
    x_data, density_data, location_data = make_toy_data(num_samples=80)
    train_DataLoader, x_val_tensor, y_density_val_tensor, y_loc_val_tensor = makeDataSet(
        x_train=x_data[:64], y_density_train=density_data[:64], y_loc_train=location_data[:64],
        x_val=x_data[64:], y_density_val=density_data[64:], y_loc_val=location_data[64:],
        batch_size=8
    )

    # 2. 모델 구성
    model = CustomCNN().to(device=device)

    # 3. 컴파일에 해당: 손실 함수와 최적화 함수 설정
    loss_fn = MSELoss()
    optimizer = Adam(params=model.parameters(), lr=0.002)

    # 모델 구조 검증: 합성 데이터 2프레임의 입력과 출력 차원 확인
    with torch.no_grad():
        pden, ploc = model.eval()(x_val_tensor[:2].to(device))
        print("입력:", tuple(x_val_tensor[:2].shape))
        print("예측 Density Map:", tuple(pden.shape))
        print("예측 Localization Heatmap:", tuple(ploc.shape))
        assert pden.shape == (2, 2, 1, 32, 32)
        assert ploc.shape == (2, 2, 1, 32, 32)
        assert torch.isfinite(pden).all() and pden.min() >= 0
        assert torch.isfinite(ploc).all() and ploc.min() >= 0 and ploc.max() <= 1

    # 4. epoch는 train() 함수 밖에서 관리함 (fit() 만들지 않음)
    epoch = 30
    train_loss_list = []
    val_loss_list = []
    val_mae_list = []

    for i in range(epoch):
        train_loss = train(
            device=device, model=model, train_DataLoader=train_DataLoader,
            loss_fn=loss_fn, optimizer=optimizer
        )
        val_map_loss, val_loc_loss, val_mae, pred_density, pred_loc = evaluate(
            device=device, model=model,
            x_val_tensor=x_val_tensor,
            y_density_val_tensor=y_density_val_tensor,
            y_loc_val_tensor=y_loc_val_tensor, loss_fn=loss_fn
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_map_loss)
        val_mae_list.append(val_mae)
        print(f"Epoch [{i+1:02d}/{epoch}] "
              f"train_loss={train_loss:.5f} "
              f"val_density_MSE={val_map_loss:.6f} "
              f"val_location_MSE={val_loc_loss:.6f} "
              f"val_count_MAE={val_mae:.3f}명")

    # 5. 마지막에 한 이미지의 실제 사람 수 / 예측 사람 수 비교
    pred_counts = pred_density.sum(dim=(2, 3, 4))
    true_counts = y_density_val_tensor.sum(dim=(2, 3, 4))
    print(f"첫 검증 sample, 이전 프레임: 정답 {true_counts[0,0].item():.0f}명, "
          f"예측 {pred_counts[0,0].item():.2f}명")
    print("완료: Siamese + U-Net 방식 Multi-scale Fusion + Density/Localization 실습")