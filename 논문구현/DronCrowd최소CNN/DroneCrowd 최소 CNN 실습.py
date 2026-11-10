"""DroneCrowd 논문의 핵심 개념을 학습하는 교육용 최소 예제.

머리 위치(point) -> 정답 density map -> 작은 CNN의 density map 예측.
실제 STNNet 전체 구현이나 논문 성능 재현이 아닙니다.
실행: python dronecrowd_customcnn_style.py
필요 패키지: torch

Density Map 예측 부분만 단순화한 실습 모델입니다.
"""
import random

import torch
from torch import nn
from torch.nn import Module, Sequential, Conv2d, ReLU, BatchNorm2d, MaxPool2d, Upsample, Softplus
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# 1. 정답 밀집도 지도(GT Density Map) 생성
#    사람 머리 좌표 하나마다 Gaussian을 1개 그리고, 각 Gaussian의 합을 1로 맞춥니다.
def make_density_map(head_points, img_size=32, sigma=1.5):
    density = torch.zeros(1, img_size, img_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32),
        torch.arange(img_size, dtype=torch.float32),
        indexing="ij",
    )

    for x, y in head_points:
        gaussian = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()  # 머리 1개 = Density Map 합계 1
        density[0] += gaussian
    return density  # (1, 32, 32)


# 2. 사용자님의 CustomCNN 코딩 스타일 유지: Module + __init__ + Sequential + forward
class CustomCNN(Module):  # JAVA의 class CustomCNN extends Module와 비슷한 상속
    def __init__(self):
        super(CustomCNN, self).__init__()

        """__init__에서는 계층을 선언하고, forward에서 실제 호출합니다."""

        # 특징 추출 층: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=8),
            MaxPool2d(kernel_size=2),

            Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=16),
            MaxPool2d(kernel_size=2),

            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2),
        )

        # 기존 분류 모델의 Flatten + Linear 대신(위치를 보존하지 않음.), 위치가 보존되는 Density Head 사용
        # 4x4 -> 8x8 -> 16x16 -> 32x32
        self.density_head = Sequential(
            Upsample(scale_factor=2, mode="bilinear", align_corners=False), # 확장을 담당한다. 전부 더했을 때 1이어야 하므로 코너값 고정인 align_coners를 꺼야 한다. 
            Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same"), # Upsample이 지도 크기를 확대한다면 Conv2d는 확대된 특징을 학습 가능한 필터로 변환한다.
            ReLU(),

            Upsample(scale_factor=2, mode="bilinear", align_corners=False), # 주의! Upsample은 해상도를 키우는 것이지, MaxPool에서 사라진 원본의 세부 정보를 완벽하게 복구하는 것은 아님
            Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding="same"),
            ReLU(),

            Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv2d(in_channels=8, out_channels=1, kernel_size=1), # Conv2d의 출력 채널이 1이다. 이 계층은 최종적으로 8개 특징 채널을 Density Map "한 장"으로 변환한다. 
            Softplus(beta=1),  # Density Map의 음수 값을 방지. Density값이 음수가 되면 사람이 음수 기여도를 가진다는 것인데, 그건 말이 안됨. 그래서 넣어논 것.
        )

    def forward(self, x):  # 실제 입력: (배치, 3, 32, 32)
        x = self.features(x)       # (배치, 32, 4, 4)
        x = self.density_head(x) / 500.0  # 밀집도 값 범위 조정, (배치, 1, 32, 32)
        return x


# 3. 원리 확인용 가짜 이미지와 정답 Density Map 제작
#    실제 DroneCrowd 사진이 아니라 Gaussian 점으로 만든 합성 이미지입니다.
def make_toy_dataset(num_images=80, img_size=32):
    images = []
    density_maps = []

    for _ in range(num_images):
        num_people = random.randint(1, 5)
        head_points = [
            (random.randint(3, img_size - 4), random.randint(3, img_size - 4))
            for _ in range(num_people)
        ]
        target_density = make_density_map(head_points, img_size=img_size)

        # RGB 이미지의 밝은 점이 사람 머리라고 가정하는 초간단 실습 데이터
        image = torch.zeros(3, img_size, img_size)
        image[0] = (target_density[0] * 12).clamp(0, 1)
        image[1] = (target_density[0] * 8).clamp(0, 1)
        image[2] = (target_density[0] * 4).clamp(0, 1)
        images.append(image)
        density_maps.append(target_density)

    return torch.stack(images), torch.stack(density_maps)


# 4. 학습: map의 모양과 총 인원 수를 동시에 맞히기
#    데모에서 흔한 '전부 0' 예측을 줄이기 위해 count loss도 사용합니다.
def train_model(model, dataloader, device, epochs=6):
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y_density in dataloader:
            x = x.to(device)
            y_density = y_density.to(device)
            
            model.train()
            pred_density = model(x)
            map_loss = loss_fn(pred_density, y_density)
            
            pred_count = pred_density.sum(dim=(1, 2, 3))
            gt_count = y_density.sum(dim=(1, 2, 3))
            count_loss = (pred_count - gt_count).abs().mean()
            loss = map_loss + 0.01 * count_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | loss={epoch_loss / len(dataloader):.4f}")


# 5. 평가: 학습 때 쓰지 않은 이미지로 Density Map의 합을 확인
@torch.no_grad()
def evaluate(model, image, target_density, device):
    model.eval()
    pred_density = model(image.unsqueeze(0).to(device))
    pred_count = pred_density.sum().item()
    gt_count = target_density.sum().item()
    print(f"테스트 결과 | 실제 인원 수: {gt_count:.1f}명 | 예측 인원 수: {pred_count:.2f}명")
    print(f"예측 Density Map shape: {tuple(pred_density.shape)}")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_images, train_maps = make_toy_dataset(num_images=80)
    test_images, test_maps = make_toy_dataset(num_images=8)
    train_loader = DataLoader(TensorDataset(train_images, train_maps), batch_size=16, shuffle=True)

    model = CustomCNN().to(device)
    train_model(model, train_loader, device, epochs=18)
    evaluate(model, test_images[0], test_maps[0], device)
