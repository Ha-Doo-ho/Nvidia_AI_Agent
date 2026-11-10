"""DroneCrowd 교육용 최소 실습: 사용자의 PyTorch 코딩 스타일로 구성.

이 파일은 STNNet 전체를 구현하지 않습니다.
학습용 합성 머리 좌표 -> Gaussian 정답 Density Map -> CustomCNN -> Loss -> Backprop -> 평가.
실행: python dronecrowd_customcnn_train_epoch_style.py
필요: torch
"""
import random
import torch
from torch import tensor
from torch.nn import (
    Module, Sequential, Conv2d, ReLU, BatchNorm2d,
    MaxPool2d, Upsample, Softplus, MSELoss, ConvTranspose2d
)
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda import is_available


# 0. 디바이스 결정
if is_available():
    device = "cuda"
else:
    device = "cpu"


# 1. 실제 머리 중심 좌표를 이용하여 정답 Density Map 생성
# 각 사람의 Gaussian 합계를 1로 맞춘다. -> 사람 수 = Density Map 전체 합계
# 입력 head_points : [(x1, y1), (x2, y2), ...]
# 출력 shape : (1, 32, 32)
def make_density_map(head_points, img_size=32, sigma=1.5):
    y_grid, x_grid = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32),
        torch.arange(img_size, dtype=torch.float32),
        indexing="ij"
    )
    density_map = torch.zeros(1, img_size, img_size)

    for x, y in head_points:
        gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()  # 한 사람에 해당하는 값들의 총합 = 1
        density_map[0] += gaussian

    return density_map


# 1-1. 교육용 합성 데이터 생성
# 실제 드론 영상이 아닌, 점의 위치를 RGB 채널에 표현한 아주 간단한 실습 데이터
# 입력 이미지 shape: (N, 3, 32, 32), 정답 지도 shape: (N, 1, 32, 32)
def make_toy_data(num_images=64):
    x_data, y_data = [], []

    for _ in range(num_images):
        n_people = random.randint(1, 5)
        head_points = [
            (random.randint(3, 28), random.randint(3, 28))
            for _ in range(n_people)
        ]
        density_map = make_density_map(head_points)

        # 학습용 RGB 형태의 합성 이미지 (0~1 사이)
        image = torch.zeros(3, 32, 32)
        image[0] = (density_map[0] * 9).clamp(0, 1)
        image[1] = (density_map[0] * 7).clamp(0, 1)
        image[2] = (density_map[0] * 5).clamp(0, 1)

        x_data.append(image)
        y_data.append(density_map)

    return torch.stack(x_data), torch.stack(y_data)


# 1-2. TensorDataset / DataLoader 생성
# x_data는 이미 (N, C, H, W)이며 0~1 범위이므로 permute()와 /255는 하지 않음.
def makeDataSet(x_train, y_train, x_val, y_val, batch_size=8):
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32)

    train_tensor = TensorDataset(x_train_tensor, y_train_tensor)
    train_DataLoader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    return train_DataLoader, x_val_tensor, y_val_tensor


# 2. 모델 구성 : 기존에 사용하던 CustomCNN(Module) 형식 그대로
class CustomCNN(Module):  # JAVA의 class CustomCNN extends Module과 유사
    def __init__(self):
        super(CustomCNN, self).__init__()

        """__init__()에서 계층을 만들고, forward()에서 실제 계층을 사용한다."""

        # 2-1. 특징 추출 층
        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=8),
            MaxPool2d(kernel_size=2),       # 32x32 -> 16x16

            Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=16),
            MaxPool2d(kernel_size=2),       # 16x16 -> 8x8

            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            BatchNorm2d(num_features=32),
            MaxPool2d(kernel_size=2)        # 8x8 -> 4x4
        )

        # 2-2. Density Map 예측 층
        # 분류 모델의 Flatten + Linear 대신, 공간 크기를 유지하는 Conv2d 출력
        self.density_head = Sequential(
            Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 4x4 -> 8x8  #확장을 담당한다. 전부 더했을 때 1이어야 하므로 코너값 고정인 align_coners를 꺼야 한다. 
            Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same"), # Upsample이 지도 크기를 확대한다면 Conv2d는 확대된 특징을 학습 가능한 필터로 변환한다.
            ReLU(),

            Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8x8 -> 16x16
            Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding="same"),
            ReLU(),

            Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16x16 -> 32x32
            Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            Softplus(beta=1)  # Density 값이 음수가 되지 않도록 함
        )

        # 합성 실습에서 처음부터 Density가 큰 양수로 나오지 않도록 하는 초기화
        # 중요: 원 논문에서 지정한 필수 설정은 아님.
        torch.nn.init.constant_(self.density_head[-2].bias, -6.0)

    def forward(self, x): #여기서 실제 모델 형태가 구성된다.
        x = self.features(x)
        x = self.density_head(x)
        return x


# 3. 컴파일 및 훈련: 한 번 호출할 때 1 epoch만 학습
# 사용자가 작성한 train(device, model, train_DataLoader, loss_fn, optimizer) 구조 유지
def train(device, model, train_DataLoader, loss_fn, optimizer):
    train_loss = 0.0
    num_samples = 0

    model.train()
    for batch_idx, (x, y) in enumerate(train_DataLoader):
        x = x.to(device)
        y = y.to(device)  # y는 클래스 번호가 아니라 GT Density Map

        # FeedForward
        # 1. 예측 Density Map 생성
        pred = model(x)
        
        # 2. 정답 지도와 예측 지도 비교
        map_loss = loss_fn(pred, y)  

        # 3 Density Map 전체 합계를 인원 수로 사용
        pred_count = pred.sum(dim=(1, 2, 3))
        true_count = y.sum(dim=(1, 2, 3))
        
        # 4 인원 수 오차 계산
        count_loss = (pred_count - true_count).abs().mean()

        # 5. 교육용 결합 Loss: 논문의 STNNet 원래 Loss와는 다름.
        loss = 100.0 * map_loss + 0.1 * count_loss

        # 6. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 배치 크기가 달라도 올바른 샘플 평균을 계산하기 위해 item() 사용
        train_loss += loss.item() * x.size(0)
        num_samples += x.size(0)

    train_loss /= num_samples
    return train_loss


# 4. 평가: 한 번 호출할 때 전체 validation set 평가
# 분류에서는 Accuracy, 여기서는 Count MAE를 사용함.
def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    model.eval()

    with torch.no_grad():
        x = x_val_tensor.to(device)
        y = y_val_tensor.to(device)

        # Density Map 예측
        pred = model(x)
        map_loss = loss_fn(pred, y)

        pred_count = pred.sum(dim=(1, 2, 3))
        true_count = y.sum(dim=(1, 2, 3))
        count_loss = (pred_count - true_count).abs().mean()

        # train()과 동일한 식으로 val loss 계산
        val_loss = (100.0 * map_loss + 0.1 * count_loss).item()
        val_count_mae = count_loss.item()

    return pred, val_loss, val_count_mae


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)

    # 1-3. 데이터 생성 / 분할
    x_data, y_data = make_toy_data(num_images=64)
    x_train, x_val = x_data[:48], x_data[48:]
    y_train, y_val = y_data[:48], y_data[48:]

    train_DataLoader, x_val_tensor, y_val_tensor = makeDataSet(
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val, batch_size=8
    )

    print("Train:", x_train.shape, y_train.shape)
    print("Validation:", x_val_tensor.shape, y_val_tensor.shape)
    print("GT 3명 합계:", make_density_map([(8, 10), (16, 17), (25, 22)]).sum().item())

    # 2. 모델 구성
    model = CustomCNN().to(device=device)

    # 3. 컴파일 설정 (Keras model.compile에 대응하는 부분)
    loss_fn = MSELoss()
    optimizer = Adam(params=model.parameters(), lr=0.002)

    # 3-1. 사용자가 원한 방식: 별도의 fit 함수 없이 아래에서 epoch 반복
    epoch = 30
    train_loss_list = []
    val_loss_list = []
    val_mae_list = []

    for i in range(epoch):
        train_loss = train(device=device, model=model, train_DataLoader=train_DataLoader,loss_fn=loss_fn, optimizer=optimizer)
        pred, eval_loss, val_mae = evaluate(device=device, model=model, x_val_tensor=x_val_tensor,y_val_tensor=y_val_tensor, loss_fn=loss_fn)

        train_loss_list.append(train_loss)
        val_loss_list.append(eval_loss)
        val_mae_list.append(val_mae)

        print(f"Epoch [{i+1}/{epoch}] train_loss={train_loss:.5f}, "
              f"val_loss={eval_loss:.5f}, val_count_MAE={val_mae:.3f}명")

    # 4-1. 마지막 평가 예시 (첫 번째 검증 이미지)
    pred_count = pred.sum(dim=(1, 2, 3)).cpu().numpy()
    true_count = y_val_tensor.sum(dim=(1, 2, 3)).cpu().numpy()
    print(f"첫 검증 이미지 정답={true_count[0]:.1f}명, 예측={pred_count[0]:.2f}명")
    print("완료: 외부 epoch 반복 구조 유지, 합성 데이터 실습 (실제 STNNet 재현 아님)")
