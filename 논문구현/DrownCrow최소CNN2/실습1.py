"""SDNet Figure 5 / Eq. (2)~(4) 이해를 위한 1단계 실습.

실제 논문 전체 모델이 아닙니다. 두 프레임의 다중 스케일 특징을
Self-Attention -> Cross-Attention -> MLP로 처리하는 흐름만 구현했습니다.

Figure 4의 왼쪽에 있는 Backbone과 DCFA 모듈까지만 구현한 것.

실행: python SDNet_DCFA_1단계_어텐션.py
필요 패키지: torch
"""

import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU, GELU, MultiheadAttention, Linear, Sigmoid, CrossEntropyLoss, LayerNorm
from torch.nn.functional import interpolate
from torch.optim import Adam

class CrossFrameUnit(Module): # unit: 특정 scale의 특징을 처리하는 DCFA 모듈 단위 
    def __init__(self, channels=16, heads=2): # channels는 돋보기의 개수(차원/특징의 개수)이니까, keras의 filter 개수랑 완전히 동일하다. 
        super().__init__()
        self.ln1 = LayerNorm(normalized_shape=channels)  # 나온 결과가 튀지 않게 눌러주어야 한다. 그 벡터가 차원인지 알아야 하므로 normalized_shape 매개변수가 벡터의 차원을 받는다. 
        self.msa = MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True) # 임베딩 몇개의 숫자로 나오는지, 벡터니까 그 길이가 곧 embed_dim이다. batch_first하면 배치, 차원수, 행, 열 순으로 나온다.  False로 하면 차원수, batch, 행, 열 순 임.
        self.ln2 = LayerNorm(normalized_shape=channels)
        self.mca = MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True) # batch_first 는 Default값이 False라서 반드시 True해야 함. ㅈㄴ 귀찮음
        self.ln3 = LayerNorm(normalized_shape=channels)
        self.mlp = Sequential(
                              Linear(in_features=channels, out_features=channels*2),  #왜 여기서 in_features의 개수가 색 채널이 아닌, 바로 특징 채널의 값이 나오는가? 설명하기 위해 일부러 이렇게 하였나?
                              GELU(),
                              Linear(in_features=channels * 2, out_features=channels)
                            )
        
    def forward(self, current, other):
        # current, other : (B, C, H, W)
        batch, channels, height, width = current.shape
        x = current.flatten(2).transpose(1, 2)
        y = other.flatten(2).transpose(1, 2)
            
        # 논문 Eq (3)-1: 같은 프레임 내부 Self-Attention + 잔차
        z = self.ln1(x)
        self_out, _ = self.msa(z,z,z, need_weights=False)
        x = x + self_out
            
        # 논문 Eq. (3)-2: 현재 프레임 Q, 상대 프레임 K/V
        # Q/K/V 선형 변환은 MultiheadAttention내부에서 수행
        cross_out, weights = self.mca(self.ln2(x), y, y, need_weights=True)
        x = x + cross_out
            
        # 논문 Eq. (3)-3: MLP + 잔차
        x = x + self.mlp(self.ln3(x))
            
        return x.transpose(1,2).reshape(batch, channels, height, width), weights

class CustomCNN(Module):
    def __init__(self):
        super().__init__()
        self.backbone = Sequential(
            Sequential(
                Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same"),
                ReLU(),
                MaxPool2d(kernel_size=2)
            ),
            Sequential(
                Conv2d(in_channels=16, out_channels=24,kernel_size=3, padding="same" ),
                ReLU(),
                MaxPool2d(kernel_size=2)
            )
        )
        
        # 간단한 FPN: 상위 스케일 특징을 Upsmaple해서 더함
        self.lateral1 = Conv2d(in_channels=16, out_channels=16, kernel_size=1)
        self.lateral2 = Conv2d(in_channels=24, out_channels=16, kernel_size=1)
        
        self.unit1 = CrossFrameUnit(channels=16, heads=2)
        self.unit2 = CrossFrameUnit(channels=16, heads=2)
        
        self.fusion = Sequential(
            Conv2d(32, 16, 3, padding="same"),
            ReLU()
        )
        
    def get_features(self, image):
        c1 = self.backbone[0](image)            # (B, 16, 8, 8)
        c2 = self.backbone[1](c1)               # (B, 24, 4, 4)
        f2 = self.lateral2(c2)                  # (B, 16, 4, 4)
        f1 = self.lateral1(c1) + interpolate(
            f2, size=c1.shape[-2:], mode="bilinear", align_corners=False
        )                                      # (B, 16, 8, 8)
        return [f1, f2]
        
    def shared_one_direction(self, current, other):
        # Unit 1: 첫 스케일에서 Self/Cross-Attention
        result1, attn1 = self.unit1(current[0], other[0])
        # Eq. (2): '이전 UNIT 출력'과 '같은 Frame의 다음 SCALE 특징' 결합
        resized = interpolate(result1, size=current[1].shape[-2:],
                              mode="bilinear", align_corners=False)
        mixed = self.fusion(torch.cat([resized, current[1]], dim=1))
        # Unit 2: 다른 스케일에서 다시 Attention 수행
        result2, attn2 = self.unit2(mixed, other[1])
        return result1, result2, attn1, attn2
    
    def forward(self, frames):
        # 입력: (B, 2, 3, 32, 32)
        features_t = self.get_features(frames[:, 0])
        features_next = self.get_features(frames[:, 1])

        # 이전 Frame 기준: t 특징이 Query, t+δ 특징이 Key/Value
        t1, shared_t, attention_t1, attention_t2 = self.shared_one_direction(
            features_t, features_next
        )
        # 현재 Frame 기준: Query와 Key/Value의 출처를 교체
        n1, shared_next, _, _ = self.shared_one_direction(features_next, features_t)

        return {
            "features_t": features_t,
            "unit1_output_t": t1,
            "shared_t": shared_t,
            "shared_next": shared_next,
            "attention_unit1": attention_t1,
            "attention_unit2": attention_t2,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CustomCNN()
    frames = torch.randn(2, 2, 3, 32, 32)    # 배치 2개, 연속 2프레임
    results = model(frames)

    print("입력:", tuple(frames.shape))
    for i, f in enumerate(results["features_t"], 1):
        print(f"F{i}_t:", tuple(f.shape))
    print("Unit 1 출력:", tuple(results["unit1_output_t"].shape))
    print("Unit 1 Attention 가중치:", tuple(results["attention_unit1"].shape))
    print("Unit 2 Attention 가중치:", tuple(results["attention_unit2"].shape))
    print("Shared_t:", tuple(results["shared_t"].shape))
    print("Shared_next:", tuple(results["shared_next"].shape))
    # [B,64,64] = Frame t의 64 Query가 Frame t+δ의 64 Key를 비교.
    # [B,16,16] = Scale 2의 16 Query가 상대 프레임의 16 Key를 비교.

            
        