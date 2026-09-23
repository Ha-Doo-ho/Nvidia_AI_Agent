""" 
VAE(variational Autoencoder) 

AE와 무엇이 다른가
AE: 이미지 → 고정된 잠재 벡터 z → 복원
VAE: 이미지 → 평균 μ와 분산 σ² → 분포에서 z 샘플링 → 복원 및 새로운 이미지 생성

VAE에서는 다음 다섯 가지를 순서대로 학습하면 된다.

1. 일반 AE의 잠재공간이 가진 문제
2. 평균 μ와 로그 분산의 역할
3. Reparameterization Trick
4. Reconstruction Loss
5. KL Divergence
""" 