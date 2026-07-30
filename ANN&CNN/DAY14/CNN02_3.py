import numpy as np
# 한 이미지의 클래스별 예측 확률
probs = np.array([0.62, 0.28, 0.10])
true_target = 2 # 실제 정답 클래스
# 확률이 높은 순서로 클래스 번호 정렬
ranked_targets = np.argsort(probs)[::-1]
pred_target = int(ranked_targets[0]) # 예측 1위
alt_target = int(ranked_targets[1]) # 비교할 2위 후보
# 세 가지 target을 한곳에 기록
targets = {
 "pred": pred_target,
 "true": true_target,
 "alt": alt_target,
}
# target 번호와 해당 클래스 score 출력
for name, target in targets.items():
 print(
 f"{name:>4}_target = {target}, "
 )
 
 # 특징 맵, 스코어 
