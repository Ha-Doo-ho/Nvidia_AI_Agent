# 에폭 별 훈련 검증 정확도
train_acc = [0.64, 0.74, 0.82, 0.89, 0.94]
val_acc = [0.63, 0.72, 0.77, 0.76, 0.74]

gaps = []

for epoch, (train, val) in enumerate(
    zip(train_acc, val_acc), start=1   
):
    gap = train - val
    gaps.append(gap)
    
    print(
        f"epoch{epoch}"
        f"train={train:.2f}, "
        f"val={val:.2f}, "
        f"gap={gap:.2f}"
    )

# 처음과 마지막 epoch의 차이 비교
print("처음 차이:", round(gaps[0], 2))
print("마지막 차이:", round(gaps[-1], 2))
print("차이가 커졌는가?", gaps[-1] > gaps[0])