import numpy as np
w = np.array([0.2, 1.0, 3.0])
l2_rate = 1e-3
# ① 가중치별 제곱과 벌점 기여도
squared = w ** 2
contribution = l2_rate * squared
penalty = contribution.sum()
share = contribution / penalty
for value, sq, part in zip(w, squared, contribution):
    print(f"w={value:.1f} w²={sq:.2f} penalty={part:.4f}")
# ② 가장 큰 가중치의 기여 비율
largest_share = share.argmax()
print("total:", round(penalty, 5))
print("largest index:", largest_share)
print("largest share:", round(share[largest_share], 3))
assert np.isclose(penalty, 0.01004)
assert largest_share == 2
assert share[2] > 0.89
print("큰 가중치가 벌점의 대부분을 차지")

# L2는 큰 가중치를 무조건 0으로 만들기보다 과도하게 커지는 것을 눌러 줍니다
# 보통 1e-4 혹은 1e-3을 시작점으로 준다. 테스트는 10배씩 건너뛰면서 진행한다. 
# 탐색 순서 예시: 1e-5 (0.00001) --> 1e-4 (0.0001) --> 1e-3 (0.001) --> 1e-2 (0.01)

"""
강도를 높여야 할 때 (예: 1e-4 --> 1e-3)
증상: 여전히 loss는 뚝뚝 떨어지며 학습을 잘하는데, val_loss는 떨어지다가 다시 치솟습니다. (과적합이 계속됨)
처방: "벌점이 너무 약하네. 특정 특징(예: 배경색)에 집착하는 가중치를 더 강하게 억눌러야겠다!" 하고 숫자를 키웁니다.

강도를 낮춰야 할 때 (예: 1e-4 --> 1e-5 또는 아예 삭제)
증상: 훈련 초기부터 loss와 val_loss 둘 다 제대로 안 떨어지고, 정확도(Accuracy)가 형편없이 나옵니다.
처방: "벌점이 너무 무서워서 뇌세포들이 활동을 못 하고 있네. 규제를 풀어줘야겠다!" 하고 숫자를 줄이거나 L2 층을 지워버립니다.
"""
