import numpy as np
from sklearn.metrics import confusion_matrix #결국 혼동행렬도 평가지표이므로 metrics에 있는 것이 맞음. 
y_true = [0, 0, 0, 1, 1, 1, 2, 2]
y_pred = [0, 1, 0, 1, 2, 1, 2, 0]
labels = [0, 1, 2]
# ① 행=실제, 열=예측
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)
# ② 대각선 정답과 바깥 오류
correct = int(np.trace(cm)) # 중앙 주 대각선 원소의 합을 구한다. offset옵션이 있는데, offset=1은 위의 주대각선, offset=-1은 아래의 주대각선의 합을 구한다. 
errors = int(cm.sum() - correct)
expected = np.array([
[2, 1, 0],
[0, 2, 1],
[1, 0, 1],
])
print("correct:", correct, "/ errors:", errors)
np.testing.assert_array_equal(cm, expected)
assert correct == 5 and errors == 3
assert cm[0, 1] == 1 and cm[2, 0] == 1