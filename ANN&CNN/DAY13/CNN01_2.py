# models = {
#     "A":(0.98, 0.86),
#     "B":(0.92, 0.89),
#     "C":(0.60, 0.59)
# }

models = [
    {"A":[0.98, 0.86]},
    {"B":[0.92, 0.89]},
    {"C":[0.60, 0.59]}
]

result = {}
for name, values in enumerate(models):
    gap = values.train_acc - values.val_acc
    result[name] = {"val":values.val_acc, "gap":values.gap}
    print(name, "val=", values.val_acc, "gap=", gap)

# 검증 정확도와 일반화 간격 계산
# result = {}
# for name, (train_acc, val_acc) in models.items():
#     gap = train_acc - val_acc
#     result[name] = {"val":val_acc, "gap":gap}
#     print(name, "val=",val_acc," gap=",gap)

#서로 다른 기준으로 모델 찾기
best_val = max(result, key=lambda k: result[k]["val"])
largest_gap = max(result, key=lambda k: result[k]["gap"])
assert best_val == "B"
assert largest_gap == "A"
#assert result["C"]["val"] < 0.60
print("best:", best_val, "/ largest gap:", largest_gap)

