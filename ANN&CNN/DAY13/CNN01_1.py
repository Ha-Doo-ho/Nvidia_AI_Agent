import numpy as np
idx = np.arange(20) 

train = idx[:14]
validation = idx[14:17]
test = idx[17:]

print("split", train, validation, test)
print(len(train) + len(validation), len(test))
assert len(train) + len(validation) + len(test) == len(idx)
assert set(train).isdisjoint(validation) # isdisjoint 메서드는 두 집합(Set)이 서로소 관계인지, 즉 공통된 요소(교집합)가 없는지 확인하여 공통 원소가 없으면 True, 하나라도 있으면 False를 반환하는 불리언(Boolean) 메서드
assert set(train).isdisjoint(test)       # 구문: set1.isdisjoint(set2) (파이썬 기준)
assert set(validation).isdisjoint(test)
assert np.array_equal(np.r_[train, validation, test], idx)
print("분리 확인 완료")