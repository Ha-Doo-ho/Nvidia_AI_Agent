import numpy as np
import matplotlib.pyplot as plt

aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 50])

def outlier(data): 
    quantile_1, quantile_2, quantile_3 = np.percentile(data, [25, 50, 75]) #25% 지점, 50%지점, 75%지점. 즉 1분위, 2분위, 3분위 값을 받겠다는 의미이다. 
    print("1사분위 : ", quantile_1)  # 4.0   1사분위 값
    print("q2 : ", quantile_2)               # 7.0   중위값(2 사분위 값)
    print("3사분위 : ", quantile_3)   # 10.0  3사분위 값
    iqr = quantile_3 - quantile_1   #네모박스 크기 
    print('IQR : ', iqr) 
    lower_bound = quantile_1 - (iqr * 1.5)  # 봐줄 수 있는 범위를 만드는 것이다. 너무 작을 수도 있지만, 이것도 정상으로 보겠다는 것이다. 
    upper_bound = quantile_3 + (iqr * 1.5)  # 봐줄 수 있는 범위를 만드는 것이다. 너무 클 수도 있지만, 이것도 정상으로 보겠다는 것이다. 
    return np.where((data > upper_bound) | (data < lower_bound)), \
        iqr, lower_bound, upper_bound # \ 넣으면 줄이 바뀌어도 다음 줄까지 이어져 있다는 것을 의미한다.
outlier_loc, iqr, low, up = outlier(aaa)    # 줄이 너무 길어질 때, 줄바꿈으로 깔끔하게 해야 하는데, 파이썬에서는 줄 잘못바꾸면 에러 나니까 그거 막으려고 사용함.
print('이상치의 위치 : ', outlier_loc)       #np.where 은 위치를 반환해준다. 그래서 반환값은 4개가 맞다. 

plt.boxplot(aaa)
plt.axhline(up, color='red', label='upper bound')
plt.axhline(low, color='blue', label='lower bound')
plt.legend()
plt.show()

# 실제로 우리가 전체 데이터를 보는 경우가 없다. 그래서 이상치를 보기는 어렵다. 
# 우리가 보는 값의 대부분 모든 값은 정상 범주의 + 1.5배, - 1.5 배 안에 있다고 판단하는 것이다. 즉, 범위를 정말 넉넉하게 준 것이다. 
# 그걸 1.5, 1.3, 1.7 내가 마음대로 정할 수 있다. 
# 그것조차 벗어난 수치들은 정말 데이터 전체를 오염시킬 수 있다고 판단하고 이상치라고 판단한다. 