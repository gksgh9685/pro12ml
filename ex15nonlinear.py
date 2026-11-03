# 비선형회귀분석(non-linear regression)
# 직선의 회귀선을 곡선으로 변환해 더 정확하게 데이터 변화를 예측 가능
# 선형 가정이 어긋날 때(비 정규성) 대처할 수 있는 방법으로 다항식 항을 추가해 다항회귀모델을 작성


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

print(np.corrcoef(x, y)[0,1])

plt.scatter(x, y)
plt.show()

# 선형회귀 모델을 적용
from sklearn.linear_model import LinearRegression
x= x[:, np.newaxis]     # 차원 확대. 1차원 -> 2차원
print(x) # [[1] [2] [3] [4] [5]]

model = LinearRegression().fit(x,y)
ypred = model.predict(x)
print('예측값 y : ', ypred)
print('결정계수 : ', r2_score(y, ypred))

plt.scatter(x, y)
plt.plot(x, ypred, c='red')
plt.show()

# 비선형 모델 작성
# 여러 방법 중 가장 일반적인 방법을 사용(polynomialfeatures, log변환, curve_fit ..)
from sklearn.preprocessing import PolynomialFeatures    # 다항식 특징(변수)을 추가 가능
poly = PolynomialFeatures(degree=2, include_bias=False) # degree=열의 갯수
x2 = poly.fit_transform(x)
print(x)    # 5 by 1
print(x2)   # 5 by 2[[ 1.  1.] [ 2.  4.] [ 3.  9.] [ 4. 16.] [ 5. 25.]]

model2 = LinearRegression().fit(x2, y)  # 추가된 특징 행렬값으로 모델 생성
ypred2 = model2.predict(x2)
print('예측값 y :', ypred2) # 예측값 y : [4.14285714 1.62857143 1.25714286 3.02857143 6.94285714]
print('결정계수 :', r2_score(y, ypred2)) # 결정계수 : 0.9892183288409703

plt.scatter(x,y)
plt.plot(x, ypred2, c='red')
plt.show()



