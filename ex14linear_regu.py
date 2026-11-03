# 과적합(overfitting)
# 선형 회귀 모델이 훈련 데이터(training data)에 너무 과하게 학습되어,
# 데이터에 포함된 노이즈(잡음)나 지엽적인 특성까지 암기해 버리는 현상
# 훈련 데이터에 대한 예측 오차는 매우 낮지만, 처음 보는 테스트 데이터(test data)나 
# 새로운 데이터가 들어왔을 때 예측 성능(일반화 능력)이 크게 떨어진다.

# linear regression에서 과적합을 해결하는 방법
# 특성(feature) 줄이기, 정규화(regularization)적용, 더 많은 데이터 수집하기...

# 이 중에서 정규화(regularization) 적용하기 진행 ...
# 회귀계수 값이 너무 커지지 않도록 비용함수(cost function)에 페널티를 부과(규제)
# ridge (L2규제): 가중치들의 제곱합을 최소화하여 계수들을 0에 가깝게 작게 만든다.
# LASSO (L1규제): 가중치들의 절대값 합을 최소화하며, 중요하지 않은 특성의 계수를 아예 0으로 만들어 제거한다.
# ELASTICNET (RIDGE + LASSO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

iris = load_iris()
# print(iris)
print(iris.keys())  # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

pd.set_option('display.max_columns', None)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_names'] = iris.target_names[iris.target]
print(iris_df.head(3))
#     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target target_names  
# 0          5.1               3.5                1.4               0.2           0       setosa  
# 1          4.9               3.0                1.4               0.2           0       setosa 
print(iris_df.shape)    # (150, 6)

# train(학습) / test(검증) 데이터로 분리 - 7 : 3
train_set, test_set = train_test_split(iris_df, test_size=0.3, random_state=12, shuffle=True)
print(train_set.shape, test_set.shape) # (105, 6) (45, 6)
# print(train_set[:3])
# print(test_set[:3])

print(train_set.iloc[:,[2]])    #  petal length (cm), 독립변수
print(train_set.iloc[:,[3]])    #  petal width (cm), 종속변수

# 회귀분석 방법1 - LinearRegression
# 학습은 train dataset 사용
model_linear = LinearRegression().fit(X=train_set.iloc[:,[2]], y=train_set.iloc[:,[3]])
print('slope :', model_linear.coef_)        # slope : [[0.42259168]]
print('bias : ', model_linear.intercept_)   # bias :  [-0.39917733]

# 모델 평가는 test dataset으로 작업
pred = model_linear.predict(test_set.iloc[:, [2]])
print('예측값 : ', np.round(pred[:5].flatten(), 1)) #  예측값 :  [0.2 1.7 0.2 1.3 2.1]
print('실제값 : ', test_set.iloc[:,[3]][:5].values.flatten())   # 실제값 :  [0.3 1.9 0.2 1.3 2.3]

print(f'r2_score(결정계수):{r2_score(test_set.iloc[:,[3]], pred)}',)    # 0.93833219

# 시각화
plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:,[3]], color='red')
plt.plot(np.array(test_set.iloc[:,[2]]), model_linear.predict(test_set.iloc[:,[2]]))
plt.show()

print('\nRidge -----------')
# 회귀분석 방법 - Ridge: alpha값을 조정(가중치 제곱합을 최소화)하여 과대/과소적합을 피한다. 다중공선성 문제 처리에 효과적.
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

#점수
print(model_ridge.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]]))  # 0.91880
print(model_ridge.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))    # 0.94101
pred_ridge = model_ridge.predict(test_set.iloc[:, [2]])
print('ridge predict : ', pred_ridge[:5])
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred_ridge)))  # 0.9410

plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='blue')
plt.plot(np.array(test_set.iloc[:, [2]]), model_ridge.predict(test_set.iloc[:, [2]]))
plt.show()

print('\nLasso -----------')
# 회귀분석 방법 - Lasso: alpha값을 조정(가중치 절대값의 합을 최소화)하여 과대/과소적합을 피한다.
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

#점수
print(model_lasso.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])) # 0.913863
print(model_lasso.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))   # 0.940663
pred_lasso = model_lasso.predict(test_set.iloc[:, [2]])
print('lasso predict : ', pred_lasso[:5])
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred_lasso)))

plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='green')
plt.plot(np.array(test_set.iloc[:, [2]]), model_lasso.predict(test_set.iloc[:, [2]]))
plt.show()

# 회귀분석 방법 4 - Elastic Net 회귀모형 : Ridge + Lasso의 형태로 가중치 절대값의 합(L1)과 제곱합(L2)을 동시에 제약 조건으로 가지는 모형
print('\nElasticNet -----------')
# 회귀분석 방법 - ElasticNet: alpha값을 조정(가중치 절대값의 합을 최소화)하여 과대/과소적합을 피한다.
from sklearn.linear_model import ElasticNet
model_elastic = ElasticNet(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

#점수
print(model_elastic.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])) # 0.913863
print(model_elastic.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))   # 0.940663
pred_elastic = model_elastic.predict(test_set.iloc[:, [2]])
print('ElasticNet predict : ', pred_elastic[:5])
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred_elastic)))

plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='cyan')
plt.plot(np.array(test_set.iloc[:, [2]]), model_elastic.predict(test_set.iloc[:, [2]]))
plt.show()