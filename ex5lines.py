# 전통적 방법의 선형회귀(긱학습 중 지도학습)
# 각 데이터에 대한 잔차제곱합이 최소가 되는 추세선(회귀선)을 만들고,
# 이를 통해 독립변수가 종석변수에 얼마나 영향을 주는지 인과관계를 분석.
# 독립변수 : 연속형, 종속변수:연속형 - 두 변수는 상관관계 및 인과관계가 있어야함.
# 정량적인 모델을 생성

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression 
import numpy as np

np.random.seed(1)

# 공부 시간과 시험 점수 데이터
# 독립변수(x, feature) : 공부시간(1~10)
x = np.linspace(1, 10, 50).reshape(-1,1) # 2차원
print(x)
# 종속변수(y, label, class) : 시험 점수
y = 5 * x.ravel() + 40 + np.random.normal(0,3,50)
print(y)

r=np.corrcoef(x.ravel(), y)[0, 1]
print(f'상관계수 : {r:.4f}')

print('\n모델 작성 방법 1 : LinearRegression 사용 - 모델 생성')

model = LinearRegression()
fit_model = model.fit(x,y) # 내부에서 아래와 같은 작업을 진행
# 1) 독립변수(x)와 종속변수(y)를 입력 받음
# 2) 예측값과 실제값 사이의 오차제곱합이 최소가 되도록 계산
# 3) 최적의 회귀계수(기울기, 절편)를 구함
# 4) 계산된 계수를 모델에 저장
# 단순선형회귀 모델에서는 y=wx + b와 같은 단순한 형태의 수식을 가진 모델이 만들어진다.
# 참고 : linearRegression은 신경망처럼 반복학습을 통해 가중치를 갱신하는 것이 아니라 계산을 통해 작성됨.
print('기울기(가중치, weight, slope):',fit_model.coef_)
print('절편(편향, bias):',fit_model.intercept_)
# yhat = 79.52463976 * x + 60.92222  일차방정식 완성

# 예측값 확인
y_newpred = fit_model.predict(x[[0]])   # 기존 데이터 0번째를 입력값
print('예측값1 : ', y_newpred)

# 여러 개의 예측값 확인
new_x = np.array([[2.34], [6.21], [8.123]])
y_new_preds = fit_model.predict(new_x)
for hour, score in zip(new_x.ravel(), y_new_preds):
    print(f'공부시간: {hour:.2f}시간 -> 예상 점수:{score:.2f}점')

print('\n모델 작성 방법 2 : ols(ordinary least squares) 사용 - 모델 생성')
# 잔차제곱합(rss)을 최소화하는 가중치 벡터를 행렬 미분으로 구하는 방법
# 실제 계상는 수치 선형대수 알고리즘을 사용
import statsmodels.formula.api as smf
import pandas as pd

data = np.array([x.ravel(), y])
df = pd.DataFrame(data.T)
df.columns = ['x', 'y']
print(df.head(3))

model2= smf.ols(formula='y~x',data=df).fit()
print(model2.summary()) # OLS Regression Results 제공
print('가중치:',model2.params['x'])
print('평향:',model2.params['Intercept'])

# 예측값 확인
new_df = pd.DataFrame({'x':[2.34, 6.21, 8.123]})
print('예측 점수 :', model2.predict(new_df))

# LinearRegression  vs      ols
# 예측 중심             통계적 추론, 회귀분석 중심
# 최소제곱법                최소제곱법
# p-value 제공 x            p-value 제공 o
# 회귀분석 결과표 x         회귀분석 결과표 o




