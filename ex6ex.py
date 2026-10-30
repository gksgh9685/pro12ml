# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
# 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
# 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
# 참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  
# 구분,지상파,종편,운동
# 1,0.9,0.7,4.2
# 2,1.2,1.0,3.8
# 3,1.2,1.3,3.5
# 4,1.9,2.0,4.0
# 5,3.3,3.9,2.5
# 6,4.1,3.9,2.0
# 7,5.8,4.1,1.3
# 8,2.8,2.1,2.4
# 9,3.8,3.1,1.3
# 10,4.8,3.1,35.0
# 11,NaN,3.5,4.0
# 12,0.9,0.7,4.2
# 13,3.0,2.0,1.8
# 14,2.2,1.5,3.5
# 15,2.0,2.0,3.5

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


# linregress
# 1. 데이터 준비
data = pd.DataFrame({
    '구분': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    '지상파': [0.9,1.2,1.2,1.9,3.3,4.1,5.8,2.8,3.8,4.8,np.nan,0.9,3.0,2.2,2.0],
    '종편': [0.7,1.0,1.3,2.0,3.9,3.9,4.1,2.1,3.1,3.1,3.5,0.7,2.0,1.5,2.0],
    '운동': [4.2,3.8,3.5,4.0,2.5,2.0,1.3,2.4,1.3,35.0,4.0,4.2,1.8,3.5,3.5]
})

print(data)
print(data.info())
print(data.isnull().sum())

data = data[data['운동'] <=10].copy()
data['지상파'] = data['지상파'].fillna(data['지상파'].mean())
print('\n처리 후 데이터')
print(data)

# 1) 지상파 시청 시강에 따른 운동 시간 예측
print('------------------------------------------------')
print('\n 지상파와 운동 시간 회귀분석')
x = data['지상파']
y = data['운동']
print('상관계수 : ', np.corrcoef(x, y)[0, 1])
model1 = stats.linregress(x, y)
print(model1)

print('가중치 : ', model1.slope)
print('편향 : ', model1.intercept)
print('p값 : ', model1.pvalue)
print('결정계수 : ', model1.rvalue ** 2)

tv = float(input('지상파 시청 시간 입력 :'))
predict1 = model1.slope * tv + model1.intercept
print('예상 운동 시간 : {:.2f}시간'.format(predict1))

plt.scatter(x, y)
plt.plot(x, model1.slope * x + model1.intercept, c='r')
plt.xlabel('지상파 시청 시간')
plt.ylabel('운동 시간')
plt.title('지상파와 운동 시간')
plt.show()

print('--------------------------------------------------')

# 2) 지상파 시청 시간에 따른 종편 시청 시간 예측

print('\n 지상파와 종편 시청 시간 회귀분석')
x = data['지상파']
y = data['종편']

print('상관계수 : ', np.corrcoef(x, y)[0, 1])

model2 = stats.linregress(x, y)
print(model2)

print('가중치 : ', model2.slope)
print('편향 : ', model2.intercept)
print('p값 : ', model2.pvalue)
print('결정계수 : ', model2.rvalue ** 2)

tv = float(input('지상파 시청 시간 입력 : '))
predict2 = model2.slope * tv + model2.intercept
print('예상 종편 시청 시간 : {:.2f}시간'.format(predict2))

plt.scatter(x, y)
plt.plot(x, model2.slope * x + model2.intercept, c='r')
plt.xlabel('지상파 시청 시간')
plt.ylabel('종편 시청 시간')
plt.title('지상파와 종편 시청 시간')
plt.show()

print('--------------------------------------------------')

# skLearn LinearRegression()
# 1) 지상파 시청 시강에 따른 운동 시간 예측
print('\n LinearRegression 사용')

x = data[['지상파']]
y = data['운동']
model3 = LinearRegression()
fit_model = model3.fit(x, y)
print('기울기 : ', fit_model.coef_)
print('절편 : ', fit_model.intercept_)

new_x = pd.DataFrame({'지상파': [1.0, 2.0, 3.0]})
y_new_predicts = fit_model.predict(new_x)
for hour, predict in zip(new_x['지상파'], y_new_predicts):
    print(f'지상파:{hour:.2f}시간 -> 예상 운동 시간:{predict:.2f}시간')

# 2) 지상파 시청 시간에 따른 종편 시청 시간 예측

y = data['종편']

model4 = LinearRegression()
fit_model2 = model4.fit(x, y)

print('기울기 : ', fit_model2.coef_)
print('절편 : ', fit_model2.intercept_)

y_new_predicts = fit_model2.predict(new_x)

for hour, pred in zip(new_x['지상파'], y_new_predicts):
    print(f'지상파:{hour:.2f}시간 -> 예상 종편 시간:{pred:.2f}시간')

print('---------------------------------------------------')

# statsmodels ols()
# 1) 지상파 시청 시강에 따른 운동 시간 예측

print('\n ols 사용')

model5 = smf.ols(formula='운동 ~ 지상파', data=data).fit()
print(model5.summary())

print('가중치 : ', model5.params['지상파'])
print('편향 : ', model5.params['Intercept'])

new_df = pd.DataFrame({'지상파': [1.0, 2.0, 3.0]})
print('예상 운동 시간 : ', model5.predict(new_df))

# 2) 지상파 시청 시간에 따른 종편 시청 시간 예측

model6 = smf.ols(formula='종편 ~ 지상파', data=data).fit()
print(model6.summary())

print('가중치 : ', model6.params['지상파'])
print('편향 : ', model6.params['Intercept'])
print('예상 종편 시간 : ', model6.predict(new_df))
