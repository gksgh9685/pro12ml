# 선형회귀분석 : mtcar dataset
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api
import koreanize_matplotlib

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars)
print(mtcars.columns)
print(mtcars.info())
# x:hp(마력수), y:mpg(연비)
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0,1])

# 시각화
# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('마력수(hp)')
# plt.ylabel('연비(mpg)')
# plt.show()

print('\n단순선형회귀 ---')
result = smf.ols(formula = 'mpg ~ hp', data=mtcars).fit()
print(result.summary())
print(f'회귀식 : mpg = {result.params['hp']:.4f} * hp + {result.params['Intercept']:.4f}')
print('마력수 110에 대한 연비 예측값 : ',
    result.predict(pd.DataFrame({'hp': [110]})).values[0])

y_pred = result.predict(mtcars)
idx = np.argsort(mtcars.hp.values)  # hp 기준으로 오름차순 정렬
plt.figure(figsize=(10,6))
plt.scatter(mtcars.hp, mtcars.mpg, label='실제값', color='blue')
plt.plot(mtcars.hp.values[idx], y_pred.values[idx], color='red', label='회귀선')
plt.xlabel('마력수(hp)')
plt.ylabel('연비(mpg)')
plt.legend()
plt.show()

print('결정계수(R2) : ', result.rsquared)

print('새로운 마력수에 대한 연비 추정 ---')
new_data = pd.DataFrame({'hp':[110, 150, 200, 250]})
new_pred = result.predict(new_data)
print(pd.DataFrame({'마력수(hp)': new_data['hp'], '예측 연비(mpg)': new_pred.round(2)}))

print('\n다중선형회귀 (x:hp, wt, y:mpg)---')
result2 = smf.ols(formula='mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary())
print('회귀계수 : \n', result2.params)

# yhat = w1 * hp + w2 * wt + b
print(f'다중 회귀식 : mpg = {result2.params['hp']:.4f} * hp'
    f' + {result2.params['wt']:.4f} * wt'
    f' + {result2.params['Intercept']:.4f}')

y_pred2 =result2.predict(mtcars)
print('실제값 : ', mtcars.mpg[:5].values.round(2))
print('예측값 : ', y_pred2[:5].values.round(2))

print('새로운 마력수, 차체 무게에 대한 연비 추정 ---')
new_data2 = pd.DataFrame({'hp':[110, 150, 200, 250], 'wt':[2.5, 3.0, 3.5, 4.0]})
new_pred2 = result2.predict(new_data2)
print(pd.DataFrame({'마력수(hp)': new_data2['hp'], 
                '차체무게(wt)': new_data2['wt'],
                '예측 연비(mpg)': new_pred2.round(2)}))

print('수정 결정계수(Adj.R2) : ', result2.rsquared_adj) # 수정 결정계수(Adj.R2) :  0.8148396209781564

# 시각화
plt.figure(figsize=(9,5))
plt.scatter(mtcars.mpg, y_pred2, label='예측값')
# 실제값과 예측값이 일치하는 기준선
min_val = min(mtcars.mpg.min(), y_pred2.min())
max_val = max(mtcars.mpg.max(), y_pred2.max())

plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='실제값=예측값')
plt.xlabel('실제연비(mpg)')
plt.ylabel('예측연비(mpg)')
plt.title('다중선형회귀:실제값과 예측값 비교')
plt.legend()
plt.show()
