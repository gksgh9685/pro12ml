# 공분산 / 상관계수
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv')
print(data.head(3))
print(data.describe())

print('평균 / 표준편차')
print(np.mean(data.친밀도), ' ', np.std(data.친밀도))
print(np.mean(data.적절성), ' ', np.std(data.적절성))
print(np.mean(data.만족도), ' ', np.std(data.만족도))

# std_values = [np.std(data.친밀도), np.std(data.적절성),np.std(data.만족도)]
# plt.scatter(['친밀도', '적절성', '만족도'], std_values)
# plt.show()

print('공분산 ---')
print(np.cov(data.친밀도, data.적절성)) # numpy함수
print(np.cov(data.친밀도, data.만족도))
print()
print(data.cov())   # pandas dataframe 함수

print('상관계수 ---')
print(np.corrcoef(data.친밀도, data.적절성)) # numpy함수
print(np.corrcoef(data.친밀도, data.만족도))
print()
print(data.corr())   # pandas dataframe 함수
print(data.corr(method='pearson'))  # 변수 연속형, 정규성을 다름
# print(data.corr(method='spearman')) # 변수 서열형, 정규성을 따르지 않음
# print(data.corr(method='kendall'))

print('\n만족도에 따른 다른 특성 사이의 관계')
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))

# 시각화
data.plot(kind='scatter', x='만족도', y='적절성')
attr = ['친밀도','적절성', '만족도']

from pandas.plotting import scatter_matrix
attr = ['친밀도','적절성', '만족도']
scatter_matrix(data[attr], figsize=(10,6))
plt.show()
plt.close()


