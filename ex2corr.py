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

# plt.hist([np,std(data.친밀도), np.std(data.적절성),np.std(data.만족도)])
# plt.show()
std_values = [np.std(data.친밀도), np.std(data.적절성),np.std(data.만족도)]
plt.scatter(['친밀도', '적절성', '만족도'], std_values)
plt.title('변수별 표준편차')
plt.show()

print('공분산 ---')
print(np.cov(data.친밀도, data.적절성)) # numpy함수
print(np.cov(data.친밀도, data.만족도))
print()
print(data.cov())   # pandas dataframe 함수










