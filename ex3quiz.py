# 상관관계 문제)
# https://github.com/pykwon/python 에 있는 Advertising.csv 파일을 읽어 tv,radio,newspaper 간의 상관관계를 파악하시오. 
# 또한 sales와 관계를 알기 위해 sales에 상관 관계를 정렬한 후 TV, radio, newspaper에 대한 영향을 해석하시오.
# 그리고 이들의 관계를 heatmap 그래프로 표현하시오. 

# 광고비(tv, radio, newspaper)와 판매량(sales)의 상관관계
# 광고 매체별(tv, radio, newspaper) sales와의 상관관계 확인

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv")
print(data.head(3))
print(data.describe())

attr = ['tv', 'radio', 'newspaper', 'sales']
data = data[attr]

print('\n광고 매체 간 상관관계')
print(data[['tv', 'radio', 'newspaper']].corr().round(3))

print('\nsales와 다른 특성 사이의 상관관계 (내림차순)')
co_re = data.corr()
# 자기 자신과의 상관계수 1.0은 제외
sales_corr = co_re['sales'].drop('sales').sort_values(ascending=False)
print(sales_corr.round(3))

print('상관관계 분석 결과 ---')
for name, r in sales_corr.items():
    if r > 0:
        msg = "양의 상관관계"
    elif r < 0:
        msg = "음의 상관관계"
    else:
        msg = "선형 상관관계 없음"

    print(f'{name} : r={r:.3f}, {msg}')

print('\nSales와 양의 상관관계가 가장 높은 광고 매체 : ', sales_corr.idxmax())

sns.heatmap(co_re, annot=True)
plt.title('광고비와 판매량의 상관관계')
plt.show()
plt.close()