from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# iq에 따른 시험 점수 예측
score_iq = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv')
print(score_iq.head(3))
print(score_iq.info())
x = score_iq.iq
y = score_iq.score
print(x[:3])
print(y[:3])

print('상관 계수 : ', np.corrcoef(x,y)[0,1])
print(score_iq[['iq','score']].corr())

# plt.scratter(x,y)
# plt.show()

# 단순선형회귀분석 모델 생성 - 인과관계가 있다는 가정 하에 진행
model = stats.linregress(x,y)
print(model)
# LinregressResult(slope=(0.6514309527270083), intercept=(-2.8564471221975793), rvalue=(0.8822203446134704), pvalue=(2.847689520667442e-50), stderr=(0.028577934409305394), intercept_stderr=(3.5462119180485323))
# 단순선형회분석 만 가능 (다중회귀 x), 예측함수(predict)x
print('가중치 : ', model.slope) # 가중치 :  0.6514309527270083
print('편향 : ', model.intercept) # 편향 :  -2.8564471221975793
print('p값 : ', model.pvalue) # p값 :  2.847689520ㄷ-50 <0.05이므로 두 변수는 인과관계가 있다.

plt.scatter(x,y)
plt.plot(x, model.slope * x + model.intercept, c='r')
plt.show()
# predict() 지원 안함

newdf = pd.DataFrame({'iq':[55, 66, 77, 88, 99]})
print('점수 예측 : \n',np.polyval([model.slope, model.intercept], newdf).ravel().round())

