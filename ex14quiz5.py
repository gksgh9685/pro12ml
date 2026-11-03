# 회귀분석 문제 5) 
# Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
# testdata 폴더 : Consumo_cerveja.csv

# Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
# feature : Temperatura Media (C) : 평균 기온(C)
#             Precipitacao (mm) : 강수(mm)
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오

# 조건 : NaN이 있는 경우 삭제!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 데이터 불러오기
pd.set_option('display.max_columns', None)
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Consumo_cerveja.csv')

print("데이터 상위 5개 값 :\n", data.head(5))
data.info()


# 결측치 삭제
print("원본 데이터의 NaN 개수 :\n", data.isna().sum())

data = data.dropna()
print("NaN 삭제 확인 : \n", data.isna().sum())


# 데이터 선택 및 타입 변경
df = pd.DataFrame()
df['Average Temperature (C)'] = (data['Temperatura Media (C)'].str.replace(',', '.').astype(float))
df['Precipitation (mm)'] = (data['Precipitacao (mm)'].str.replace(',','.').astype(float))
df['Beer Consumption (liters)'] = (data['Consumo de cerveja (litros)'])
df.info()


# 데이터 세트 나누기
train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
print(train_set.shape, test_set.shape)
print("train set 상위 3개 값 :\n", train_set.head(3))
print("test set 상위 3개 값 :\n", test_set.head(3))


# 모델 생성 및 학습
model_linear = LinearRegression().fit(
    X=train_set[['Average Temperature (C)', 'Precipitation (mm)']],
    y=train_set['Beer Consumption (liters)'])


# 모델 파라미터
print(f"slope : {model_linear.coef_}")
print(f"bias : {model_linear.intercept_}")


# 모델 평가
pred = model_linear.predict(X=test_set[['Average Temperature (C)', 'Precipitation (mm)']])
print(f"예측한 맥주 소비량 : {np.round(pred[:5],3)}")
print(f"실제 맥주 소비량 : {test_set['Beer Consumption (liters)'][:5].values.flatten()}")

mse_val = mean_squared_error(test_set['Beer Consumption (liters)'], pred)
r2score_val = r2_score(test_set['Beer Consumption (liters)'], pred)

print(f"MSE(평균제곱오차) : {mse_val:.3f}") 
print(f"RMSE(평균제곱근오차): {np.sqrt(mse_val):.3f}") # 3.649
print(f"r2 score(결정계수) : {r2score_val:.3f}")    # 0.39, 보통


# 시각화
actual_val = test_set['Beer Consumption (liters)']

plt.scatter(x=actual_val, y=pred, alpha=0.7)

min_value = min(actual_val.min(), pred.min())
max_value = max(actual_val.max(), pred.max())
plt.plot([min_value,max_value], [min_value,max_value], 'r--', label='예측값=실제값')

plt.title('실제값과 예측값 비교')
plt.xlabel('실제 맥주 소비량')
plt.ylabel('예측 맥주 소비량')
plt.grid(alpha=0.3)
plt.legend()

plt.show()
