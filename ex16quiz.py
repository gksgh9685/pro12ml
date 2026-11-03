# 다항회귀분석 문제) 
# 데이터 로드 (Servo, UCI) : "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data"
# 제어 시스템(servo system), 특히 서보 제어기의 응답 속도(response time) 를 예측하기 위해 수집된 고전 데이터.
# 당시 뉴럴 네트워크와 회귀 분석 연구에 자주 쓰임.
# cols = ["motor", "screw", "pgain", "vgain", "class"]  # 모터 위치, 스크류 위치, P게인(비례 이득), V게인(속도 이득), 응답 시간(타깃)
#  - 타깃/피처 (숫자만 사용: pgain, vgain)
#    x = df[["pgain", "vgain"]].astype(float)   
#    y = df["class"].values
#  - 학습/테스트 분할 ( 8:2 )
#  - 스케일링 (StandardScaler)
#  - 다항 특성 (degree=2) + LinearRegression 또는 Ridge 학습
#  - 성능 평가 
#  - 시각화

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 데이터 로드
pd.set_option('display.max_columns', None)
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data",header=None)
df.columns = ['motor', 'screw', 'pgain', 'vgain', 'class']
print(df.head(2))
#   motor screw  pgain  vgain     class
# 0     E     E      5      4  0.281251
# 1     B     D      6      5  0.506252
print(df[['pgain', 'vgain', 'class']].corr())
#           pgain     vgain     class
# pgain  1.000000  0.812268 -0.598129
# vgain  0.812268  1.000000 -0.364383
# class -0.598129 -0.364383  1.000000

# 타깃 / 피처 (숫자만 사용)
x = df[['pgain', 'vgain']].astype(float) # x : pgain, vgain 두 개의 입력 변수 (2차원)
y = df['class'].values # y : 응답 시간 값 (1차원)
print(x.head(3))
#    pgain  vgain
# 0    5.0    4.0
# 1    6.0    5.0
# 2    4.0    3.0
print(y[:3]) # [0.28125095 0.5062525  0.35625148]

# 학습 / 테스트 분할 (8:2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 학습 데이터(x_train, y_train) 대략 80%, 테스트 데이터(x_test, y_test) 대략 20%   8 : 2

# 스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# x_train_scaled, x_test_scaled는 스케일링된 독립변수 데이터

# 다항 특성 (2차)
quad = PolynomialFeatures(degree=2, include_bias=False)
x_train_quad = quad.fit_transform(x_train_scaled)
x_test_quad = quad.transform(x_test_scaled)
print(quad.get_feature_names_out(['pgain', 'vgain']))
print(x_train_quad[:3]) # [[ 1.84967634 ... 1.24881136]]

# 1차 선형회귀 (피처가 두 개이므로 다중선형회귀)
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_lin_pred = model.predict(x_test_scaled)
model_r2 = r2_score(y_test, y_lin_pred)

print('\n1차 선형회귀')
print('예측값 :', y_lin_pred)                             # [2.64610672 ... 2.64610672]
print('실측값 :', y_test)                                 # [4.499986 ... 4.6999955]
print('학습 결정계수 :', model.score(x_train_scaled, y_train))   # 0.365525
print('테스트 결정계수 :', model_r2)                             # 0.511371
print('MAE :', mean_absolute_error(y_test, y_lin_pred))         # 0.933675
print('RMSE :',np.sqrt(mean_squared_error(y_test, y_lin_pred))) # 1.087191

# 다항회귀 (2차)
model_quad = LinearRegression()
model_quad.fit(x_train_quad, y_train)
y_quad_pred = model_quad.predict(x_test_quad)
q_r2 = r2_score(y_test, y_quad_pred)
print('\n2차 다항회귀')
print('예측값 :', y_quad_pred)                              # [3.44493849 ... 3.44493849]
print('실측값 :', y_test)                                   # [4.499986 ... 4.6999955]
print('학습 결정계수 :', model_quad.score(x_train_quad, y_train)) # 0.565527
print('테스트 결정계수 :', q_r2)                                  # 0.511371
print('MAE :', mean_absolute_error(y_test, y_quad_pred))         # 0.630769
print('RMSE :',np.sqrt(mean_squared_error(y_test, y_quad_pred))) # 0.860625

# 시각화
plt.scatter(y_test, y_lin_pred, label='linear fit(d=1), $R^2=%.3f$'%model_r2, lw=2, c='b')
plt.scatter(y_test, y_quad_pred, label='quad fit(d=2), $R^2=%.3f$'%q_r2, lw=2, c='r')
min_value = min(y_test.min(), y_lin_pred.min(), y_quad_pred.min())
max_value = max(y_test.max(), y_lin_pred.max(), y_quad_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle='--', c='k', label='실제값 = 예측값')
plt.xlabel('실제 응답 시간')
plt.ylabel('예측 응답 시간')
plt.title('Servo 선형회귀와 다항회귀 비교')
plt.legend()
plt.show()