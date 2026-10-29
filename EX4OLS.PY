# 최소 제곱 문제를 행렬방정식으로 표현하고,
# 수치 선형대수 연산을 통해 최적의 회귀계수(w,d) 구하기

import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

x = np.array([0,1,2,3])
y = np.array([-1, 0.2, 0.5, 2.1])

# plt.scatter(x,y)
# plt.grid(True)
# plt.show()

A = np.vstack([x, np.ones(len(x))]).T
print(A)

# 본래 데이터를 직선으로 표현하기 위해 선형대수학 이용해 y = wx + b의 w,b 구하기
import numpy.linalg as lin
weight, bias = lin.lstsq(A,y)[0]    # 최소제곱범 연산
print(weight, ' ', bias)    # 0.96   -0.98999999999
print()
# 회기식 y^ = 0.96 * x + -0.989999  <= y = wx + b
print(0.96 * 0 + -0.989999) # 실제값 -1  예측값 -0.989999
print(0.96 * 1 + -0.989999) # 실제값 0.2 예측값 -0.029999
print(0.96 * 2 + -0.989999) # 실제값 0.5 예측값  0.930001
print(0.96 * 3 + -0.989999) # 실제값 2.1 예측값  1.890000
print(0.96 * 7 + -0.989999) # 실제값  예측값  5.730001

plt.scatter(x, y, marker='o', label='실제값')
plt.plot(x, weight * x + bias, 'r', label='최적화된 선형직선')
plt.grid(True)
plt.show()

# 경험하지 못한 x값에 대한 예측값 y는?
new_X = 7.5
yhat = weight * new_X + bias    # 0.96 * 7.5 + -0.98999999
print('예측 결과 : ', yhat) # 예측 결과 :  6.21    신빙성 있는 추론된 결과로 미래의 의사결정을 위한 참고자룔 활용 가능

print()
new_X = 1.2345678
yhat = weight * new_X + bias
print('예측 결과 : ', yhat) # 예측 결과 :  0.19518508800000034

print('---------')
# 선형회귀모델 추세식 개선(최소제곱법) 클래스
class MysimpleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # 최소 제곱법 공식 적용
        bunja = np.sum((x - x_mean) * (y - y_mean))
        bunmo = np.sum((x-x_mean) ** 2)
        self.w = bunja / bunmo # 기울기

        self.b = y_mean - (self.w * x_mean) # 절편(편향)

    def predict(self, new_x:np.ndarray):
        # 추정된 w, b를 이용해 새로운 x에 대한 예측 값 반환
        return self.w * new_x + self.b

# 실행 - 임의의 성인 50명의 키, 몸무게 데이터 생성후 이를 바탕으로 추세선을 만들기
np.random.seed(42)
x_height = np.random.normal(175, 5, 50) # 키(평균키, 표준편차, 갯수)
y_weight = np.random.normal(70, 10, 50) # 몸무게
print(x_height)
print(y_weight)

# 모델 생성
model = MysimpleLinearRegression()
model.fit(x_height, y_weight)

print('추정된 w(기울기):', model.w)
print('추정된 b(절편):', model.b)
# yhat = 0.20615129092160 * 키 + 34.3337414765

print()
y_pred = model.predict(x_height)    # 기존 데이터 예측값 확인
# 학습된 회귀계수를 이용해 새로운 키에 대한 몸무게를 추정

for i in range(len(x_height)):
    print(f'키:{x_height[i]:.2f}cm의 실제 몸무게:{y_weight[i]:.2f}kg, 예측몸무게:{y_weight[1]:.2f}kg')

print()
# 새로운 키에 대한 몸무게 예측
new_height = np.array([198])
new_weight = model.predict(new_height)
print(f'키 {new_height}cm인 경우 몸무게는 대략 {new_weight}kg')

# 시각화
plt.figure(figsize=(9,6))
plt.scatter(x_height, y_weight, color='blue', label='실제 데이터')
x_line = np.linspace(x_height.min(),x_height.max(), 100)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=2, label='회귀직선')
plt.xlabel('키(cm)')
plt.ylabel('몸무게(kg)')
plt.title('키에 따른 몸무게 선형회귀 분석')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
