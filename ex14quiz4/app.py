# 회귀분석 문제 4) 
# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# Django 또는 Flask로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
# LinearRegression 사용. Ajax 처리!!!      참고: Ajax 처리가 힘들면 그냥 submit()을 해도 됩니다.

import pandas as pd
import pymysql
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

config = {
    'host': '127.0.0.1',
    'user': 'root', 
    'password': '123',
    'database': 'test',
    'port': 3306, 'charset': 'utf8'
}

app = Flask(__name__)

def load_data():
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
    sql = '''
        select jikwonname, jikwonjik, year(now()) - year(jikwonibsail) as 근무년수, jikwonpay
        from jikwon
    '''
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(), columns=['이름', '직급', '근무년수', '연봉'])
    cursor.close()
    conn.close()
    return df


df = load_data()
x = df[['근무년수']].values
y = df['연봉'].values
model = LinearRegression().fit(x, y)
r2 = r2_score(y, model.predict(x))

jik_pay = df.groupby('직급')['연봉'].mean()

@app.get('/')
def index():
    jik_list = [{'jik': jik, 'pay': f'{round(pay):,}'} for jik, pay in jik_pay.items()]
    return render_template('index.html', jik_list=jik_list)


@app.get('/api/predict')
def api_predict():
    year_str = request.args.get('year', '').strip()

    if not year_str.isdigit():
        return jsonify({'ok': False, 'error': '근무년수를 숫자로 입력하세요'}), 400
    year = int(year_str)
    pay = model.predict([[year]])[0]

    return jsonify({
        'ok': True,
        'year': year,
        'pay': round(float(pay)),
        'r2': f'{r2 * 100:.2f}%',
        'formula': f'pay = {model.coef_[0]:.4f} * years + {model.intercept_:.4f}'
    })


if __name__ == '__main__':
    app.run(debug=True)