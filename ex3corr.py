# 외국인(미국,일본,중국)이 국내 관광지(5개) 방문 관련자료 사용
# 나라별 관광지 상관관계 확인하기

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 산점도 그리기
def setScatterGraph(tour_table, all_table, tourpoint):
    print('\ntour_table 크기 :', tour_table.shape)
    print('all_table 크기 :', all_table.shape)

    # 현재 관광지 데이터만 추출
    print('관과지명 :', tourpoint)
    
    # 계산할 관광지명에 해당하는 자료만 뽑아 tour에 저장 후 외국인 자료와 병합
    tour = tour_table[tour_table['resNm']== tourpoint]
    # print('관광지 외국인 방문 데이터 건수 : ', len(tour))
    # print('외국인 방문 데이터 건수 : ', len(all_table))
    print(tour)
    관과지명 : 종묘
    # resNm  ForNum
    # yyyymm              
    # 201101    종묘    6837
    
    merge_table = pd.merge(tour, all_table, left_index=True, right_index= True)    
    print(merge_table)
    # resNm  ForNum   china   japan    USA
    # yyyymm                                     
    # 201101    종묘    6837   91252  209184  43065

    # 시각화 - 상관계수
    fig = plt.figure()
    fig.suptitle(tourpoint + '상관관계 분석')

    plt.subplot(1,3,1)
    plt.xlabel('중국인 방문 수')
    plt.ylabel('외국인 입장객 수')
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    print('r1 : ', r1)  # r1 :  -0.5834218986767471
    plt.title('r={:.5f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], alpha=0.7, s=6, c='yellow')

    plt.subplot(1,3,2)
    plt.xlabel('일본인 방문 수')
    plt.ylabel('외국인 입장객 수')
    lamb2 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    print('r2 : ', r2)
    plt.title('r={:.5f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], alpha=0.7, s=6, c='red')

    plt.subplot(1,3,3)
    plt.xlabel('미국인 방문 수')
    plt.ylabel('외국인 입장객 수')
    lamb3 = lambda p:merge_table['USA'].corr(merge_table['ForNum'])
    r3 = lamb3(merge_table)
    print('r3 : ', r3)
    plt.title('r={:.5f}'.format(r3))
    plt.scatter(merge_table['USA'], merge_table['ForNum'], alpha=0.7, s=6, c='blue')

    plt.tight_layout()
    plt.show()

    return [tourpoint, r1, r2, r3]

def processFunc():
    # 서울시 관광지 정보 파일 읽기
    fname = '서울특별시_관광지입장정보_2011_2016.json'
    jsonTP = json.loads(open(fname, mode='r', encoding='utf-8').read())
    # print(jsonTP)
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'resNm', 'ForNum'))    # 년월, 관광지명, 외국인 방문객수
    tour_table = tour_table.set_index('yyyymm')
    print(tour_table[:2])
    resNm = tour_table.resNm.unique()
    # print(resNm[:5])    # ['창덕궁', '운현궁', '경복궁', '창경궁', '종묘']
    
    # 중국인 관광지 정보 파일 읽기
    cdf = '중국인방문객.json'
    jdata = json.loads(open(cdf, mode='r', encoding='utf-8').read())
    china_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    china_table = china_table.rename(columns={'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    print(china_table[:2])
    print()
    # 일본인 관광지 정보 파일 읽기
    jdf = '일본인방문객.json'
    jdata = json.loads(open(jdf, mode='r', encoding='utf-8').read())
    japan_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    japan_table = japan_table.rename(columns={'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    print(japan_table[:2])
    print()
    # 일본인 관광지 정보 파일 읽기
    udf = '미국인방문객.json'
    jdata = json.loads(open(udf, mode='r', encoding='utf-8').read())
    usa_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    usa_table = usa_table.rename(columns={'visit_cnt':'USA'})
    usa_table = usa_table.set_index('yyyymm')
    print(usa_table[:2])

    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True)
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)
    # print(all_table)    # [72 rows x 3 columns]
    #          china   japan    USA
    # yyyymm                       
    # 201101   91252  209184  43065

    r_list = []
    for tourpoint in resNm[:5]:
        r_list.append(setScatterGraph(tour_table, all_table, tourpoint))

    print('r_list : ', r_list) # r_list :  ['창덕궁', (-0.05879110406006312), (0.2774443570141012), (0.40281606330501574)]

    r_df = pd.DataFrame(r_list, columns=('고궁명', '중국', '일본', '미국'))
    r_df = r_df.set_index('고궁명')
    print(r_df)
    #           중국        일본        미국
    # 고궁명                              
    # 창덕궁 -0.058791  0.277444  0.402816
    # 운현궁  0.445945  0.302615  0.281258
    # 경복궁  0.525673 -0.435228  0.425137
    # 창경궁  0.451233 -0.164586  0.624540
    # 종묘   -0.583422  0.529870 -0.121127

    r_df.plot(kind='bar', rot=50)
    plt.show()

if __name__ == '__main__':
    processFunc()

